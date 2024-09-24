# from https://github.com/fishaudio/fish-speech

import os
import time
from contextlib import nullcontext
from typing import Optional, Tuple, List
from pathlib import Path

import click
from tqdm import tqdm
from loguru import logger

import soundfile as sf
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


from llama import DualARTransformer, CODEBOOK_PAD_TOKEN_ID
from vqgan import FireflyArchitecture


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token(
    model: nn.Module,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)
    
    codebooks = [
        sample(
            x.logits,
            previous_tokens=(
                previous_tokens[0] if previous_tokens is not None else None
            ),  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]

    x = x.hidden_states

    for layer in model.fast_layers:
        layer.self_attn.kv_cache.k_cache.fill_(0)
        layer.self_attn.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor([codebook_idx], device=x.device, dtype=torch.long)
        logits = model.forward_generate_fast(x, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        x = model.fast_embeddings(a)
        codebooks.append(a)
        
    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: nn.Module,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    im_end_id: int = 4,
    decode_one_token=decode_one_token,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with (
            sdpa_kernel([SDPBackend.MATH])
            if torch.cuda.is_available()
            else nullcontext()
        ):
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == im_end_id:
            break

        yield cur_token


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    im_end_id: int = 4,
    decode_one_token=callable,
    **sampling_kwargs,
) -> torch.Tensor:
    T = prompt.size(1)

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1, max_seq_len=T_new, dtype=next(model.parameters()).dtype
        )

    codebook_dim = 1 + model.config.num_codebooks
    empty = torch.empty((codebook_dim, T_new), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = decode_one_token(
        model, prompt.view(1, codebook_dim, -1), input_pos, **sampling_kwargs
    )
    seq[:, T : T + 1] = next_token
    yield next_token.unsqueeze(0)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    gen = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        im_end_id=im_end_id,
        decode_one_token=decode_one_token,
        **sampling_kwargs,
    )

    for token in gen:
        yield token


def encode_tokens(
    tokenizer,
    string,
    device="cuda",
    prompt_tokens=None, # 4,L
    num_codebooks=4,
    inp_only_audio: bool = False,
):
    pre = "<|im_start|>user\n"
    pre_encoded = tokenizer.encode(
        pre,
        add_special_tokens=False,
        truncation=False,
        max_length=10**6,
    )
    if inp_only_audio:
        final_text = pre
    
    else:
        final_text = f"{pre}{string}"

    encoded = tokenizer.encode(
        final_text,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )
    
    prompt_length = len(encoded)
    pre_length = len(pre_encoded)
    semantic_length = prompt_tokens.shape[1]
    
    tokens = (
        encoded
        + [tokenizer.convert_tokens_to_ids("<|semantic|>")] * (semantic_length + pre_length - prompt_length) # additional ph start
        + tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        + tokenizer.convert_tokens_to_ids(["<|im_start|>"])
        + tokenizer.encode("assistant\n", add_special_tokens=False)
    )
    
    tokens = torch.tensor([tokens], dtype=torch.int, device=device)
    seq_len = tokens.shape[1]

    pre_tokens = torch.ones(
        (num_codebooks, pre_length), dtype=torch.int, device=device
    ) * CODEBOOK_PAD_TOKEN_ID
    pad_tokens = torch.ones(
        (num_codebooks, seq_len - pre_length - semantic_length), dtype=torch.int, device=device
    ) * CODEBOOK_PAD_TOKEN_ID

    semantic_tokens = torch.cat([pre_tokens, prompt_tokens + 1, pad_tokens], dim=1)

    prompt = torch.cat((tokens, semantic_tokens), dim=0)
    return prompt


def generate_long(
    *,
    model,
    vqmodel,
    device: str | torch.device,
    decode_one_token: callable,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    prompt_text: Optional[str] = None,
    prompt_tokens: Optional[torch.Tensor] = None,
    inp_only_audio: bool = False,
    chunk_size: int = 0,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    tokenizer = model.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    start = time.perf_counter()
    encoded_prompts = encode_tokens(
        tokenizer,
        string=prompt_text,
        device=device,
        prompt_tokens=prompt_tokens,
        num_codebooks=model.config.num_codebooks,
        inp_only_audio=inp_only_audio,
    )

    prompt_length = encoded_prompts.size(1)

    gen = generate(
        model=model,
        prompt=encoded_prompts,
        max_new_tokens=max_new_tokens,
        im_end_id=im_end_id,
        decode_one_token=decode_one_token,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    gen_tokens = []
    i = 0
    n = 0
    for token in gen:
        if i == 0:
            end = time.perf_counter()
            logger.info(f"First Token: {end - start}")
        gen_tokens.append(token)
        n += 1
        
        if len(gen_tokens) == chunk_size:
            all_tokens = torch.cat(gen_tokens, dim=2)
            y = all_tokens.squeeze(0)
            gen_text = tokenizer.decode(y[0], skip_special_tokens=True)
            codes = y[1:, ] - 1
            audio = codes2wav(codes, vqmodel)
            gen_tokens = []
            yield gen_text, audio
        
        i += 1
    
    if len(gen_tokens):
        all_tokens = torch.cat(gen_tokens, dim=2)
        y = all_tokens.squeeze(0)
        gen_text = tokenizer.decode(y[0], skip_special_tokens=True)
        codes = y[1:, ] - 1
        audio = codes2wav(codes, vqmodel)
        yield gen_text, audio

    
    logger.info(f"generated: {n} tokens")
    

def load_llama(ckpt_path: Path, device):
    model = DualARTransformer.from_pretrained(ckpt_path)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model.eval()


def load_vqgan(ckpt_path: Path, device):
    model = FireflyArchitecture()
    state_dict = torch.load(ckpt_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def wav2codes(wav, model):
    device = model.device
    if isinstance(wav, tuple):
        sr, audio = wav
        audio = audio / 32767
    else:
        audio, sr = librosa.load(wav, sr=model.spec_transform.sample_rate)
    audio = torch.from_numpy(audio.clip(-1, 1)).unsqueeze(0)
    audios = audio[None]
    audio_lengths = torch.tensor([audios.shape[2]], dtype=torch.long)
    indices = model.encode(audios.to(device), audio_lengths.to(device))[0][0]
    return indices


def codes2wav(codes, model):
    device = model.device
    feature_lengths = torch.tensor([codes.shape[1]], device=device)
    audios, _audio_lengths = model.decode(indices=codes[None].to(device), feature_lengths=feature_lengths)
    if device != "cpu":
        audio = audios[0, 0].float().detach().cpu().numpy()
    else:
        audio = audios[0, 0].float().numpy()
    return audio


def normalize(audio):
    audio = audio / np.abs(audio).max()
    audio = audio * 32767
    audio = audio.astype(np.int16)
    audio = audio.reshape(-1, 1)
    return audio



@click.command()
@click.option("--user-text", type=str, default=None)
@click.option("--user-audio", type=click.Path(path_type=Path, exists=True))
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option("--checkpoint-path", type=click.Path(path_type=Path, exists=True), default="ckpt/")
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=42)
def main(
    user_text: Optional[str],
    user_audio: Path,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    seed: int,
) -> None:

    inp_only_audio = False
    vqm = load_vqgan(checkpoint_path, device)
    model = load_llama(checkpoint_path, device)

    user_tokens = wav2codes(user_audio, vqm)
    
    if user_text is None:
        inp_only_audio = True
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    gen = generate_long(
        model=model,
        vqmodel=vqm,
        device=device,
        decode_one_token=decode_one_token,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        prompt_text=user_text,
        prompt_tokens=user_tokens,
        inp_only_audio=inp_only_audio,
        chunk_size=0,
    )
    gen_audios = []
    gen_text = ""
    for text, audio in gen:
        gen_text += text
        audio = normalize(audio)
        gen_audios.append(audio)
    
    out_audio = np.concatenate(gen_audios)
    logger.info(f"OutputText: {gen_text} | audio: {out_audio.shape}")
    
    sf.write("llm_gen.wav", out_audio, vqm.spec_transform.sample_rate)



if __name__ == "__main__":
    main()
