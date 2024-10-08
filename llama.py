# from https://github.com/fishaudio/fish-speech

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger

from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer


SEMANTIC_TOKEN = "<|semantic|>"
CODEBOOK_PAD_TOKEN_ID = 0


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    intermediate_size_fast: int = None
    n_local_heads: int = -1
    n_local_heads_fast: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False
    attention_qkv_bias_fast: bool = False

    n_fast_layer: int = 4
    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 8

    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ModelArgs(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self, config, tokenizer, init_weights: bool = True
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.semantic_token_id = tokenizer.convert_tokens_to_ids(SEMANTIC_TOKEN)

        # Slow transformer
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True, typ="base") for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.lm_head = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.dim // config.n_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.self_attn.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                head_dim,
                dtype=dtype,
            )

    def embed(self, x: Tensor) -> Tensor:
        vocab_embeds = [self.embed_tokens(x[:, 0])]
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(x[:, i + 1] + i * self.config.codebook_size)
            emb[x[:, 0] != self.semantic_token_id] = 0
            vocab_embeds.append(emb)

        x = torch.stack(vocab_embeds, dim=3)
        x = x.sum(dim=3)

        return x

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        assert (
            self.max_seq_len != -1 and self.max_batch_size != -1
        ), "Please call setup_caches before forward_generate"

        x = self.embed(x)

        mask = self.causal_mask[
            None, None, input_pos, : self.max_seq_len
        ]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # If prefill, we only calculate the logits of last token
        if x.size(1) > 1 and not return_all:
            x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embed_tokens.weight)
        else:
            token_logits = self.lm_head(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    @staticmethod
    def from_pretrained(
        path: str,
        ckpt_file: str | None = None,
        load_weights: bool = False,
        max_length: int | None = None,
        rope_base: int | None = None,
    ) -> "BaseTransformer":
        config = ModelArgs.from_pretrained(str(path))
        if max_length is not None:
            config.max_seq_len = max_length
            logger.info(f"Override max_seq_len to {max_length}")

        if rope_base is not None:
            config.rope_base = rope_base
            logger.info(f"Override rope_base to {rope_base}")

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = DualARTransformer(config, tokenizer=tokenizer)
       
        ckpt_path = Path(path) / "model.pth"
        weights = torch.load(
            ckpt_path, map_location="cpu", mmap=True
        )

        if "state_dict" in weights:
            weights = weights["state_dict"]

        if next(iter(weights.keys())).startswith("model."):
            new_weights = OrderedDict()
            for k, v in weights.items():
                new_weights[k.replace("model.", "")] = v
            weights = new_weights

        for k, v in model.named_parameters():
            if k not in weights:
                logger.warning(f"No weight for {k}")
            elif v.shape != weights[k].shape:
                logger.warning(
                    f"Shape mismatch for {k}: {v.shape} vs {weights[k].shape}"
                )
        err = model.load_state_dict(weights, strict=True, assign=True)
        logger.info(f"Loaded weights with error: {err}")
        return model


class DualARTransformer(BaseTransformer):
    def __init__(self, config, tokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        # Fast transformer
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.dim)

        # The equivalent bs is so large that sdpa doesn't work
        self.fast_layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=False, typ="fast") for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.dim,
            config.codebook_size,
            bias=False,
        )

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        head_dim = self.config.dim // self.config.n_head

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.self_attn.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.n_local_heads_fast,
                head_dim,
                dtype=dtype,
            )

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        # Fast transformer
        x = x.view(1, 1, -1)

        fast_mask = self.causal_mask[
            None, None, input_pos, : self.config.num_codebooks
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits


class TransformerBlock(nn.Module):
    def __init__(self, config, use_sdpa: bool = True, typ: str="fast") -> None:
        super().__init__()
        self.self_attn = Attention(config, use_sdpa=use_sdpa, typ=typ)
        self.feed_forward = FeedForward(config, typ)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None
    ) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.post_attention_layernorm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config, use_sdpa: bool = True, typ: str="fast"):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.typ = typ

        total_head_dim = (config.n_head + 2 * config.n_local_heads_fast) * config.head_dim

        if typ != "fast":
            n_local_heads = config.n_local_heads
            self.q_proj = nn.Linear(config.dim, config.n_head * config.head_dim, bias=config.attention_qkv_bias)
            self.k_proj = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=config.attention_qkv_bias)
            self.v_proj = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=config.attention_qkv_bias)

            self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        else:
            n_local_heads = config.n_local_heads_fast
            # key, query, value projections for all heads, but in a batch
            self.wqkv = nn.Linear(
                config.dim, total_head_dim, bias=config.attention_qkv_bias_fast
            )

            self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        if self.typ != "fast":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            kv_size = self.n_local_heads * self.head_dim
            q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                        # No third party attn_mask here to use flash_attention
                    )
            else:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            y = self.eq_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        if self.typ != "fast":
            return self.o_proj(y)
        else:
            return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config, typ: str="fast") -> None:
        super().__init__()
        if typ != "fast":
            im_size = config.intermediate_size
        else:
            im_size = config.intermediate_size_fast
        self.gate_proj = nn.Linear(config.dim, im_size, bias=False)
        self.up_proj = nn.Linear(config.dim, im_size, bias=False)
        self.down_proj = nn.Linear(im_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)