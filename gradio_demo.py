from pathlib import Path
import gradio as gr
import numpy as np
import torch
import time

from generate import load_vqgan, load_llama, decode_one_token, generate_long, wav2codes, normalize


checkpoint_path = Path("./ckpt/")
device = "cuda"
vqm = load_vqgan(checkpoint_path, device)
model = load_llama(checkpoint_path, device)
SR = vqm.spec_transform.sample_rate


def run(user_audio, user_text, inp_type):
    print(">> Input: ", user_audio, user_text, inp_type)

    inp_only_audio = False
    
    if inp_type == "audio":
        inp_only_audio = True
        if user_audio is None:
            raise gr.Error("Audio needed.")
    else:
        if user_audio is None or user_text is None:
            raise gr.Error("Both audio and text needed.")
        
    user_tokens = wav2codes(user_audio, vqm)
    gen = generate_long(
        model=model,
        vqmodel=vqm,
        device=device,
        decode_one_token=decode_one_token,
        max_new_tokens=0,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        prompt_text=user_text,
        prompt_tokens=user_tokens,
        inp_only_audio=inp_only_audio,
    )

    gen_audios = []
    gen_text = ""
    for text, audio in gen:
        gen_text += text
        audio = normalize(audio)
        gen_audios.append(audio)
    print(f"Gen text: {gen_text}")
    out_audio = np.concatenate(gen_audios)
    yield SR, out_audio


with gr.Blocks() as demo:
    gr.Markdown("# Audio/Text Chat")
    with gr.Row():
        with gr.Column(scale=1):
            user_text = gr.Textbox(label="Input", value="你好。", max_lines=3)
            user_audio = gr.Audio(type="filepath", label="Microphone")
            inp_type = gr.Radio(["audio", "audio+text"], value="audio", label="Input Type")
            btn = gr.Button("Run")
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="OutputAudio", autoplay=True)
    btn.click(run, inputs=[user_audio, user_text, inp_type], outputs=[audio_output])

demo.queue().launch(share=False, server_name="0.0.0.0")