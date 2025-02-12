# handler.py
import gradio as gr
from ic_light import process_relight, process_normal, run_rmbg, process
import numpy as np
import io
import torch
import safetensors.torch as sf
import os
from PIL import Image
import base64
from io import BytesIO

# Load the model and other resources
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_encoder = ...  # Load your text encoder
vae = ...  # Load your VAE
unet = ...  # Load your UNet
rmbg = ...  # Load your background remover

def process_relight_api(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg = np.array(Image.open(BytesIO(base64.b64decode(input_fg))).convert('RGB'))
    input_bg = np.array(Image.open(BytesIO(base64.b64decode(input_bg))).convert('RGB')) if input_bg else None
    results = process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    return [base64.b64encode(io.BytesIO(img).getvalue()).decode() for img in results]

def process_normal_api(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg = np.array(Image.open(BytesIO(base64.b64decode(input_fg))).convert('RGB'))
    input_bg = np.array(Image.open(BytesIO(base64.b64decode(input_bg))).convert('RGB')) if input_bg else None
    results = process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    return [base64.b64encode(io.BytesIO(img).getvalue()).decode() for img in results]

def run_rmbg_api(input_fg):
    input_fg = np.array(Image.open(BytesIO(base64.b64decode(input_fg))).convert('RGB'))
    result, _ = run_rmbg(input_fg)
    return base64.b64encode(io.BytesIO(result).getvalue()).decode()

def process_api(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg = np.array(Image.open(BytesIO(base64.b64decode(input_fg))).convert('RGB'))
    input_bg = np.array(Image.open(BytesIO(base64.b64decode(input_bg))).convert('RGB')) if input_bg else None
    results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    return [base64.b64encode(io.BytesIO(img).getvalue()).decode() for img in results + extra_images]
