"""Simple SD3 image generation script."""
import os
import sys
import torch

# Allow running as `python scripts/sample_sd3.py` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
from diffusers import StableDiffusion3Pipeline

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if hf_token and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = hf_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

print(f"Using device: {device}")
print("Loading SD3 model...")

pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=dtype,
    token=hf_token,
)
pipeline.to(device)

# Reduce memory usage
if hasattr(pipeline, "enable_attention_slicing"):
    pipeline.enable_attention_slicing()

print("Model loaded successfully!")

# Sampling parameters
seed = 42
prompt = "Computer Graphics with Neural Networks"
guidance_scale = 7.0
num_inference_steps = 28

out_dir = 'samples_sd3'
os.makedirs(out_dir, exist_ok=True)

print(f"Generating image for prompt: '{prompt}'")

generator = torch.Generator(device=device).manual_seed(seed)
image = pipeline(
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
).images[0]

output_path = f'{out_dir}/tel_aviv_university.png'
image.save(output_path)
print(f"Image saved to: {output_path}")
