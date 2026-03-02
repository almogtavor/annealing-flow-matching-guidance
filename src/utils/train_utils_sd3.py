"""SD3-specific training utilities for annealing guidance.

Only contains functions that differ from the SDXL path.
Shared functions (get_data_loader, save_model, get_timestep, linear_schedule,
add_noise_to_prompt) are imported from train_utils.
"""
import os
import sys
import glob
import subprocess
import torch
from src.utils.train_utils import add_noise_to_prompt, linear_schedule, get_timestep


def load_models(config, device):
    """Load SD3 pipeline + guidance MLP for training."""
    from src.pipelines.my_pipeline_stable_diffusion3 import MyStableDiffusion3Pipeline
    from src.model.guidance_scale_model import ScalarMLP

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    dtype = torch.float16 if config.get('low_memory', True) else torch.float32

    pipeline = MyStableDiffusion3Pipeline.from_pretrained(
        config['diffusion']['model_id'], torch_dtype=dtype, token=hf_token)
    pipeline.to(device)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    for enc in [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]:
        if enc is not None:
            enc.requires_grad_(False)
    if hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
        pipeline.transformer.enable_gradient_checkpointing()

    model = ScalarMLP(**config['guidance_scale_model'])
    model.to(device, dtype=torch.float32)
    model.device, model.dtype = device, torch.float32
    return pipeline, model


def encode_prompt_sd3(pipeline, prompt):
    """Encode prompts using SD3's three text encoders."""
    device, dtype = pipeline.device, pipeline.transformer.dtype
    pe, npe, ppe, nppe = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    prompt_embeds = torch.cat([npe, pe], dim=0).to(device=device, dtype=dtype)
    pooled = torch.cat([nppe, ppe], dim=0).to(device=device, dtype=dtype)
    return prompt_embeds, pooled


def denoise_single_step_sd3(pipeline, latents, prompt_embeds, pooled, timestep):
    """Single SD3 transformer forward (uncond + cond)."""
    return pipeline.transformer(
        hidden_states=torch.cat([latents] * 2), timestep=torch.cat([timestep] * 2),
        encoder_hidden_states=prompt_embeds, pooled_projections=pooled,
        return_dict=False)[0]


def to_noisy_latents_sd3(pipeline, image, timestep, size=(1024, 1024)):
    """VAE encode + flow-matching noise. Returns (noisy_latents, velocity_gt)."""
    with torch.no_grad():
        vae = pipeline.vae.to(torch.float32)
        image = image.to(device=vae.device, dtype=vae.dtype)
        image = torch.nn.functional.interpolate(image, size=size, mode='bilinear')
        latents = vae.encode(image).latent_dist.sample(generator=None)
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    dt = pipeline.transformer.dtype
    latents = latents.to(dtype=dt)
    noise = torch.randn_like(latents)
    sigma = (timestep.float() / 1000.0).to(dtype=dt, device=latents.device).view(-1, 1, 1, 1)
    noisy_latents = (1 - sigma) * latents + sigma * noise
    return noisy_latents, noise - latents  # (z_t, v_gt = eps - x_0)


def prompt_add_noise_sd3(prompt_embeds, pooled, timestep, n_timesteps,
                         add_noise, noise_scale, rescale, psi, t1, t2):
    """CADS noise on SD3 prompt embeddings (conditional half only)."""
    if add_noise:
        gamma = linear_schedule(timestep / n_timesteps, t1, t2)
        neg, cond = prompt_embeds.chunk(2)
        prompt_embeds = torch.cat([neg, add_noise_to_prompt(cond, gamma, noise_scale, psi, rescale=rescale)])
        neg_p, cond_p = pooled.chunk(2)
        pooled = torch.cat([neg_p, add_noise_to_prompt(cond_p, gamma, noise_scale, psi, rescale=rescale)])
    return prompt_embeds, pooled


def forward_pass(config, pipeline, model, images, prompts):
    """SD3 two-pass training with flow-matching CFG++ and VJP memory split."""
    B = images.size(0)
    dtype = pipeline.transformer.dtype

    l = torch.rand(B).to(pipeline.device)
    timestep = get_timestep(pipeline, batch_size=B)
    noisy_latents, velocity_gt = to_noisy_latents_sd3(pipeline, images, timestep)

    with torch.no_grad():
        pe, ppe = encode_prompt_sd3(pipeline, prompts)
    pe, ppe = prompt_add_noise_sd3(
        pe, ppe, timestep, pipeline.scheduler.config.get('num_train_timesteps', 1000),
        **config['training']['prompt_noise'])

    # Pass 1: SD3 at z_t (frozen)
    with torch.no_grad():
        pred = denoise_single_step_sd3(pipeline, noisy_latents, pe, ppe, timestep)
    vu, vt = pred.float().chunk(2)
    del pred

    w = model(timestep.float(), l, vu, vt)
    v_guided = vu + w * (vt - vu)

    # CFG++ step (flow matching)
    n_steps = config['diffusion'].get('num_timesteps', 50)
    st = (timestep.float() / 1000.0).to(device=noisy_latents.device)
    st1 = (st - 1.0 / n_steps).clamp(min=1e-4)
    st_, st1_ = st.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1)
    zf = noisy_latents.float()
    x0 = zf - st_ * v_guided
    eps_u = zf + (1.0 - st_) * vu
    z_next = (1.0 - st1_) * x0 + st1_ * eps_u

    # Pass 2: VJP split for delta-loss
    t_next = (st1 * 1000.0).to(dtype=timestep.dtype, device=timestep.device)
    del x0, eps_u, zf
    torch.cuda.empty_cache()

    zd = z_next.detach().to(dtype=dtype).requires_grad_(True)
    pred2 = denoise_single_step_sd3(pipeline, zd, pe, ppe, t_next)
    vu2, vt2 = pred2.chunk(2)
    delta = vt2.float() - vu2.float()
    dl_per = (delta ** 2).mean(dim=[1, 2, 3])
    grad_z = torch.autograd.grad((l * dl_per).sum(), zd)[0]
    delta_proxy = (grad_z.float() * z_next).sum() / B

    eps_loss = ((1 - l) * ((v_guided - velocity_gt.float()) ** 2).mean(dim=[1, 2, 3])).mean()
    return eps_loss + delta_proxy


def run_auto_sample(config):
    """Find latest checkpoint and submit sampling SLURM job."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = config['training']['out_dir']
    pts = sorted(glob.glob(os.path.join(out_dir, 'checkpoints_*', 'checkpoint_step_*.pt')),
                 key=os.path.getmtime)
    if not pts:
        print("No checkpoints found for auto-sampling.", flush=True)
        return
    latest = os.path.abspath(pts[-1])
    lr = config.get('training', {}).get('optimizer_kwargs', {}).get('lr')
    if lr is not None:
        ckpt_id = f"sd3_lr{lr}"
    else:
        ckpt_id = os.path.basename(os.path.dirname(latest))
    script = os.path.join(repo, "submit_sd3_sample.sh")
    os.makedirs(os.path.join(repo, "logs", "sampling"), exist_ok=True)
    print(f"\n{'='*60}\nSUBMITTING SAMPLING JOB: {ckpt_id}\n{'='*60}\n", flush=True)
    export_vars = f"ALL,SD3_SAMPLE_CHECKPOINT={latest},SD3_SAMPLE_CHECKPOINT_ID={ckpt_id}"
    result = subprocess.run(
        ["sbatch", "--export", export_vars, script],
        cwd=repo, capture_output=True, text=True)
    print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr.strip()}", flush=True)
