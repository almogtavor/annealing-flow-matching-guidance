import math
import os
import sys
import time
import torch
import datetime
import tqdm

# Allow running as `python scripts/train.py` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.utils.model_utils as model_utils
import src.utils.train_utils as train_utils
import src.utils.train_utils_sd3 as train_utils_sd3
import src.utils.wandb_utils as wb
import src.utils.resume_utils as resume_utils
import src.utils.ddp_utils as ddp_utils; ddp_utils.setup()


def train(config, pipeline, model, optimizer, dataloader, forward_fn=None, resume_step=0):
    if forward_fn is None:
        forward_fn = forward_pass
    train_config = config['training']
    max_images = train_config.get('max_images')
    if max_images is None:
        max_images = train_config.get('max_steps')
    if max_images is None:
        raise KeyError("training.max_images (or legacy training.max_steps) must be set")
    accumulation_steps = max(train_config.get('accumulation_steps', 1), 1)
    grad_clip = train_config.get('grad_clip', 1.0)
    global_batch_size = train_config.get('batch_size', 1) * int(os.environ.get("WORLD_SIZE", 1))
    max_steps = math.ceil(max_images / global_batch_size)
    max_epochs = math.ceil(max_steps / len(dataloader))

    train_end = False
    global_step = 0
    last_saved_step = 0
    nan_count = 0

    datetime_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if resume_step * global_batch_size >= max_images:
        print(
            f"Resume checkpoint already reached target ({resume_step * global_batch_size} >= {max_images} global images); exiting.",
            flush=True,
        )
        return

    for epoch in range(max_epochs):
        if hasattr(dataloader, 'distributed_sampler'):
            dataloader.distributed_sampler.set_epoch(epoch)
        epoch_start = time.time()
        for batch in tqdm.tqdm(dataloader, miniters=100, mininterval=60):
            model.train()
            prompts, images, image_paths = batch

            images = images.to(pipeline.device)

            if global_step < resume_step:
                global_step += 1
                continue

            completed_step = global_step + 1

            # Update FSG global image counter for delayed FSG start
            import src.utils.fsg_utils as _fsg_mod
            _fsg_mod._fsg_global_images = completed_step * global_batch_size

            result = forward_fn(config, pipeline, model, images, prompts, image_paths=image_paths)
            if isinstance(result, dict):
                loss = result['loss']
                extra_metrics = {k: v for k, v in result.items() if k != 'loss'}
            else:
                loss = result
                extra_metrics = None

            # Skip NaN losses to prevent weight corruption
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count % 10 == 1:
                    print(f"WARNING: NaN/Inf loss at step {completed_step} (total skipped: {nan_count})", flush=True)
                optimizer.zero_grad()
                global_step = completed_step
                continue

            loss = loss / accumulation_steps  # Normalize loss by accumulation steps
            loss.backward()
            wb.log_train(completed_step, loss.item() * accumulation_steps, model, extra_metrics=extra_metrics)

            if completed_step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if completed_step % config['training']['save_interval'] == 0:
                samples_seen = completed_step * global_batch_size
                print(f"Saving model at {samples_seen} global images (iter {completed_step})...")
                resume_utils.save_checkpoint(
                    config,
                    model,
                    optimizer,
                    completed_step,
                    datetime_timestamp,
                    display_step=samples_seen,
                    global_samples_seen=samples_seen,
                )
                last_saved_step = completed_step

            global_step = completed_step
            if completed_step * global_batch_size >= max_images:
                train_end = True
                break

        if train_end:
            break

        # Single marker line between epochs for SLURM logs
        epoch_seconds = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{max_epochs} finished in {epoch_seconds:.1f}s (global_step={global_step})",
            flush=True,
        )

    if global_step > 0 and global_step != last_saved_step:
        samples_seen = global_step * global_batch_size
        print(f"Saving final model at {samples_seen} global images (iter {global_step})...")
        resume_utils.save_checkpoint(
            config,
            model,
            optimizer,
            global_step,
            datetime_timestamp,
            display_step=samples_seen,
            global_samples_seen=samples_seen,
        )


def forward_pass(
    config,
    pipeline,
    model,
    images,
    prompts,
    image_paths=None,
):
    batch_size = images.size(0)
    
    # Select lambda values
    l = torch.rand(batch_size).to(pipeline.unet.device)

    # Select timestep values
    timestep = train_utils.get_timestep(pipeline, batch_size=batch_size)

    # Get noisy latents and ground truth noise
    # x_0 -> z_t
    noisy_latents, noise_gt = train_utils.to_noisy_latents(pipeline, images, timestep) # (z_t, eps)

    # Get prompt embeddings
    with torch.no_grad():
        prompt_embeds, added_cond_kwargs = train_utils.encode_prompt(pipeline, prompts)

    # Use CADS to add noise to conditioning signal (if enabled)
    prompt_embeds, added_cond_kwargs = train_utils.prompt_add_noise(
        prompt_embeds,
        added_cond_kwargs,
        timestep,
        pipeline.scheduler.config['num_train_timesteps'],
        **config['training']['prompt_noise']
    )

    # Predict epsilon_null + epsilon_cond
    noise_pred = train_utils.denoise_single_step(
        pipeline,
        noisy_latents,
        prompt_embeds,
        timestep,
        added_cond_kwargs,
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
    # Predict guidance scale
    # noise_pred, guidance_scale, _ = pipeline.perform_guidance(noise_pred_uncond, noise_pred_text, model.cfg, timestep, model, l=l)
    guidance_scale_pred = model(timestep, l, noise_pred_uncond, noise_pred_text)
    
    # Apply classifier free guidance
    noise_pred = noise_pred_uncond + guidance_scale_pred * (noise_pred_text - noise_pred_uncond)
    
    # Renoise to next latent
    # z_t -> z_{t+1}
    results =  pipeline.scheduler.step(noise_pred, timestep, noisy_latents, return_dict=True, noise_pred_uncond=noise_pred_uncond)
    pred_latents_prev = results['prev_sample']


    # Calculate delta_t_minus_one
    # Reverse scheduler_timesteps for ascending order in search
    scheduler_timesteps = pipeline.scheduler.timesteps.clone().to(device=timestep.device)
    scheduler_timesteps_reversed = scheduler_timesteps.flip(0)

    # Find indices in the reversed array
    timestep_indices = torch.searchsorted(scheduler_timesteps_reversed, timestep, right=True) - 1
    valid_indices = (timestep_indices - 1) >= 0
    if valid_indices.any():
        timestep_indices = timestep_indices[valid_indices]
        timestep_prev = scheduler_timesteps_reversed[timestep_indices - 1]
        pred_latents_prev = pred_latents_prev[valid_indices]
        prompt = [p for p, valid in zip(prompts, valid_indices.cpu().numpy()) if valid]

        with torch.no_grad():
            prompt_embeds, added_cond_kwargs = train_utils.encode_prompt(pipeline, prompt)

        noise_pred_prev = train_utils.denoise_single_step(
            pipeline,
            pred_latents_prev,
            prompt_embeds,
            timestep_prev,
            {k: v[valid_indices] if isinstance(v, torch.Tensor) and v.shape[0] == valid_indices.shape[0] else v
            for k, v in added_cond_kwargs.items()},
        )[0]

        noise_pred_uncond_prev, noise_pred_text_prev = noise_pred_prev.chunk(2, dim=0)
        delta_t_minus_one = noise_pred_uncond_prev - noise_pred_text_prev

    # calc loss
    ema_normalize = config['training'].get('ema_loss_normalization', True)
    loss = train_utils.calc_loss(noise_pred, noise_gt, delta_t_minus_one, l, ema_normalize=ema_normalize)

    return loss


def forward_pass_sd3(
    config,
    pipeline,
    model,
    images,
    prompts,
    image_paths=None,
):
    """SD3 forward pass: flow-matching adaptation of Algorithm 1."""
    B = images.size(0)
    dtype = pipeline.transformer.dtype

    l = torch.rand(B).to(pipeline.device)
    fixed_lam = config['training'].get('fixed_lambda')
    if fixed_lam is not None:
        l = torch.full_like(l, fixed_lam)
    timestep = train_utils.get_timestep(pipeline, batch_size=B)
    noisy_latents, velocity_gt = train_utils_sd3.to_noisy_latents_sd3(pipeline, images, timestep)

    cache_dir = config['training'].get('prompt_cache_dir')
    if cache_dir and image_paths:
        pe, ppe = train_utils_sd3.load_cached_prompt_sd3(
            cache_dir, config['training']['image_root'], image_paths, pipeline.device)
    else:
        with torch.no_grad():
            pe, ppe = train_utils_sd3.encode_prompt_sd3(pipeline, prompts)
    pe, ppe = train_utils_sd3.prompt_add_noise_sd3(
        pe, ppe, timestep, pipeline.scheduler.config.get('num_train_timesteps', 1000),
        **config['training']['prompt_noise'])

    # Pass 1: SD3 at z_t (frozen)
    with torch.no_grad():
        pred = train_utils_sd3.denoise_single_step_sd3(pipeline, noisy_latents, pe, ppe, timestep)
    vu, vt = pred.float().chunk(2)
    del pred

    w = model(timestep.float(), l, vu, vt)
    v_guided = vu + w.view(-1, 1, 1, 1) * (vt - vu)

    use_vanilla_cfg = bool(config['diffusion'].get('vanilla_cfg', 0))
    # Match the original repo: step on the scheduler's discrete timestep grid.
    t_next = train_utils_sd3.get_prev_timestep(pipeline.scheduler, timestep)
    st = (timestep.float() / 1000.0).to(device=noisy_latents.device)
    st1 = (t_next.float() / 1000.0).to(device=noisy_latents.device)
    st_, st1_ = st.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1)
    zf = noisy_latents.float()
    x0 = zf - st_ * v_guided
    eps_u = zf + (1.0 - st_) * (v_guided if use_vanilla_cfg else vu)
    z_next = (1.0 - st1_) * x0 + st1_ * eps_u

    # Pass 2: direct delta loss (full backprop through frozen transformer)
    del x0, eps_u, zf
    torch.cuda.empty_cache()

    pred2 = train_utils_sd3.denoise_single_step_sd3(pipeline, z_next.to(dtype=dtype), pe, ppe, t_next)
    vu2, vt2 = pred2.float().chunk(2)
    delta = vt2 - vu2

    # Same loss structure as SDXL but with velocity instead of noise
    ema_normalize = config['training'].get('ema_loss_normalization', True)
    loss = train_utils.calc_loss(v_guided, velocity_gt.float(), delta, l, ema_normalize=ema_normalize)

    eps_val = ((1 - l) * ((v_guided - velocity_gt.float()) ** 2).mean(dim=[1, 2, 3])).mean()
    diff_val = (l * (delta ** 2).mean(dim=[1, 2, 3])).mean()

    # Delta/velocity diagnostics
    delta_t = (vt - vu)  # delta at current timestep
    delta_norm = delta_t.view(B, -1).norm(dim=1).mean().item()
    delta_next_norm = delta.view(B, -1).norm(dim=1).mean().item()
    return {
        'loss': loss,
        'train/eps_loss': eps_val.item(),
        'train/diff_loss': diff_val.item(),
        'train/delta_norm': delta_norm,
        'train/delta_next_norm': delta_next_norm,
    }


def forward_pass_sc(
    config,
    pipeline,
    model,
    images,
    prompts,
    image_paths=None,
):
    """L2 self-consistency objective (arXiv:2510.00815v1), flow-matching adaptation.

    Sample continuous (sigma_s, sigma_t) with a training gap >= delta_gap (0.1 default),
    noise x_0 with a SHARED eps to both sigma_t (z_t) and sigma_s (z_s_true),
    run one guided Euler step z_t -> z_s_pred, and minimize ||z_s_pred - z_s_true||^2.
    """
    import torch.nn.functional as F

    B = images.size(0)
    dtype = pipeline.transformer.dtype
    device = pipeline.device

    sc = config['training'].get('sc', {})
    delta_gap = float(sc.get('delta_gap', 0.1))
    sigma_min = float(sc.get('sigma_min', 0.0))
    sigma_max = float(sc.get('sigma_max', 0.99))

    # Continuous (sigma_s, sigma_t) with sigma_t - sigma_s >= delta_gap.
    span_s = max(sigma_max - delta_gap - sigma_min, 1e-6)
    u = torch.rand(B, device=device)
    sigma_s = sigma_min + u * span_s
    span_t = torch.clamp(sigma_max - sigma_s - delta_gap, min=0.0)
    v = torch.rand(B, device=device)
    sigma_t = sigma_s + delta_gap + v * span_t
    sigma_s = sigma_s.clamp(0.0, sigma_max - delta_gap)
    sigma_t = torch.maximum(sigma_t, sigma_s + delta_gap).clamp(max=sigma_max)

    # SD3 transformer expects int timesteps in [0, 1000].
    t_step = (sigma_t * 1000.0).round().to(torch.long).clamp(1, 999)

    # VAE encode x_0 in fp32 (mirror to_noisy_latents_sd3).
    with torch.no_grad():
        vae = pipeline.vae.to(torch.float32)
        img = images.to(device=vae.device, dtype=vae.dtype)
        img = F.interpolate(img, size=(1024, 1024), mode='bilinear')
        latents = vae.encode(img).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    latents = latents.to(dtype=dtype)

    # Shared noise -> z_t and true z_s.
    eps = torch.randn_like(latents)
    st = sigma_t.to(dtype=dtype).view(-1, 1, 1, 1)
    ss = sigma_s.to(dtype=dtype).view(-1, 1, 1, 1)
    z_t = (1 - st) * latents + st * eps
    z_s_true = ((1 - ss) * latents + ss * eps).float()

    # Prompt embeddings (no CADS for self-consistency training).
    cache_dir = config['training'].get('prompt_cache_dir')
    if cache_dir and image_paths:
        pe, ppe = train_utils_sd3.load_cached_prompt_sd3(
            cache_dir, config['training']['image_root'], image_paths, device)
    else:
        with torch.no_grad():
            pe, ppe = train_utils_sd3.encode_prompt_sd3(pipeline, prompts)
    # Note: ppe is [uncond; cond] (doubled); c_emb is the cond half.
    # Keep c_emb in fp32 for the MLP; cast prompts to transformer dtype for the transformer call.
    c_emb = ppe[ppe.shape[0] // 2:].float()
    pe = pe.to(dtype=dtype)
    ppe = ppe.to(dtype=dtype)

    # Frozen transformer at t.
    with torch.no_grad():
        pred = train_utils_sd3.denoise_single_step_sd3(pipeline, z_t, pe, ppe, t_step)
    vu, vt = pred.float().chunk(2)
    del pred

    lam = torch.rand(B, device=device)
    fixed_lam = config['training'].get('fixed_lambda')
    if fixed_lam is not None:
        lam = torch.full_like(lam, fixed_lam)

    interval_norm = (sigma_t - sigma_s).float()  # already in [0, 1]
    w = model(t_step.float(), lam, vu, vt, interval=interval_norm, c_emb=c_emb).view(-1, 1, 1, 1)

    v_guided = vu + w * (vt - vu)
    # One guided Euler step from t -> s in flow-matching parameterization.
    z_s_pred = z_t.float() - (st.float() - ss.float()) * v_guided

    loss = ((z_s_pred - z_s_true) ** 2).mean()

    return {
        'loss': loss,
        'train/w_mean': w.detach().mean().item(),
        'train/w_std': w.detach().std(unbiased=False).item() if w.numel() > 1 else 0.0,
        'train/sigma_t_mean': sigma_t.mean().item(),
        'train/sigma_gap_mean': (sigma_t - sigma_s).mean().item(),
    }


if not torch.cuda.is_available():
    print("FATAL: CUDA not available. Refusing to run on CPU.")
    sys.exit(1)
device = torch.device("cuda")

props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"GPU total VRAM (GiB): {props.total_memory / (1024**3):.2f}", flush=True)

config_path = os.environ.get('ANNEALING_GUIDANCE_CONFIG', 'scripts/config.yaml')
_, config = model_utils.load_config(config_path=config_path)
is_sd3 = 'stable-diffusion-3' in config['diffusion']['model_id']
if is_sd3:
    pipeline, guidance_scale_network = train_utils_sd3.load_models(config, device)
else:
    _, pipeline, guidance_scale_network = model_utils.load_models(config_path=config_path, device=device)
guidance_scale_network = ddp_utils.wrap(guidance_scale_network)

# Optional overrides (useful for SLURM sanity-check runs)
_env_max_images = os.environ.get("ANNEALING_GUIDANCE_MAX_IMAGES") or os.environ.get("ANNEALING_GUIDANCE_MAX_STEPS")
if _env_max_images:
    config.setdefault("training", {})
    config["training"]["max_images"] = int(_env_max_images)

_env_save_interval = os.environ.get("ANNEALING_GUIDANCE_SAVE_INTERVAL")
if _env_save_interval:
    config.setdefault("training", {})
    config["training"]["save_interval"] = int(_env_save_interval)

_env_resume_from = os.environ.get("ANNEALING_GUIDANCE_RESUME_FROM")
if _env_resume_from:
    config.setdefault("training", {})
    config["training"]["resume_from"] = _env_resume_from

print("Models/pipeline loaded; building optimizer and dataloader...", flush=True)

optimizer = torch.optim.AdamW(guidance_scale_network.parameters(), **config['training']['optimizer_kwargs'])
dataloader = train_utils.get_data_loader(config)

print(f"Dataloader ready: {len(dataloader)} batches/epoch", flush=True)

resume_step = resume_utils.maybe_resume(config, guidance_scale_network, optimizer)

_objective = config.get('training', {}).get('objective')
if is_sd3 and _objective == 'sc_l2':
    forward_fn = forward_pass_sc
    print("Self-consistency L2 objective enabled (arXiv:2510.00815v1)", flush=True)
elif is_sd3 and config.get('fsg', {}).get('enabled', False):
    from src.utils.fsg_utils import forward_pass_fsg
    forward_fn = forward_pass_fsg
    print("FSG training enabled", flush=True)
else:
    forward_fn = forward_pass_sd3 if is_sd3 else forward_pass
wb.init_training(config, guidance_scale_network, n_samples=len(dataloader))
train(config, pipeline, guidance_scale_network, optimizer, dataloader, forward_fn=forward_fn, resume_step=resume_step)
wb.finish()
if is_sd3:
    train_utils_sd3.run_auto_sample(config)
