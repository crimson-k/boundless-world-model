import torch
from einops import rearrange


def attach_frozen_evaluator(pipe, evaluator, loss_weight):
    evaluator.to(pipe.device)
    evaluator.requires_grad_(False)
    evaluator.eval()
    pipe.evaluator = evaluator
    pipe.evaluator_loss_weight = float(loss_weight)


def decode_clean_video(pipe, history_latents, future_latents, num_views):
    """Decode clean history and future latents to RGB [B,V,C,F,H,W]."""
    clean_latents = torch.cat([history_latents, future_latents], dim=2)
    num_views = int(num_views)
    if clean_latents.shape[-2] % num_views:
        raise ValueError(
            f"Latent height {clean_latents.shape[-2]} is not divisible by num_views={num_views}."
        )

    batch_size = clean_latents.shape[0]
    latents_by_view = rearrange(
        clean_latents,
        "b c t (v h) w -> (b v) c t h w",
        v=num_views,
    ).to(dtype=pipe.torch_dtype)
    video = pipe.vae.decode(latents_by_view, device=pipe.device)
    return rearrange(
        video,
        "(b v) c t h w -> b v c t h w",
        b=batch_size,
        v=num_views,
    )


def _group_eef(eef, group_ids):
    grouped = eef.new_zeros((eef.shape[0], int(group_ids[-1]) + 1, eef.shape[-1]))
    grouped.index_add_(1, group_ids, eef)
    counts = torch.bincount(group_ids).to(eef).view(1, -1, 1)
    return grouped / counts


def prior_feature_matching_loss(
    evaluator,
    generated_video,
    expert_video,
    eef,
    num_history_frames,
):
    """Return generated/expert prior features and their future-only MSE loss."""
    eef = eef.to(dtype=next(evaluator.parameters()).dtype)
    num_frames = generated_video.shape[3]
    if (num_frames - 1) % 4:
        raise ValueError(f"Expected 1+4k frames, got {num_frames}.")

    group_ids = torch.cat([
        torch.zeros(1, dtype=torch.long, device=eef.device),
        torch.arange(1, (num_frames - 1) // 4 + 1, device=eef.device).repeat_interleave(4),
    ])

    reset = torch.zeros(
        (eef.shape[0], int(group_ids[-1]) + 1),
        dtype=torch.bool,
        device=eef.device,
    )
    reset[:, 0] = True
    shared = {
        "eef": _group_eef(eef, group_ids),
        "group_ids": group_ids,
        "reset": reset,
    }

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(generated_video.device) if generated_video.is_cuda else None
    )
    evaluator.eval()
    generated_prior = evaluator({**shared, "rgb": generated_video})["prior_prediction"]

    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, generated_video.device)
    with torch.no_grad():
        expert_prior = evaluator({**shared, "rgb": expert_video})["prior_prediction"]

    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, generated_video.device)

    history_groups = 1 + (int(num_history_frames) - 1) // 4
    error = (generated_prior - expert_prior).square().mean(-1)
    valid = ~reset
    valid[:, :history_groups] = False
    loss = (error * valid).sum() / valid.sum().clamp_min(1)
    return {
        "loss": loss,
        "generated_prior": generated_prior,
        "expert_prior": expert_prior,
    }
