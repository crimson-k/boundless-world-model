import torch
from einops import rearrange
from pefm import load_frozen_evaluator
from torch.utils.checkpoint import checkpoint


def attach_frozen_evaluator(pipe, evaluator, loss_weight):
    evaluator.to(pipe.device)
    evaluator.requires_grad_(False)
    evaluator.eval()
    object.__setattr__(pipe, "evaluator", evaluator)
    pipe.evaluator_loss_weight = float(loss_weight)


def initialize_frozen_evaluator(pipe, evaluator_path, loss_weight):
    pipe.evaluator_loss_weight = float(loss_weight)
    if pipe.evaluator_loss_weight == 0:
        return
    if not evaluator_path:
        raise ValueError("evaluator_path is required when evaluator_loss_weight is nonzero.")

    evaluator = load_frozen_evaluator(
        bundle_path=evaluator_path,
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    attach_frozen_evaluator(pipe, evaluator, pipe.evaluator_loss_weight)


def _decode_checkpointed(pipe, latents):
    if not latents.requires_grad:
        return pipe.vae.single_decode(latents, pipe.device)

    model = pipe.vae.model
    scale = [value.to(latents) for value in pipe.vae.scale]
    latents = latents / scale[1].view(1, model.z_dim, 1, 1, 1)
    latents = latents + scale[0].view(1, model.z_dim, 1, 1, 1)
    hidden = model.conv2(latents)
    model.clear_cache()
    cache = tuple(model._feat_map)
    frames = []

    for index in range(hidden.shape[2]):
        first_chunk = index == 0

        def decode_step(frame, cached, first_chunk=first_chunk):
            cached = list(cached)
            output, cached, _ = model.decoder(
                frame,
                feat_cache=cached,
                feat_idx=[0],
                first_chunk=first_chunk,
            )
            return output, tuple(cached)

        frame, cache = checkpoint(
            decode_step,
            hidden[:, :, index : index + 1],
            cache,
            use_reentrant=False,
        )
        frames.append(frame)

    video = rearrange(
        torch.cat(frames, dim=2),
        "b (c r q) t h w -> b c t (h q) (w r)",
        q=2,
        r=2,
    )
    return video.clamp(-1, 1)


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
    video = torch.cat([_decode_checkpointed(pipe, latents[None]) for latents in latents_by_view])
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
    parameter = next(evaluator.parameters())
    eef = torch.as_tensor(eef, device=generated_video.device, dtype=parameter.dtype)
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

    evaluator.eval()
    generated_prior = checkpoint(
        lambda rgb: evaluator({**shared, "rgb": rgb})["prior_prediction"],
        generated_video,
        use_reentrant=True,
    )

    with torch.no_grad():
        expert_prior = evaluator({**shared, "rgb": expert_video})["prior_prediction"]

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
