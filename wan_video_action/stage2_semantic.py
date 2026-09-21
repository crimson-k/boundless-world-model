"""BWM ownership of the Phase-C V-JEPA--RSSM semantic loss."""

from collections.abc import Mapping

import torch


def noise_sampling_contract(min_timestep_boundary, max_timestep_boundary):
    return {
        "min_timestep_boundary": float(min_timestep_boundary),
        "max_timestep_boundary": float(max_timestep_boundary),
    }


def initialize_stage2_distiller(
    pipe,
    *,
    teacher_path,
    adapter_path,
    loss_weight,
    selected_block=None,
    weight_schedule="constant",
    noise_sampling=None,
):
    """Attach the frozen PEFM Teacher and Adapter without registering trainable parameters."""
    pipe.stage2_loss_weight = float(loss_weight)
    pipe.stage2_weight_schedule = str(weight_schedule)
    if pipe.stage2_loss_weight == 0:
        object.__setattr__(pipe, "stage2_distiller", None)
        pipe.stage2_selected_block = None
        return
    if not teacher_path or not adapter_path:
        raise ValueError(
            "stage2_teacher_path and stage2_adapter_path are required when "
            "stage2_loss_weight is nonzero"
        )

    try:
        from pefm import load_stage2_distiller
    except ImportError as error:
        raise ImportError(
            "PEFM is required for Stage 2. Add the PEFM repository to PYTHONPATH "
            "or install it in the BWM environment."
        ) from error

    bundle = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, Mapping):
        raise ValueError("Stage 2 Adapter bundle must be a mapping")
    bundle_block = int(bundle["selected_block"])
    selected_block = bundle_block if selected_block is None else int(selected_block)
    if selected_block != bundle_block:
        raise ValueError(
            f"Selected DiT block mismatch: config={selected_block}, bundle={bundle_block}"
        )

    distiller = load_stage2_distiller(
        teacher_bundle=teacher_path,
        adapter_bundle=adapter_path,
        device=pipe.device,
        dtype=pipe.torch_dtype,
        dit_dim=int(pipe.dit.dim),
        selected_block=selected_block,
        noise_sampling=noise_sampling,
    )
    object.__setattr__(pipe, "stage2_distiller", distiller)
    pipe.stage2_selected_block = selected_block


def build_teacher_batch(rgb, raw_eef, time_division_factor=4):
    """Create the fixed 21-group Teacher input from raw BWM sample tensors."""
    if rgb is None or raw_eef is None:
        raise KeyError("Stage 2 requires GT video and unnormalized absolute EEF data")
    from pefm.data import stage2_teacher_batch

    rgb = torch.as_tensor(rgb)
    raw_eef = torch.as_tensor(raw_eef, device=rgb.device, dtype=torch.float32)
    return stage2_teacher_batch(rgb, raw_eef, time_division_factor=time_division_factor)


def categorical_prior_kl(target_logits, student_logits):
    """KL(target || student), summed over RSSM variables without free-nats clipping."""
    target_logits = target_logits.detach().float()
    student_logits = student_logits.float()
    target_logprob = torch.log_softmax(target_logits, dim=-1)
    student_logprob = torch.log_softmax(student_logits, dim=-1)
    target_prob = torch.softmax(target_logits, dim=-1)
    return (target_prob * (target_logprob - student_logprob)).sum(-1).sum(-1)


def semantic_weight(base_weight, sigma, schedule):
    sigma = sigma.detach().float().mean()
    if schedule == "constant":
        multiplier = torch.ones_like(sigma)
    elif schedule == "one_minus_sigma":
        multiplier = (1.0 - sigma).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown Stage 2 weight schedule: {schedule!r}")
    return multiplier * float(base_weight)


def stage2_prior_loss(pipe, model_output, inputs, sigma):
    """Run the frozen PEFM path and reduce its 17 paired logits on the BWM side."""
    distiller = getattr(pipe, "stage2_distiller", None)
    if distiller is None:
        raise ValueError("Stage 2 loss is enabled but no distiller is attached")
    teacher_batch = build_teacher_batch(
        inputs.get("stage2_rgb"),
        inputs.get("stage2_eef"),
        time_division_factor=4,
    )
    result = distiller(
        hidden=model_output["hidden"],
        dit_grid=model_output["dit_grid"],
        valid_len=model_output["valid_len"],
        teacher_batch=teacher_batch,
        num_views=1,
        phase="bwm",
    )
    prior = result["prior"]
    per_step = categorical_prior_kl(
        prior["gt_posterior"], prior["generated_prior"],
    )
    loss = per_step.mean()
    weight = semantic_weight(
        pipe.stage2_loss_weight, sigma, pipe.stage2_weight_schedule,
    )
    return loss, weight, per_step.detach()
