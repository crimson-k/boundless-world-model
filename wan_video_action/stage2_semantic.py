"""用最终向量场恢复的 clean latent 驱动冻结 evaluator，BWM 负责 KL 归约。"""

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
    loss_weight,
    weight_schedule="constant",
):
    """挂载冻结 evaluator；不将其注册到 BWM 的优化器或 FSDP 模型树中。"""
    pipe.stage2_loss_weight = float(loss_weight)
    pipe.stage2_weight_schedule = str(weight_schedule)
    if pipe.stage2_loss_weight == 0:
        object.__setattr__(pipe, "stage2_distiller", None)
        return
    if not teacher_path:
        raise ValueError(
            "stage2_teacher_path is required when stage2_loss_weight is nonzero"
        )

    try:
        from pefm import load_stage2_distiller
    except ImportError as error:
        raise ImportError(
            "PEFM is required for Stage 2. Add the PEFM repository to PYTHONPATH "
            "or install it in the BWM environment."
        ) from error

    distiller = load_stage2_distiller(
        teacher_bundle=teacher_path,
        device=pipe.device,
        dtype=torch.float32,
    )
    object.__setattr__(pipe, "stage2_distiller", distiller)
    object.__setattr__(
        pipe, "stage2_action_delta_stats",
        getattr(distiller.teacher, "action_delta_stats", None),
    )


def build_teacher_batch(clean_latents, raw_eef, time_division_factor=4, action_delta_stats=None):
    """直接使用本次 BWM 输入的 GT VAE latent，并按相同采样帧分组动作。"""
    if clean_latents is None or raw_eef is None:
        raise KeyError("Stage 2 requires GT VAE latents and unnormalized absolute EEF data")
    from pefm.data import normalize_action_delta, relative_eef_actions

    raw_eef = torch.as_tensor(raw_eef, device=clean_latents.device, dtype=torch.float32)
    if raw_eef.ndim == 2:
        raw_eef = raw_eef.unsqueeze(0)
    batch, _, groups, _, _ = clean_latents.shape
    if raw_eef.ndim != 3 or raw_eef.shape[0] != batch or raw_eef.shape[-1] != 14:
        raise ValueError("Expected absolute EEF [B,F,14] matching clean latents batch")
    frames = 1 + (groups - 1) * time_division_factor
    if raw_eef.shape[1] == frames:
        raw_eef = raw_eef[:, ::time_division_factor]
    elif raw_eef.shape[1] != groups:
        raise ValueError(f"Expected {frames} EEF frames or {groups} grouped endpoints")
    return {
        "clean_latents": clean_latents,
        "action_delta": normalize_action_delta(relative_eef_actions(raw_eef), action_delta_stats),
        "reset": (torch.arange(groups, device=clean_latents.device) == 0).expand(batch, -1),
    }


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


def stage2_prior_loss(pipe, clean_latents, inputs, sigma):
    """恢复 latent → 冻结 aggregator/RSSM → 17 步 prior KL；梯度回到 DiT。"""
    distiller = getattr(pipe, "stage2_distiller", None)
    if distiller is None:
        raise ValueError("Stage 2 loss is enabled but no distiller is attached")
    teacher_batch = build_teacher_batch(
        inputs.get("input_latents"),
        inputs.get("stage2_eef"),
        time_division_factor=4,
        action_delta_stats=getattr(pipe, "stage2_action_delta_stats", None),
    )
    if next(distiller.parameters()).device != clean_latents.device:
        distiller.to(device=clean_latents.device)
    # BWM 的 BF16 autocast 不应改变已训练 FP32 evaluator 的计算精度。
    with torch.autocast(device_type=clean_latents.device.type, enabled=False):
        result = distiller(
            clean_latents, teacher_batch["clean_latents"],
            teacher_batch["action_delta"], teacher_batch["reset"],
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
