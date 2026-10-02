"""BWM ownership of the Phase-C fixed-embedding alignment and prediction loss."""

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
    align_scale=1.0,
    pred_scale=1.0,
):
    """Attach the frozen PEFM Teacher and Adapter without registering trainable parameters."""
    pipe.stage2_align_scale = float(align_scale)
    pipe.stage2_pred_scale = float(pred_scale)
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

    # 辅助模块加载不改变后续 FM 采样的随机数序列；A/B/C 采用相同精度。
    with torch.random.fork_rng(devices=[]):
        distiller = load_stage2_distiller(
            teacher_bundle=teacher_path,
            adapter_bundle=adapter_path,
            device=pipe.device,
            dtype=torch.float32,
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


def semantic_weight(base_weight, sigma, schedule):
    sigma = sigma.detach().float().mean()
    if schedule == "constant":
        multiplier = torch.ones_like(sigma)
    elif schedule == "one_minus_sigma":
        multiplier = (1.0 - sigma).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown Stage 2 weight schedule: {schedule!r}")
    return multiplier * float(base_weight)


def stage2_embedding_loss(pipe, model_output, inputs, sigma):
    """固定辅助模块的参数，但保留两项损失对 DiT hidden 的梯度。"""
    distiller = pipe.stage2_distiller
    weight = semantic_weight(pipe.stage2_loss_weight, sigma, pipe.stage2_weight_schedule)
    if float(sigma.max()) > distiller.sigma_max:
        zero = model_output["hidden"].float().sum() * 0
        return zero, {"align_mse": zero.detach(), "pred_mse": zero.detach(),
                      "stage2_weight": weight.detach(), "stage2_active": zero.detach()}
    teacher_batch = build_teacher_batch(inputs.get("stage2_rgb"), inputs.get("stage2_eef"))
    with torch.autocast(device_type=model_output["hidden"].device.type, enabled=False):
        result = distiller(hidden=model_output["hidden"], dit_grid=model_output["dit_grid"],
                           valid_len=model_output["valid_len"], teacher_batch=teacher_batch,
                           sigma=sigma, num_views=1)
    loss = weight * (pipe.stage2_align_scale * result["align"] + pipe.stage2_pred_scale * result["pred"])
    metrics = {"align_mse": result["align"].detach(), "pred_mse": result["pred"].detach(),
               "stage2_weight": weight.detach(), "stage2_active": result["active_fraction"].detach()}
    for name in ("align", "pred"):
        start = 3 if name == "align" else 4
        metrics.update({f"{name}_mse_t{start + i:02d}": value.mean().detach()
                        for i, value in enumerate(result[name + "_per_step"].unbind(1))})
    return loss, metrics
