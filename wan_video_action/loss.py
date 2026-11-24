import torch

from .stage2_semantic import stage2_prior_loss


def flow_match_forward(pipe, *, timestep_id=None, **inputs):
    """Return the flow loss and clean latents estimated from the final velocity."""
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    if timestep_id is None:
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    else:
        timestep_id = torch.as_tensor(timestep_id, dtype=torch.long).reshape(1)
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs["input_latents"]
    noise = inputs["noise"]

    inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)  # (1 - sigma) * z₀ + sigma * ε
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)  # ε - z₀

    history_t = int(inputs.get("fused_condition_latent_frames") or 0)
    history_condition_latents = inputs.get("history_condition_latents")
    if history_t > 0:
        inputs["latents"][:, :, :history_t] = history_condition_latents[:, :, :history_t]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    velocity_pred = pipe.model_fn(
        **models,
        **inputs,
        timestep=timestep,
    )
    if velocity_pred.shape != input_latents.shape:
        raise ValueError(
            f"Final DiT velocity must match GT VAE latents, got {tuple(velocity_pred.shape)} "
            f"and {tuple(input_latents.shape)}"
        )
    # add_noise 按实际传入的 timestep 查最近项；BF16 舍入后不一定仍是原抽样索引。
    noise_timestep_id = (pipe.scheduler.timesteps - timestep.detach().cpu()).abs().argmin()
    sigma = pipe.scheduler.sigmas[noise_timestep_id].to(device=pipe.device, dtype=torch.float32).reshape(1)
    clean_latents = inputs["latents"].float() - sigma.reshape(-1, 1, 1, 1, 1) * velocity_pred.float()
    if history_t > 0:
        clean_latents = torch.cat(
            [history_condition_latents[:, :, :history_t].float(), clean_latents[:, :, history_t:]],
            dim=2,
        )

    if pipe.action_injection_mode == "adaln" and history_t > 0:
        velocity_pred = velocity_pred[:, :, history_t:]
        training_target = training_target[:, :, history_t:]

    flow_loss = torch.nn.functional.mse_loss(velocity_pred.float(), training_target.float())
    flow_loss = flow_loss * pipe.scheduler.training_weight(timestep)

    return {
        "flow_loss": flow_loss,
        "clean_latents": clean_latents,
        "velocity_pred": velocity_pred,
        "input_latents": input_latents,
        "history_t": history_t,
        "timestep_id": timestep_id,
        "sigma": sigma,
    }


def FlowMatchSFTLossWanAction(pipe, **inputs):
    stage2_enabled = float(getattr(pipe, "stage2_loss_weight", 0.0)) != 0
    output = flow_match_forward(
        pipe,
        **inputs,
    )
    flow_loss = output["flow_loss"]
    sigma = output["sigma"]
    pipe.loss_metrics = {
        "flow_loss": flow_loss.detach(),
        "noise_sigma": sigma.detach().mean(),
    }

    if stage2_enabled:
        prior_kl, stage2_weight, per_step = stage2_prior_loss(
            pipe, output["clean_latents"], inputs, sigma,
        )
        pipe.loss_metrics.update({
            "prior_kl": prior_kl.detach(),
            "stage2_weight": stage2_weight.detach(),
        })
        pipe.loss_metrics.update({
            f"prior_kl_t{logical_step:02d}": per_step[:, offset].mean()
            for offset, logical_step in enumerate(range(4, 4 + per_step.shape[1]))
        })
        return flow_loss + stage2_weight * prior_kl

    return flow_loss
