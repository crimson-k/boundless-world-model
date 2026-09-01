import warnings

import torch

from .evaluator_loss import decode_clean_video, prior_feature_matching_loss


def FlowMatchSFTLossWanAction(pipe, **inputs):
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
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

    velocity_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if pipe.action_injection_mode == "adaln" and history_t > 0:
        velocity_pred = velocity_pred[:, :, history_t:]
        training_target = training_target[:, :, history_t:]

    flow_loss = torch.nn.functional.mse_loss(velocity_pred.float(), training_target.float())
    flow_loss = flow_loss * pipe.scheduler.training_weight(timestep)

    evaluator = getattr(pipe, "evaluator", None)
    evaluator_weight = float(getattr(pipe, "evaluator_loss_weight", 0.0))
    if evaluator_weight == 0:
        if not getattr(pipe, "_evaluator_zero_weight_warned", False):
            warnings.warn(
                "evaluator_loss_weight=0; evaluator loss is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            pipe._evaluator_zero_weight_warned = True
        return flow_loss
    if evaluator is None:
        raise ValueError("evaluator_loss_weight is nonzero but no evaluator is attached.")

    sigma = pipe.scheduler.sigmas[timestep_id].to(device=pipe.device, dtype=torch.float32)
    future_noise_latents = inputs["latents"][:, :, history_t:]
    pred_clean_latents = future_noise_latents.float() - sigma * velocity_pred.float()
    history_latents = input_latents[:, :, :history_t]
    expert_future_latents = input_latents[:, :, history_t:]

    input_video = inputs.get("input_video")
    precomputed_latents = inputs.get("precomputed_latents")
    num_views = int(
        input_video.shape[0] if input_video is not None else precomputed_latents.shape[0]
    )
    generated_video = decode_clean_video(
        pipe, history_latents, pred_clean_latents, num_views=num_views
    )
    with torch.no_grad():
        expert_video = decode_clean_video(
            pipe, history_latents, expert_future_latents, num_views=num_views
        )
    interaction_loss = prior_feature_matching_loss(
        evaluator=evaluator,
        generated_video=generated_video,
        expert_video=expert_video,
        eef=inputs["action"],
        num_history_frames=inputs["num_history_frames"],
    )["loss"]
    return flow_loss + evaluator_weight * interaction_loss
