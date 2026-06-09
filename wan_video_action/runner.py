import datetime
import os
import time

import torch
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from tqdm import tqdm


def launch_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    learning_rate=1e-5,
    weight_decay=1e-2,
    num_workers=1,
    save_steps=None,
    num_epochs=1,
    max_train_steps=None,
    max_grad_norm=1.0,
    args=None,
):
    resume_from = None
    ckpt_path = None
    dataset_repeat = 1
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_train_steps = args.max_train_steps
        max_grad_norm = args.max_grad_norm
        resume_from = args.resume_from
        ckpt_path = args.ckpt_path
        dataset_repeat = args.dataset_repeat
    if resume_from and ckpt_path:
        raise ValueError("`--resume_from` and `--ckpt_path` are mutually exclusive. Use only one of them.")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader_kwargs = {
        "shuffle": False,
        "collate_fn": lambda x: x[0],
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 64
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    if args is not None and len(getattr(accelerator, "log_with", [])) > 0:
        tracker_init_kwargs = {}
        if args.use_wandb:
            tracker_init_kwargs["wandb"] = {"name": args.output_path}
        if args.use_swanlab:
            experiment_name = args.swanlab_experiment_name or args.output_path
            tracker_init_kwargs["swanlab"] = {"experiment_name": experiment_name}
        if tracker_init_kwargs:
            accelerator.init_trackers(
                project_name="DiffSynth-Studio",
                config=vars(args),
                init_kwargs=tracker_init_kwargs,
            )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    start_epoch = 0
    if resume_from:
        accelerator.print(f"Resuming training from checkpoint: {resume_from}")
        accelerator.load_state(resume_from)
        basename = os.path.basename(os.path.normpath(resume_from))
        if basename.startswith("epoch-"):
            try:
                resume_epoch_label = int(basename.split("-", 1)[1])
                if dataset_repeat > 1 and (resume_epoch_label + 1) % dataset_repeat == 0:
                    start_epoch = (resume_epoch_label + 1) // dataset_repeat
                else:
                    start_epoch = resume_epoch_label + 1
            except ValueError:
                start_epoch = 0
        state_step = getattr(accelerator.state, "step", None)
        if state_step is not None:
            model_logger.num_steps = state_step

    epoch_id = start_epoch
    last_step_time = time.monotonic()
    while max_train_steps is not None or epoch_id < num_epochs:
        if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
            break
        train_loss = 0.0
        loss_count = 0
        optimizer.zero_grad()
        epoch_label = (epoch_id + 1) * dataset_repeat - 1

        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for data in progress_bar:
            with accelerator.accumulate(model):
                if getattr(dataset, "load_from_cache", False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                train_loss += loss.detach().float().item()
                loss_count += 1
                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients and max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.item()

                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    now = time.monotonic()
                    step_time = now - last_step_time
                    last_step_time = now

                    avg_loss = train_loss / loss_count
                    model_logger.on_step_end(
                        accelerator,
                        model,
                        save_steps,
                        loss=avg_loss,
                        grad_norm=grad_norm,
                        optimizer=optimizer,
                        epoch=epoch_label,
                        force_step=True,
                    )
                    train_loss = 0.0
                    loss_count = 0

                    if save_steps is not None:
                        next_save_steps = save_steps - (model_logger.num_steps % save_steps)
                        if next_save_steps == 0:
                            next_save_steps = save_steps
                        eta_steps = next_save_steps
                        eta_label = "next_save_eta"
                        if max_train_steps is not None:
                            remaining_train_steps = max_train_steps - model_logger.num_steps
                            if remaining_train_steps < eta_steps:
                                eta_steps = remaining_train_steps
                                eta_label = "train_end_eta"
                        progress_bar.set_postfix({
                            "step": model_logger.num_steps,
                            "next_save_steps": next_save_steps,
                            eta_label: str(datetime.timedelta(seconds=max(0, int(step_time * eta_steps)))),
                        })

                    if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                        break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_label)
        epoch_id += 1
    model_logger.on_training_end(accelerator, model, save_steps)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
