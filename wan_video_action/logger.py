import json
import os
import shutil
from contextlib import contextmanager

from diffsynth.diffusion.runner import ModelLogger as DiffSynthModelLogger


class TrainingLogger:
    def __init__(self, accelerator, output_path, args):
        self.accelerator = accelerator
        self.output_path = output_path
        self.args = args
        self.metric_log_path = os.path.join(output_path, "train_metrics.jsonl")
        self.tracker_names = [name for name in ("wandb", "swanlab") if getattr(args, f"use_{name}")]

        if accelerator.is_main_process:
            os.makedirs(output_path, exist_ok=True)
        accelerator.wait_for_everyone()
        metric_log_path = self.metric_log_path if accelerator.is_main_process else os.devnull
        self.metric_log_file = open(metric_log_path, "a", encoding="utf-8")

    @property
    def run_name(self):
        return self.args.run_name or self.output_path

    def init_trackers(self):
        if not self.tracker_names:
            return

        accelerator_tracker_names = {str(tracker) for tracker in self.accelerator.log_with}
        missing_tracker_names = set(self.tracker_names) - accelerator_tracker_names
        if missing_tracker_names:
            raise ValueError(f"Accelerator is missing requested trackers: {sorted(missing_tracker_names)}")

        tracker_init_kwargs = {}
        if self.args.use_wandb:
            tracker_init_kwargs["wandb"] = {"name": self.run_name}
        if self.args.use_swanlab:
            tracker_init_kwargs["swanlab"] = {"name": self.run_name}

        self.accelerator.init_trackers(
            project_name=self.run_name,
            config=vars(self.args),
            init_kwargs=tracker_init_kwargs,
        )

    def print_resolved_config(self, items):
        for name, value in items:
            self.accelerator.print(f"[resolved_config] {name}:", value)

    def log_step(self, metrics, step):
        self.metric_log_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.metric_log_file.flush()

        if self.tracker_names:
            remote_metrics = {
                "train/loss": metrics["loss"],
                "train/lr": metrics["lr"],
                "train/grad_norm": metrics["grad_norm"],
                "train/epoch": metrics["epoch"],
                "train/batch_in_epoch": metrics["batch_in_epoch"],
            }
            self.accelerator.log(remote_metrics, step=step)

    def log_validation(self, val_loss, step):
        metrics = {"step": step, "val/loss": val_loss}
        self.metric_log_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.metric_log_file.flush()
        if self.tracker_names:
            self.accelerator.log({"val/loss": val_loss}, step=step)

    def update_progress_bar(self, progress_bar, postfix):
        progress_bar.set_postfix(postfix)

    def close(self):
        self.metric_log_file.close()
        if self.tracker_names:
            self.accelerator.end_training()


class ModelLogger(DiffSynthModelLogger):
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        super().__init__(
            output_path,
            remove_prefix_in_ckpt=remove_prefix_in_ckpt,
            state_dict_converter=state_dict_converter,
        )
        self.epoch_id = 0
        self.batch_in_epoch = 0
        self._saved_state_steps = set()

    @property
    def state_dir(self):
        return os.path.join(self.output_path, "states", f"step-{self.num_steps}")

    @property
    def state_metadata_path(self):
        return self.state_dir + ".json"

    def state_dict(self):
        return {
            "num_steps": self.num_steps,
            "epoch_id": self.epoch_id,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]
        self.epoch_id = state["epoch_id"]
        self.batch_in_epoch = state["batch_in_epoch"]

    def set_progress(self, epoch_id, batch_in_epoch):
        self.epoch_id = epoch_id
        self.batch_in_epoch = batch_in_epoch

    def on_step_end(
        self,
        accelerator,
        model,
        save_steps,
        epoch_id,
        batch_in_epoch,
    ):
        self.num_steps += 1
        self.set_progress(epoch_id, batch_in_epoch)
        if self.num_steps % save_steps == 0:
            self.save_checkpoint(accelerator, model)

    def on_training_end(self, accelerator, model, save_steps):
        if self.num_steps % save_steps != 0:
            self.save_checkpoint(accelerator, model)

    def save_checkpoint(self, accelerator, model):
        if self.num_steps in self._saved_state_steps:
            accelerator.print(f"Training state for step {self.num_steps} already saved at {self.state_dir}")
            return self.state_dir
        if os.path.exists(self.state_dir) or os.path.exists(self.state_metadata_path):
            raise FileExistsError(f"Refusing to overwrite existing training state: {self.state_dir}")

        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            return self.save_training_state(accelerator, sharded=True)
        else:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            return self.save_training_state(accelerator)

    @contextmanager
    def fsdp_state_dict_type(self, accelerator, state_dict_type=None, offload_to_cpu=None):
        plugin = getattr(accelerator.state, "fsdp_plugin", None)
        if plugin is None or state_dict_type is None:
            yield
            return

        original = (
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        )
        plugin.state_dict_type = state_dict_type
        plugin.state_dict_config = None
        plugin.optim_state_dict_config = None
        plugin.set_state_dict_type()
        if offload_to_cpu is not None:
            plugin.state_dict_config.offload_to_cpu = offload_to_cpu
            plugin.optim_state_dict_config.offload_to_cpu = offload_to_cpu
        try:
            yield
        finally:
            (
                plugin.state_dict_type,
                plugin.state_dict_config,
                plugin.optim_state_dict_config,
            ) = original

    def save_model(self, accelerator, model, file_name):
        path = os.path.join(self.output_path, file_name)
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite existing model checkpoint: {path}")
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix_in_ckpt,
            )
            state_dict = {
                name: value.to(unwrapped_model.pipe.torch_dtype)
                for name, value in state_dict.items()
            }
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            accelerator.save(state_dict, path, safe_serialization=True)

    def save_training_state(self, accelerator, sharded=False):
        state_dir = self.state_dir
        metadata_path = self.state_metadata_path
        if self.num_steps in self._saved_state_steps:
            accelerator.print(f"Training state for step {self.num_steps} already saved at {state_dir}")
            return state_dir
        if os.path.exists(state_dir) or os.path.exists(metadata_path):
            raise FileExistsError(f"Refusing to overwrite existing training state: {state_dir}")

        pending_state_dir = state_dir + ".pending"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and os.path.exists(pending_state_dir):
            shutil.rmtree(pending_state_dir)
        accelerator.wait_for_everyone()
        has_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
        state_dict_type = "SHARDED_STATE_DICT" if sharded and has_fsdp else None
        with self.fsdp_state_dict_type(accelerator, state_dict_type, offload_to_cpu=False):
            accelerator.save_state(pending_state_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.rename(pending_state_dir, state_dir)
            metadata = {
                "num_steps": self.num_steps,
                "epoch_id": self.epoch_id,
                "batch_in_epoch": self.batch_in_epoch,
                "state_dir": state_dir,
                "state_dict_type": state_dict_type or "FULL_STATE_DICT",
            }
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")
        accelerator.wait_for_everyone()
        self._saved_state_steps.add(self.num_steps)
        return state_dir

    def load_training_state(self, accelerator, state_dir):
        sharded_model_dir = os.path.join(state_dir, "pytorch_model_fsdp_0")
        if not os.path.isdir(sharded_model_dir):
            accelerator.load_state(state_dir)
            return

        from torch.distributed.checkpoint import FileSystemReader, load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with self.fsdp_state_dict_type(
            accelerator,
            "SHARDED_STATE_DICT",
            offload_to_cpu=False,
        ):
            optimizers = accelerator._optimizers
            accelerator._optimizers = []
            try:
                accelerator.load_state(state_dir)
            finally:
                accelerator._optimizers = optimizers

            for index, (model, optimizer) in enumerate(zip(accelerator._models, optimizers)):
                optimizer_dir = os.path.join(state_dir, f"optimizer_{index}")
                optimizer_state = load_sharded_optimizer_state_dict(
                    model.state_dict(),
                    optimizer_key="optimizer",
                    storage_reader=FileSystemReader(optimizer_dir),
                )["optimizer"]
                optimizer_state = FSDP.optim_state_dict_to_load(
                    model=model,
                    optim=optimizer,
                    optim_state_dict=optimizer_state,
                )
                optimizer.load_state_dict(optimizer_state)
