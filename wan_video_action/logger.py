import json
import os
import shutil

from diffsynth.diffusion.runner import ModelLogger as DiffSynthModelLogger


class ModelLogger(DiffSynthModelLogger):
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        super().__init__(
            output_path,
            remove_prefix_in_ckpt=remove_prefix_in_ckpt,
            state_dict_converter=state_dict_converter,
        )
        self.epoch_id = 0
        self.batch_in_epoch = 0

    @property
    def latest_state_dir(self):
        return os.path.join(self.output_path, "states", "latest")

    @property
    def latest_state_metadata_path(self):
        return os.path.join(self.output_path, "states", "latest.json")

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
        save_steps=None,
        epoch_id=None,
        batch_in_epoch=None,
        save_full_state=True,
        **kwargs,
    ):
        self.num_steps += 1
        if epoch_id is not None and batch_in_epoch is not None:
            self.set_progress(epoch_id, batch_in_epoch)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            if save_full_state:
                self.save_training_state(accelerator)

    def on_epoch_end(
        self,
        accelerator,
        model,
        epoch_id,
        progress_epoch_id=None,
        save_full_state=True,
    ):
        if progress_epoch_id is not None:
            self.set_progress(progress_epoch_id, 0)
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")
        if save_full_state:
            self.save_training_state(accelerator)

    def on_training_end(self, accelerator, model, save_steps=None, save_full_state=True):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            if save_full_state:
                self.save_training_state(accelerator)

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix_in_ckpt,
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

    def save_training_state(self, accelerator):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and os.path.isdir(self.latest_state_dir):
            shutil.rmtree(self.latest_state_dir)
        accelerator.wait_for_everyone()
        accelerator.save_state(self.latest_state_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metadata = {
                "num_steps": self.num_steps,
                "epoch_id": self.epoch_id,
                "batch_in_epoch": self.batch_in_epoch,
                "state_dir": self.latest_state_dir,
            }
            os.makedirs(os.path.dirname(self.latest_state_metadata_path), exist_ok=True)
            with open(self.latest_state_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")
        accelerator.wait_for_everyone()
