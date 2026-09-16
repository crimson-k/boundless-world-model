"""Save at a completed update boundary when a training worker receives SIGUSR1."""

import os
import signal

import torch


class GracefulExit:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.requested = False
        self.batch = 0
        signal.signal(signal.SIGUSR1, self.request)
        print(f"[exit] worker PID={os.getpid()}: kill -USR1 {os.getpid()}", flush=True)

    def request(self, signum, frame):
        # Only set a flag here; saving and distributed collectives belong in the loop.
        self.requested = True

    def sync(self):
        flag = torch.tensor(int(self.requested), device=self.accelerator.device)
        self.requested = bool(self.accelerator.reduce(flag, "sum").item())
        return self.requested

    def save(self, model_logger, split):
        state_dir = model_logger.save_training_state(self.accelerator, sharded=True)
        self.accelerator.print(
            f"[exit] Saved resumable sharded state to {state_dir} after "
            f"{split} batch {self.batch}; stopping."
        )
