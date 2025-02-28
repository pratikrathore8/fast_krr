import time
import wandb


class Logger:
    def __init__(self, log_freq):
        self.log_freq = log_freq
        self.start_time = time.time()
        self.iter_time = 0
        self.cum_time = 0

    def reset_timer(self):
        self.start_time = time.time()

    def update_cum_time(self):
        self.iter_time = time.time() - self.start_time
        self.cum_time += self.iter_time

    def compute_log_reset(self, i, compute_fn, *args):
        metrics_dict = {}
        if (i + 1) % self.log_freq == 0:
            metrics_dict = compute_fn(*args)

        wandb.log({"iter_time": self.iter_time} | metrics_dict)

        self.reset_timer()
