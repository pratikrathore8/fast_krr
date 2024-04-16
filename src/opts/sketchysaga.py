import torch

from .minibatch_generator import MinibatchGenerator
from .opt_utils_sgd import (
    _get_precond_L_inducing,
    _get_table_aux,
    _apply_precond,
    _get_minibatch,
)


class SketchySAGA:
    def __init__(self, model, bg, bH=None, precond_params=None):
        self.model = model
        self.bg = bg
        self.bH = bH
        self.precond_params = precond_params

    def run(self, max_iter, logger=None):
        logger_enabled = False
        if logger is not None:
            logger_enabled = True

        if logger_enabled:
            logger.reset_timer()

        # Set hyperparameters if not provided
        if self.bH is None:
            self.bH = int(self.model.n**0.5)

        precond, L = _get_precond_L_inducing(self.model, self.bH, self.precond_params)

        eta = 0.5 / L

        table = torch.zeros(self.model.n, device=self.model.device)
        u = torch.zeros(self.model.m, device=self.model.device)  # Running average in SAGA

        if (
            logger_enabled
        ):  # We use K_nmTb instead of b because we are using inducing points
            logger.compute_log_reset(
                self.model.lin_op, self.model.K_tst, self.model.w, self.model.K_nmTb, self.model.b_tst, self.model.b_norm, self.model.task, -1, True
            )

        generator = MinibatchGenerator(self.model.n, self.bg)

        for i in range(max_iter):
            idx = _get_minibatch(generator)

            # Compute the new table weights and aux vector
            new_weights, aux = _get_table_aux(self.model, idx, self.model.w, table)

            g = u + 1 / idx.shape[0] * aux

            u += 1 / self.model.n * aux

            # Update the table at the sampled indices
            table[idx] = new_weights

            # Update parameters, taking regularization into account
            dir = _apply_precond(g + self.model.lambd * (self.model.K_mm @ self.model.w), precond)

            # Update parameters
            self.model.w -= eta * dir

            if logger_enabled:
                logger.compute_log_reset(
                    self.model.lin_op, self.model.K_tst, self.model.w, self.model.K_nmTb, self.model.b_tst, self.model.b_norm, self.model.task, i, True
                )

