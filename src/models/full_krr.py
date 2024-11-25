import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import (
    _get_kernel,
    _get_kernels_start,
    _get_row,
    _get_trace,
    _get_diag,
)
from .model import Model


class FullKRR(Model):
    def __init__(
        self, x, b, x_tst, b_tst, kernel_params, Ktr_needed, lambd, task, w0, device
    ):
        super().__init__(x, b, x_tst, b_tst, kernel_params, lambd, task, w0, device)
        self.inducing = False
        self.x_j, self.K, self.K_tst = _get_kernels_start(
            self.x, self.x_tst, self.kernel_params, Ktr_needed
        )

    def lin_op(self, v):
        return self.K @ v + self.lambd * v

    def _compute_train_metrics(self, v):
        v_lin_op = self.lin_op(v)
        residual = v_lin_op - self.b
        rel_residual = torch.norm(residual) / self.b_norm
        loss = 1 / 2 * torch.dot(v, v_lin_op) - torch.dot(self.b, v)

        metrics_dict = {
            "rel_residual": rel_residual,
            "train_loss": loss,
        }

        return metrics_dict

    def _get_block_grad(self, w, block):
        xb_i = LazyTensor(self.x[block][:, None, :])
        Kbn = _get_kernel(xb_i, self.x_j, self.kernel_params)

        return Kbn @ w + self.lambd * self.w[block] - self.b[block]

    def _get_full_lin_op(self):
        def K_lin_op(v):
            return self.K @ v

        K_trace = _get_trace(self.K.shape[0], self.kernel_params)

        return K_lin_op, K_trace

    def _get_block_lin_ops(self, block):
        xb_i = LazyTensor(self.x[block][:, None, :])
        xb_j = LazyTensor(self.x[block][None, :, :])
        Kb = _get_kernel(xb_i, xb_j, self.kernel_params)

        def Kb_lin_op(v):
            return Kb @ v

        def Kb_lin_op_reg(v):
            return Kb @ v + self.lambd * v

        Kb_trace = _get_trace(Kb.shape[0], self.kernel_params)

        return Kb_lin_op, Kb_lin_op_reg, Kb_trace

    def _get_diag(self, sz=None):
        if sz is None:
            return _get_diag(self.n, self.kernel_params)
        else:
            return _get_diag(sz, self.kernel_params)

    def _get_diag_fn(self):
        def K_diag_fn(n):
            return _get_diag(n, self.kernel_params)

        return K_diag_fn

    def _get_kernel_fn(self):
        def K_fn(x_i, x_j, get_row):
            if get_row:
                return _get_row(x_i, x_j, self.kernel_params)  # Tensor
            else:
                x_i_lz = LazyTensor(x_i[:, None, :])
                x_j_lz = LazyTensor(x_j[None, :, :])
                return _get_kernel(x_i_lz, x_j_lz, self.kernel_params)  # LazyTensor

        return K_fn
