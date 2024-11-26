import numpy as np
import torch
from pykeops.torch import LazyTensor

from ..kernels.kernel_inits import _get_kernel, _get_trace
from .model import Model


class InducingKRR(Model):
    def __init__(
        self,
        x,
        b,
        x_tst,
        b_tst,
        kernel_params,
        Knm_needed,
        inducing_pts,
        lambd,
        task,
        w0,
        device,
    ):
        super().__init__(x, b, x_tst, b_tst, kernel_params, lambd, task, w0, device)
        self.inducing_pts = inducing_pts
        self.m = self.inducing_pts.shape[0]
        self.inducing = True

        # Get inducing points kernel
        x_inducing_i = LazyTensor(self.x[self.inducing_pts][:, None, :])
        self.x_inducing_j = LazyTensor(self.x[self.inducing_pts][None, :, :])
        self.K_mm = _get_kernel(x_inducing_i, self.x_inducing_j, self.kernel_params)

        # Get kernel between full training set and inducing points
        if Knm_needed:
            x_i = LazyTensor(self.x[:, None, :])
            self.K_nm = _get_kernel(x_i, self.x_inducing_j, self.kernel_params)
            self.K_nmTb = self.K_nm.T @ self.b  # Useful for computing metrics
            self.K_nmTb_norm = torch.norm(self.K_nmTb)

        # Get kernel for test set
        x_tst_i = LazyTensor(self.x_tst[:, None, :])
        self.K_tst = _get_kernel(x_tst_i, self.x_inducing_j, self.kernel_params)

    def _Knm_lin_op(self, v):
        return self.K_nm @ v

    def _Kmm_lin_op(self, v):
        return self.K_mm @ v

    def lin_op(self, v):
        return self.K_nm.T @ self._Knm_lin_op(v) + self.lambd * self._Kmm_lin_op(v)

    def _compute_train_metrics(self, v):
        K_nmv = self._Knm_lin_op(v)
        K_mmv = self._Kmm_lin_op(v)
        residual = self.K_nm.T @ K_nmv + self.lambd * K_mmv - self.K_nmTb
        rel_residual = torch.norm(residual) / self.K_nmTb_norm
        loss = 1 / 2 * torch.norm(K_nmv - self.b) ** 2 + self.lambd / 2 * torch.dot(
            v, K_mmv
        )

        metrics_dict = {
            "rel_residual": rel_residual,
            "train_loss": loss,
        }

        return metrics_dict

    def _get_grad_regularizer(self, w):
        return self.lambd * (self.K_mm @ w)

    def _get_table_aux(self, idx, w, table):
        x_idx_i = LazyTensor(self.x[idx][:, None, :])
        K_nm_idx = _get_kernel(x_idx_i, self.x_inducing_j, self.kernel_params)
        new_weights = self.n * (K_nm_idx @ w - self.b[idx])
        aux = K_nm_idx.T @ (new_weights - table[idx])
        return new_weights, aux

    def _get_subsampled_lin_ops(self, bH, bH2):
        hess_pts = torch.from_numpy(np.random.choice(self.n, bH, replace=False))
        x_hess_i = LazyTensor(self.x[hess_pts][:, None, :])
        K_sm = _get_kernel(x_hess_i, self.x_inducing_j, self.kernel_params)

        hess_pts_lr = torch.from_numpy(np.random.choice(self.n, bH2, replace=False))
        x_hess_lr_i = LazyTensor(self.x[hess_pts_lr][:, None, :])
        K_sm_lr = _get_kernel(x_hess_lr_i, self.x_inducing_j, self.kernel_params)

        adj_factor = self.n / bH
        adj_factor2 = self.n / bH2

        def K_inducing_sub_lin_op(v):
            return adj_factor * K_sm.T @ (K_sm @ v)

        def K_inducing_sub_Kmm_lin_op(v):
            return adj_factor2 * K_sm_lr.T @ (K_sm_lr @ v) + self.lambd * (
                self.K_mm @ v
            )

        K_inducing_fro_norm2 = torch.sum(
            (K_sm**2).sum() @ torch.ones(1, device=self.device)
        ).item()
        K_inducing_trace = adj_factor * K_inducing_fro_norm2

        return K_inducing_sub_lin_op, K_inducing_sub_Kmm_lin_op, K_inducing_trace

    def _get_Kmm_trace(self):
        return _get_trace(self.m, self.kernel_params)
