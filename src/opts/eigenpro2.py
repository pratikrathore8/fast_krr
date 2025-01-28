from typing import Optional

import torch

from .eigenpro import EigenPro
from .utils.sgd import _get_minibatch


class EigenPro2(EigenPro):
    # Based on https://github.com/EigenPro/EigenPro-pytorch/tree/master
    def __init__(
        self,
        model,
        bg: int,
        block_sz: int,
        r: int,
        gamma: Optional[float] = 0.95,
    ):
        super().__init__(model, bg, block_sz, r)
        self.gamma = gamma
        self._apply_precond, self.eta, self.block = self._setup()
        # print(f"EigenPro2: eta={self.eta}")

    def _setup(self):
        eigvals, eigvecs, beta, tail_eigval, block = self._get_top_eigensys()
        scale = (eigvals[0] / tail_eigval) ** self.gamma
        diag = (1 - torch.pow(tail_eigval / eigvals, self.gamma)) / eigvals

        def _apply_precond(v, kmat):
            return eigvecs @ (diag * (eigvecs.T @ (kmat @ v)))

        new_top_eigval = eigvals[0] / scale
        eta = self._compute_eta(new_top_eigval, beta)

        return _apply_precond, eta, block

    def step(self):
        idx = _get_minibatch(self.generator)
        grad = self.model._get_block_grad(self.model.w, idx)
        self.model.w[idx] -= self.eta * grad
        Kbm = self.K_fn(self.model.x[self.block], self.model.x[idx], get_row=False)
        d = self._apply_precond(grad, Kbm)
        self.model.w[self.block] += self.eta * d
