import torch

from fast_krr.opts.optimizer import Optimizer
from fast_krr.opts.utils.general import _apply_precond
from fast_krr.opts.utils.pcg import _get_precond


class PCG(Optimizer):
    def __init__(self, model, precond_params=None):
        super().__init__(model, precond_params)

        self.rhs = self.model.K_nmTb if self.model.inducing else self.model.b
        self.precond = _get_precond(self.model, self.precond_params)

        self.r, self.z, self.p = self._init_pcg()

    def _init_pcg(self):
        r = self.rhs - self.model.lin_op(self.model.w)
        z = _apply_precond(r, self.precond)
        p = z.clone()
        return r, z, p

    def step(self):
        Kp = self.model.lin_op(self.p)
        r0_dot_z0 = torch.dot(self.r, self.z)
        alpha = r0_dot_z0 / torch.dot(self.p, Kp)
        self.model.w += alpha * self.p
        self.r -= alpha * Kp
        self.z = _apply_precond(self.r, self.precond)
        beta = torch.dot(self.r, self.z) / r0_dot_z0
        self.p = self.z + beta * self.p
