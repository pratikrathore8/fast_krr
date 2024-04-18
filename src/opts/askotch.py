import torch

from .opt_utils_bcd import (
    _get_blocks,
    _get_block_properties,
    _get_block_update,
)


class ASkotch:
    def __init__(self, model, B, beta=0, precond_params=None):
        self.model = model
        self.B = B
        self.beta = beta
        self.precond_params = precond_params

        self.blocks = _get_blocks(self.model.n, self.B)
        self.alpha = (1 - self.beta) / 2  # Controls acceleration
        self.block_preconds, self.block_etas, self.block_Ls = _get_block_properties(
            self.model, self.blocks, self.precond_params
        )
        self.S_alpha = sum([L**self.alpha for L in self.block_Ls])
        self.block_probs = torch.tensor(
            [L**self.alpha / self.S_alpha for L in self.block_Ls]
        )
        self.sampling_dist = torch.distributions.categorical.Categorical(
            self.block_probs
        )
        self.tau = 2 / (1 + (4 * (self.S_alpha**2) / self.model.lambd + 1) ** 0.5)
        self.gamma = 1 / (self.tau * self.S_alpha**2)

        self.y = self.model.w.clone()
        self.z = self.model.w.clone()

    def step(self):
        # Randomly select a block
        block_idx = self.sampling_dist.sample()

        # Get the block, step size, and update direction
        block, eta, dir = _get_block_update(
            self.model,
            self.model.w,
            block_idx,
            self.blocks,
            self.block_preconds,
            self.block_etas,
        )

        # Update y
        self.y = self.model.w.clone()
        self.y[block] -= eta * dir

        # Update z
        self.z = (1 / (1 + self.gamma * self.model.lambd)) * (
            self.z + self.gamma * self.model.lambd * self.model.w
        )
        self.z[block] -= (
            (1 / (1 + self.gamma * self.model.lambd))
            * self.gamma
            / (self.block_probs[block_idx] * (self.block_Ls[block_idx] ** self.beta))
            * dir
        )

        # Update w
        self.model.w = self.tau * self.z + (1 - self.tau) * self.y
