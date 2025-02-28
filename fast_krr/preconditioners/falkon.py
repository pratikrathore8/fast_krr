import torch

from fast_krr.preconditioners.preconditioner import Preconditioner


class Falkon(Preconditioner):
    def __init__(self, device, lambd):
        super().__init__(device)
        self.lambd = lambd
        self.T = None
        self.R = None

    def update(self, K_mm_lin_op, K_mm_trace, n, m):
        # Instantiate K_mm as dense tensor
        K_mm = K_mm_lin_op(torch.eye(m, device=self.device))

        # Shift factor for numerical stability
        shift = torch.finfo(K_mm.dtype).eps * K_mm_trace

        # Get preconditioning matrices via Cholesky factorization
        try:
            T = torch.linalg.cholesky(
                K_mm + shift * torch.eye(m).to(self.device), upper=True
            )
            R = torch.linalg.cholesky(
                n / m * (T @ T.T) + self.lambd * torch.eye(m).to(self.device),
                upper=True,
            )
        except (
            RuntimeError
        ) as e:  # Perform calculations on CPU and transfer back if we run out of memory
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                T = torch.linalg.cholesky(
                    K_mm.cpu() + shift * torch.eye(m).cpu(), upper=True
                )
                R = torch.linalg.cholesky(
                    n / m * (T @ T.T) + self.lambd * torch.eye(m).cpu(), upper=True
                )
                T = T.to(self.device)
                R = R.to(self.device)
            else:
                raise e

        self.T = T
        self.R = R

    def inv_lin_op(self, v):
        # Computes T\(R\(R.T\(T.T\v)))
        v = v.reshape(v.shape[0], 1)
        v = torch.linalg.solve_triangular(self.T.T, v, upper=False)
        v = torch.linalg.solve_triangular(self.R.T, v, upper=False)
        v = torch.linalg.solve_triangular(self.R, v, upper=True)
        v = torch.linalg.solve_triangular(self.T, v, upper=True)
        return v.reshape(
            v.shape[0],
        )
