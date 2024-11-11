from abc import ABC, abstractmethod


class Kernel(ABC):
    @staticmethod
    @abstractmethod
    def _check_kernel_params(kernel_params):
        """Check the kernel parameters."""
        pass

    @staticmethod
    @abstractmethod
    def _compute_kernel(x1_lazy, x2_lazy, kernel_params):
        """Compute the kernel between x1 and x2 with given parameters."""
        pass

    @staticmethod
    @abstractmethod
    def _get_row(x_i, x, kernel_params):
        """Returns row of kernel matrix corresponding to
        training point x_i as a dense tensor."""
        pass

    @staticmethod
    @abstractmethod
    def _get_diag(n):
        """Return the diagonal of the kernel matrix."""
        pass

    @staticmethod
    @abstractmethod
    def _get_trace(n):
        """Return the trace of the kernel matrix."""
        pass

    # def __matmul__(self, v):
    #     """Handle the matrix multiplication operator @ for Kernel instances."""
    #     return self.K @ v

    # def __getattr__(self, name):
    #     """Handle attribute access for attributes not explicitly defined."""
    #     if name == "T":
    #         return self.K.T
    #     raise AttributeError(
    #         f"'{self.__class__.__name__}' object has no attribute '{name}'"
    #     )
