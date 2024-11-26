from .falkon import Falkon
from .newton import Newton
from .nystrom import Nystrom
from .partial_cholesky import PartialCholesky

PRECOND_CLASSES = {
    "falkon": Falkon,
    "newton": Newton,
    "nystrom": Nystrom,
    "partial_cholesky": PartialCholesky,
}


def _get_precond(precond_params, update_params, lambd, device):
    if precond_params is None:
        return None

    precond_params_sub = {
        k: v for k, v in precond_params.items() if k != "type" and k != "blk_size"
    }
    # the regularization is an input to the preconditioner class
    # to calculate the damping parameter
    precond_params_sub["lambd"] = lambd
    precond = PRECOND_CLASSES[precond_params["type"]](
        device=device, **precond_params_sub
    )
    precond.update(**update_params)
    return precond
