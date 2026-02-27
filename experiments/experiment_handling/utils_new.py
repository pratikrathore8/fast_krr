import random
from typing import Union
import warnings

import numpy as np
import torch

from fast_krr.models import FullKRR, InducingKRR
from fast_krr.opts import ASkotch, ASkotchV2, ASkotchV3, EigenPro2, EigenPro3, Mimosa, PCG

OPT_CLASSES = {
    "askotch": ASkotch,
    "askotchv2": ASkotchV2,
    "askotchv3": ASkotchV3,
    "eigenpro2": EigenPro2,
    "eigenpro3": EigenPro3,
    "mimosa": Mimosa,
    "pcg": PCG,
}


# Validation rules for optimizers and preconditioners
VALIDATION_RULES = {
    "askotch": {
        "required": ["b", "beta", "accelerated"],
        "optional": [],
    },
    "askotchv2": {
        "required": ["block_sz_frac", "sampling_method", "accelerated"],
        "optional": [],
    },
    "askotchv3": {
        "required": ["block_sz_frac", "sampling_method"],
        "optional": [],
    },
    "eigenpro2": {
        "required": ["block_sz", "r"],
        "optional": ["bg", "gamma"],
    },
    "eigenpro3": {
        "required": ["block_sz", "r"],
        "optional": ["bg", "proj_inner_iters"],
    },
    "mimosa": {
        "required": ["m", "bg"],
        "optional": ["bH", "bH2"],
    },
    "pcg": {
        "required": ["precond_params"],
        "optional": [],
    },
}

PRECOND_TYPES = {"falkon", "newton", "nystrom", "partial_cholesky"}


def validate_experiment_args(experiment_args):
    """
    Validate the experiment arguments based on optimizer type and preconditioner.
    :param experiment_args: Dictionary of experiment arguments.
    """
    # Validate max_time or max_iter
    if "max_time" not in experiment_args and "max_iter" not in experiment_args:
        raise ValueError("At least one of max_time or max_iter must be provided")

    # Validate optimizer-specific arguments
    opt_type = experiment_args.get("opt")
    if opt_type not in VALIDATION_RULES:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    opt_rules = VALIDATION_RULES[opt_type]
    for required_arg in opt_rules["required"]:
        if required_arg not in experiment_args or experiment_args[required_arg] is None:
            raise ValueError(f"{required_arg} must be provided for {opt_type}")

    for optional_arg in opt_rules["optional"]:
        if optional_arg not in experiment_args or experiment_args[optional_arg] is None:
            warnings.warn(
                f"{optional_arg} is not provided for {opt_type}. Using default."
            )

    # Validate preconditioner parameters if present
    precond_params = experiment_args.get("precond_params")
    if precond_params is not None:
        validate_precond_params(precond_params)


def validate_precond_params(precond_params):
    """
    Validate the preconditioner parameters.
    :param precond_params: Dictionary of preconditioner parameters.
    """
    if precond_params is None:
        return
    precond_type = precond_params.get("type")
    precond_rho = precond_params.get("rho")
    if precond_type is None:
        raise ValueError("Preconditioner type must be provided")
    if precond_type not in PRECOND_TYPES:
        raise ValueError(f"Unsupported preconditioner type: {precond_type}")
    if precond_type != "falkon":
        if not (
            isinstance(precond_rho, float)
            or precond_rho in ["regularization", "damped"]
        ):
            raise ValueError(f"Invalid rho value for {precond_type} preconditioner")


def set_precision(precision):
    if precision == "float32":
        torch.set_default_dtype(torch.float32)
    elif precision == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError("Precision must be either 'float32' or 'float64'")


def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility across NumPy, Python's random module,
    and PyTorch.

    This function ensures that the random number generation is
    consistent and reproducible by setting the same seed across different libraries.
    It also sets the seed for CUDA if a GPU is being used.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_full_krr(Xtr, ytr, Xtst, ytst, kernel_params, Ktr_needed, lambd, task, device):
    w0 = torch.zeros(Xtr.shape[0], device=device)
    return FullKRR(
        x=Xtr,
        b=ytr,
        x_tst=Xtst,
        b_tst=ytst,
        kernel_params=kernel_params,
        Ktr_needed=Ktr_needed,
        lambd=lambd,
        task=task,
        w0=w0,
        device=device,
    )


def get_inducing_krr(
    Xtr, ytr, Xtst, ytst, kernel_params, Knm_needed, m, lambd, task, device
):
    w0 = torch.zeros(m, device=device)
    inducing_pts = torch.randperm(Xtr.shape[0])[:m]
    return InducingKRR(
        x=Xtr,
        b=ytr,
        x_tst=Xtst,
        b_tst=ytst,
        kernel_params=kernel_params,
        Knm_needed=Knm_needed,
        inducing_pts=inducing_pts,
        lambd=lambd,
        task=task,
        w0=w0,
        device=device,
    )


def build_opt_params(model, config):
    if config.opt == "askotch":
        return {
            "model": model,
            "B": config.b,
            "no_store_precond": config.no_store_precond,
            "precond_params": config.precond_params,
            "beta": config.beta,
            "accelerated": config.accelerated,
        }
    elif config.opt == "askotchv2":
        return {
            "model": model,
            "block_sz": config.block_sz,
            "sampling_method": config.sampling_method,
            "precond_params": config.precond_params,
            "mu": config.mu,
            "nu": config.nu,
            "accelerated": config.accelerated,
        }
    elif config.opt == "askotchv3":
        return {
            "model": model,
            "block_sz": config.block_sz,
            "sampling_method": config.sampling_method,
            "precond_params": config.precond_params,
            "accelerated": config.accelerated,
        }
    elif config.opt == "eigenpro2":
        return {
            "model": model,
            "block_sz": config.block_sz,
            "r": config.r,
            "bg": config.bg,
            "gamma": config.gamma,
        }
    elif config.opt == "eigenpro3":
        return {
            "model": model,
            "block_sz": config.block_sz,
            "r": config.r,
            "bg": config.bg,
            "proj_inner_iters": config.proj_inner_iters,
        }
    elif config.opt == "mimosa":
        return {
            "model": model,
            "bg": config.bg,
            "bH": config.bH,
            "bH2": config.bH2,
            "precond_params": config.precond_params,
        }
    elif config.opt == "pcg":
        return {
            "model": model,
            "precond_params": config.precond_params,
        }


def get_opt(model, config):
    # Build the parameter dictionary for the specified optimizer
    opt_params = build_opt_params(model, config)
    # Initialize the optimizer with the specified parameters
    return OPT_CLASSES[config.opt](**opt_params)


def _get_sqrt_dim(X: torch.Tensor) -> float:
    return X.shape[1] ** 0.5


def _get_median_pairwise_dist(X: torch.Tensor) -> float:
    # Compute pairwise distances
    pairwise_distances = torch.cdist(X, X, p=2)  # Pairwise Euclidean distances
    # Extract upper triangle (excluding diagonal) for unique pairwise distances
    distances = pairwise_distances[
        torch.triu_indices(X.shape[0], X.shape[0], offset=1).unbind()
    ]
    # Compute median
    return torch.median(distances).item()


def get_bandwidth(X: torch.Tensor, sigma: Union[str, float], n_pairs: int) -> float:
    if sigma == "sqrt_dim":
        return _get_sqrt_dim(X)
    elif sigma == "median":
        # Subsample to compute median pairwise distance
        n_sub = min(n_pairs, X.shape[0])
        X_sub = X[torch.randperm(X.shape[0])[:n_sub]]
        return _get_median_pairwise_dist(X_sub)
    else:
        return float(sigma)
