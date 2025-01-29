from src.data_configs import DATA_CONFIGS
from src.experiment_configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
    FALKON_INDUCING_POINTS_GRID,
    EIGENPRO3_INDUCING_POINTS_GRID,
)
from src.generate_configs_utils import save_configs
from generate_configs_full_krr import generate_combinations as gc_full_krr
from generate_configs_eigenpro2 import generate_combinations as gc_eigenpro2
from generate_configs_eigenpro3 import generate_combinations as gc_eigenpro3
from generate_configs_falkon import generate_combinations as gc_falkon
from generate_configs_mimosa import generate_combinations as gc_mimosa

SEED = 0

PRECONDITIONERS = ["nystrom"]
CHOLESKY_MODES = [None]
SAMPLING_MODES = ["uniform"]
ACC_MODES = [True]
BG_MODES = [65536]
RHO_MODES = ["damped", "regularization"]
RHO_MODES_MIMOSA = [1e6, 3e6, 1e7, 3e7, 1e8]
BLK_SZ_FRAC = 0.0005

if __name__ == "__main__":
    sweep_params_askotchv2 = {
        "dataset": ["taxi"],
        "model": ["full_krr"],
        "opt.type": ["askotchv2"],
        "precond.r": [50, 100, 200, 500],
        "training.log_freq": [200],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }
    sweep_params_pcg = {
        "dataset": ["taxi"],
        "model": ["full_krr"],
        "opt.type": ["pcg"],
        "precond.r": [50],
        "training.log_freq": [200],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }
    sweep_params_eigenpro2 = {
        "dataset": ["taxi"],
        "model": ["full_krr"],
        "opt.type": ["eigenpro2"],
        "training.log_freq": [200],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr"],
    }
    sweep_params_eigenpro3 = {
        "dataset": ["taxi"],
        "model": ["inducing_krr"],
        "m": EIGENPRO3_INDUCING_POINTS_GRID,
        "opt.type": ["eigenpro3"],
        "training.log_freq": [50],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }
    sweep_params_falkon = {
        "dataset": ["taxi"],
        "model": ["inducing_krr"],
        "m": FALKON_INDUCING_POINTS_GRID,
        "opt.type": ["pcg"],
        "training.log_freq": [200],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }
    sweep_params_mimosa = {
        "dataset": ["taxi"],
        "model": ["inducing_krr"],
        "m": [1_000_000],
        "opt.type": ["mimosa"],
        "precond.r": [500],
        "precond.use_cpu": [False],
        "training.log_freq": [200],
        "training.precision": ["float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }

    combinations_askotchv2 = gc_full_krr(
        sweep_params_askotchv2,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
        RHO_MODES,
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        BLK_SZ_FRAC,
        PRECONDITIONERS,
    )
    combinations_pcg = gc_full_krr(
        sweep_params_pcg,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
        RHO_MODES,
        CHOLESKY_MODES,
        SAMPLING_MODES,
        ACC_MODES,
        BLK_SZ_FRAC,
        PRECONDITIONERS,
    )
    combinations_eigenpro2 = gc_eigenpro2(
        sweep_params_eigenpro2,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    combinations_eigenpro3 = gc_eigenpro3(
        sweep_params_eigenpro3,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    combinations_falkon = gc_falkon(
        sweep_params_falkon,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    combinations_mimosa = gc_mimosa(
        sweep_params_mimosa,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
        RHO_MODES_MIMOSA,
        BG_MODES,
    )

    save_configs(combinations_askotchv2 + combinations_pcg, "taxi_full_krr")
    save_configs(combinations_eigenpro2, "taxi_eigenpro2")
    save_configs(combinations_eigenpro3, "taxi_eigenpro3")
    save_configs(combinations_falkon, "taxi_falkon")
    save_configs(combinations_mimosa, "taxi_mimosa")
