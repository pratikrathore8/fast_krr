import itertools

from data_handling.configs import DATA_CONFIGS, PERFORMANCE_DATASETS
from experiment_handling.configs import (
    KERNEL_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
)
from config_gen.utils import (
    add_kernel_params,
    get_nested_config,
    save_configs,
)


SEED = 0
BLOCKSZ = 12_000
R = 100


def generate_eigenpro2_configs(base_config):
    configs = []
    config = base_config.copy()
    config["opt"] = {
        "type": "eigenpro2",
        "block_sz": BLOCKSZ,
        "r": R,
        "bg": None,
        "gamma": None,
    }
    configs.append(config)
    return configs


def generate_combinations(
    sweep_params,
    kernel_configs,
    data_configs,
    performance_time_configs,
    log_test_only,
):
    keys, values = zip(*sweep_params.items())
    base_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    all_combinations = []

    for base_config in base_combinations:
        nested_config = get_nested_config(
            base_config, data_configs, performance_time_configs, log_test_only
        )
        add_kernel_params(nested_config, kernel_configs)
        nested_config["lambd_unscaled"] = 0.0  # EigenPro2 does not use regularization
        all_combinations.extend(generate_eigenpro2_configs(nested_config))

    return all_combinations


if __name__ == "__main__":
    sweep_params_performance_eigenpro2 = {
        "dataset": PERFORMANCE_DATASETS,
        "model": ["full_krr"],
        "opt.type": ["eigenpro2"],
        "training.log_freq": [100],
        "training.precision": ["float32"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_full_krr_v2"],
    }

    output_dir = "performance_full_krr_ep2"

    combinations_eigenpro2 = generate_combinations(
        sweep_params_performance_eigenpro2,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    save_configs(combinations_eigenpro2, output_dir)
