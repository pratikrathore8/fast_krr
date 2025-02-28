import itertools
from pprint import pprint

from data_handling.configs import DATA_CONFIGS, PERFORMANCE_DATASETS
from experiment_handling.configs import (
    KERNEL_CONFIGS,
    LAMBDA_CONFIGS,
    PERFORMANCE_TIME_CONFIGS,
    LOG_TEST_ONLY,
    FALKON_INDUCING_POINTS_GRID,
)
from config_gen.utils import (
    add_kernel_params,
    generate_falkon_configs,
    get_nested_config,
    save_configs,
)

SEED = 0


def generate_pcg_configs(base_config):
    configs = []
    config = base_config.copy()
    config["opt"] = {
        "type": "pcg",
    }
    configs.extend(generate_falkon_configs(config))
    return configs


def generate_combinations(
    sweep_params,
    kernel_configs,
    data_configs,
    lambda_configs,
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
        nested_config["lambd_unscaled"] = lambda_configs[base_config["dataset"]]
        all_combinations.extend(generate_pcg_configs(nested_config))

    return all_combinations


if __name__ == "__main__":
    sweep_params_performance_falkon = {
        "dataset": PERFORMANCE_DATASETS,
        "model": ["inducing_krr"],
        "m": FALKON_INDUCING_POINTS_GRID,
        "opt.type": ["pcg"],
        "training.log_freq": [50],
        "training.precision": ["float32", "float64"],
        "training.seed": [SEED],
        "training.max_iter": [None],
        "wandb.project": ["performance_inducing_krr"],
    }

    output_dir = "performance_inducing_krr"

    combinations = generate_combinations(
        sweep_params_performance_falkon,
        KERNEL_CONFIGS,
        DATA_CONFIGS,
        LAMBDA_CONFIGS,
        PERFORMANCE_TIME_CONFIGS,
        LOG_TEST_ONLY,
    )
    pprint(combinations[:5])  # Debug: Print a sample of generated combinations
    save_configs(combinations, output_dir)
