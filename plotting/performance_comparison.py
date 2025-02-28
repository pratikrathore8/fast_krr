from functools import partial

from tqdm import tqdm

from constants import (
    USE_LATEX,
    FONTSIZE,
    X_AXIS,
    HPARAMS_TO_LABEL,
    BASE_SAVE_DIR,
    EXTENSION,
)
from constants import ENTITY_NAME, PROJECT_FULL_KRR, PROJECT_INDUCING_KRR
from constants import PERFORMANCE_DATASETS_CFG
from base_utils import set_fontsize, render_in_latex
from cfg_utils import get_save_dir, create_krr_config, plot_runs_dataset_grid

# save directory
SAVE_DIR = "performance_comparison"

# filters for runs
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
    "rho_damped": lambda run: run.config.get("precond_params", {}).get("rho", None)
    == "damped",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "block_sz_frac": lambda run: run.config["block_sz_frac"] == 0.01,
    "finished": lambda run: run.state == "finished",
}
EIGENPRO2_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "eigenpro2",
    "finished": lambda run: run.state == "finished",
}
EIGENPRO3_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "eigenpro3",
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT32_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float32",
    "not_greedy_cholesky": lambda run: not (
        run.config["precond_params"]["type"] == "partial_cholesky"
        and run.config["precond_params"]["mode"] == "greedy"
    ),
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT64_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float64",
    "not_greedy_cholesky": lambda run: not (
        run.config["precond_params"]["type"] == "partial_cholesky"
        and run.config["precond_params"]["mode"] == "greedy"
    ),
    "finished": lambda run: run.state == "finished",
}


if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    if USE_LATEX:
        render_in_latex()

    plot_fn = partial(
        plot_runs_dataset_grid,
        entity_name=ENTITY_NAME,
        hparams_to_label=HPARAMS_TO_LABEL,
        x_axis=X_AXIS,
        save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
        extension=EXTENSION,
    )

    full_krr_cfg_float32 = create_krr_config(
        PROJECT_FULL_KRR, [ASKOTCH_FILTER, PCG_FLOAT32_FILTER]
    )
    inducing_krr_cfg_float32 = create_krr_config(
        PROJECT_INDUCING_KRR, [PCG_FLOAT32_FILTER]
    )
    full_krr_cfg_float64 = create_krr_config(
        PROJECT_FULL_KRR, [ASKOTCH_FILTER, EIGENPRO2_FILTER, PCG_FLOAT64_FILTER]
    )
    inducing_krr_cfg_float64 = create_krr_config(
        PROJECT_INDUCING_KRR, [EIGENPRO3_FILTER, PCG_FLOAT64_FILTER]
    )

    with tqdm(
        total=2 * len(PERFORMANCE_DATASETS_CFG), desc="Performance comparison"
    ) as pbar:
        for datasets_cfg in PERFORMANCE_DATASETS_CFG:
            plot_fn(
                full_krr_cfg=full_krr_cfg_float32,
                inducing_krr_cfg=inducing_krr_cfg_float32,
                datasets_cfg=datasets_cfg,
                name_stem="float32_",
                keep_largest_m_runs=False,
            )
            pbar.update(1)
            plot_fn(
                full_krr_cfg=full_krr_cfg_float64,
                inducing_krr_cfg=inducing_krr_cfg_float64,
                datasets_cfg=datasets_cfg,
                name_stem="float64_",
            )
            pbar.update(1)
