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
from constants import ENTITY_NAME
from constants import TAXI
from base_utils import set_fontsize, render_in_latex
from cfg_utils import get_save_dir, create_krr_config, plot_runs_dataset_grid

# wandb project names
PROJECT_FULL_KRR = "performance_full_krr_"
PROJECT_INDUCING_KRR = "performance_inducing_krr_"

# save directory
SAVE_DIR = "showcase_v2"

# filters for runs
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
    "rho_damped": lambda run: run.config.get("precond_params", {}).get("rho", None)
    == "damped",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
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
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT64_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float64",
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
        PROJECT_INDUCING_KRR, [PCG_FLOAT64_FILTER, EIGENPRO3_FILTER]
    )

    with tqdm(total=2, desc="Showcase") as pbar:
        plot_fn(
            full_krr_cfg=full_krr_cfg_float32,
            inducing_krr_cfg=inducing_krr_cfg_float32,
            datasets_cfg=TAXI,
            name_stem="float32_",
        )
        pbar.update(1)
        plot_fn(
            full_krr_cfg=full_krr_cfg_float64,
            inducing_krr_cfg=inducing_krr_cfg_float64,
            datasets_cfg=TAXI,
            name_stem="float64_",
        )
        pbar.update(1)
