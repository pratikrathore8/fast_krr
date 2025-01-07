import matplotlib.cm as cm

from compressed_log_norm import CompressedLogNorm

# high-level plotting parameters
USE_LATEX = False
FONTSIZE = 18
X_AXIS = "time"
HPARAMS_TO_LABEL = {
    "askotchv2": ["precond", "r", "sampling_method"],
    "pcg": ["precond", "r", "m"],
    "mimosa": ["precond", "r", "m"],
}
BASE_SAVE_DIR = "./plots"
EXTENSION = "pdf"

# colormap for each optimizer
OPT_CMAPS = {
    "askotchv2": cm.get_cmap("Oranges"),
    "skotchv2": cm.get_cmap("Purples"),
    "pcg": cm.get_cmap("Blues"),
    "mimosa": cm.get_cmap("Greys"),
}

# Normalize rank values to colormap
RANK_MIN = 0 + 1  # Minimum rank
RANK_MAX = 500 + 1  # Maximum rank
NORM = CompressedLogNorm(vmin=RANK_MIN, vmax=RANK_MAX, compress_factor=0.4)
FALKON_PLOTTING_RANK = 100  # Dummy rank to use when plotting Falkon

# markers for each preconditioner
PRECOND_MARKERS = {
    "nystrom": {"damped": "o", "regularization": "x"},
    "partial_cholesky": {"greedy": "s", "rpc": "v"},
    "falkon": {10000: "d", 20000: "*", 50000: "p"},
}

# linestyles for each sampling method
SAMPLING_LINESTYLES = {
    "uniform": "solid",
    "rls": "dashed",
}

# number of markers and marker size
TOT_MARKERS = 10
MARKERSIZE = 8

# plotting functions for each metric
METRIC_AX_PLOT_FNS = {
    "rel_residual": "semilogy",
    "train_loss": "plot",
    "test_acc": "plot",
    "test_mse": "plot",
    "test_rmse": "plot",
    "test_mae": "plot",
    "test_smape": "semilogy",
    "rel_suboptim": "semilogy",
}

# labels for metrics, optimizers, and hyperparameters
METRIC_LABELS = {
    "rel_residual": "Relative residual",
    "train_loss": "Training loss",
    "test_acc": "Test accuracy",
    "test_mse": "Test MSE",
    "test_rmse": "Test RMSE",
    "test_mae": "Test MAE",
    "test_smape": "Test SMAPE",
    "rel_suboptim": "Relative suboptimality",
}
OPT_LABELS = {
    "askotchv2": "ASkotch",
    "skotchv2": "Skotch",
    "pcg": "PCG",
    "mimosa": "Mimosa",
}
RANK_LABEL = "r"
RHO_LABEL = "rho"
PRECOND_LABELS = {
    "nystrom": r"Nystr$\ddot{\mathrm{o}}$m",
    "partial_cholesky": "Partial Cholesky",
    "falkon": "Falkon",
}
MODE_LABELS = {
    "greedy": "greedy",
    "rpc": "RPC",
}
RHO_LABELS = {
    "damped": "damped",
    "regularization": "regularization",
}
SAMPLING_LABELS = {
    "uniform": "uniform",
    "rls": "RLS",
}
X_AXIS_LABELS = {
    "time": "Time (s)",
    "datapasses": "Full data passes",
    "iters": "Iterations",
}

# sorting keys for hyperparameters
SORT_KEYS = ["opt", "accelerated", "sampling_method", "precond_type", "r", "m"]

# wandb project names
ENTITY_NAME = "sketchy-opts"
PROJECT_FULL_KRR = "performance_full_krr_v2_"
PROJECT_INDUCING_KRR = "performance_inducing_krr_"

# dataset-specific plotting parameters
VISION = {
    "datasets": {
        "cifar10": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "fashion_mnist": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "mnist": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "svhn": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
    },
    "grid": {"n_rows": 2, "n_cols": 2},
    "name_ext": "vision",
}
PARTICLE_PHYSICS = {
    "datasets": {
        "miniboone": {
            "ylim": [0.6, 1.0],
            "metric": "test_acc",
        },
        "susy": {
            "ylim": [0.6, 0.9],
            "metric": "test_acc",
        },
        "higgs": {
            "ylim": [0.5, 0.8],
            "metric": "test_acc",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 3},
    "name_ext": "particle_physics",
}
TABULAR_CLASSIFICATION = {
    "datasets": {
        "covtype_binary": {
            "ylim": [0.0, 1.0],
            "metric": "test_acc",
        },
        "comet_mc": {
            "ylim": [0.4, 1.0],
            "metric": "test_acc",
        },
        "click_prediction": {
            "ylim": [0.4, 0.9],
            "metric": "test_acc",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 3},
    "name_ext": "tabular_classification",
}
QM9 = {
    "datasets": {
        "qm9": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 1},
    "name_ext": "qm9",
}
MOLECULES_BIG = {
    "datasets": {
        "toluene": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "ethanol": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "benzene": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "malonaldehyde": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 2, "n_cols": 2},
    "name_ext": "molecules_big",
}
MOLECULES_SMALL = {
    "datasets": {
        "uracil": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "aspirin": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "salicylic": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "naphthalene": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 2, "n_cols": 2},
    "name_ext": "molecules_small",
}
TABULAR_REGRESSION = {
    "datasets": {
        "yolanda": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "yearpredictionmsd": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
        "acsincome": {
            "ylim": [0.0, 2.0],
            "metric": "test_smape",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 3},
    "name_ext": "tabular_regression",
}
PERFORMANCE_DATASETS_CFG = [
    VISION,
    PARTICLE_PHYSICS,
    TABULAR_CLASSIFICATION,
    QM9,
    MOLECULES_BIG,
    MOLECULES_SMALL,
    TABULAR_REGRESSION,
]

TAXI = {
    "datasets": {
        "taxi": {
            "ylim": [200.0, 300.0],
            "metric": "test_rmse",
        },
    },
    "grid": {"n_rows": 1, "n_cols": 1},
    "name_ext": "taxi",
}
