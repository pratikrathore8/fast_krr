DATA_DIR = "./data/"
DATA_CONFIGS = {
    "a9a": {
        "tr": "a9a",
        "tst": "a9a.t",
        "loading": "libsvm_multiple",
        "task": "classification",
    },
    "acsincome": {
        "tr": "acsincome_data.pkl",
        "tgt": "acsincome_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "task": "regression",
    },
    "airlines": {
        "tr": "airlines_data.pkl",
        "tgt": "airlines_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "task": "regression",
    },
    "aspirin": {
        "tr": "md17_aspirin.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "benzene": {
        "tr": "md17_benzene2017.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "cadata": {"tr": "cadata", "loading": "libsvm", "split": 0.8, "task": "regression"},
    "cifar10": {
        "tr": "cifar10_data.pkl",
        "tgt": "cifar10_target.pkl",
        "loading": "pkl",
        "split": 50000,
        "shuffle": False,
        "label_map": {i: 1 if i != 0 else -1 for i in range(10)},
        "task": "classification",
    },
    "click_prediction": {
        "tr": "click_prediction_data.pkl",
        "tgt": "click_prediction_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "cod_rna": {
        "tr": "cod-rna",
        "tst": "cod-rna.t",
        "loading": "libsvm_multiple",
        "task": "classification",
    },
    "comet_mc": {
        "tr": "comet_mc_data.pkl",
        "tgt": "comet_mc_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "connect_4": {
        "tr": "connect-4",
        "loading": "libsvm",
        "split": 0.8,
        "label_map": {-1: -1, 0: 1, 1: 1},
        "task": "classification",
    },
    "covtype_binary": {
        "tr": "covtype.libsvm.binary.scale",
        "loading": "libsvm",
        "split": 0.8,
        "label_map": {1: -1, 2: 1},
        "task": "classification",
    },
    "creditcard": {
        "tr": "creditcard_data.pkl",
        "tgt": "creditcard_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "diamonds": {
        "tr": "diamonds_data.pkl",
        "tgt": "diamonds_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "one_hot_features": ["cut", "color", "clarity"],
        "task": "regression",
    },
    "ethanol": {
        "tr": "md17_ethanol.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "fashion_mnist": {
        "tr": "fashionmnist_data.pkl",
        "tgt": "fashionmnist_target.pkl",
        "loading": "pkl",
        "split": 60000,
        "shuffle": False,
        "label_map": {i: 1 if i != 0 else -1 for i in range(10)},
        "task": "classification",
    },
    "higgs": {
        "tr": "HIGGS",
        "loading": "libsvm",
        "split": 10500000,
        "shuffle": False,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "hls4ml": {
        "tr": "hls4ml_data.pkl",
        "tgt": "hls4ml_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {"g": -1, "w": 1, "t": 1, "z": 1, "q": 1},
        "task": "classification",
    },
    "ijcnn1": {
        "tr": "ijcnn1.tr",
        "tst": "ijcnn1.t",
        "loading": "libsvm_multiple",
        "task": "classification",
    },
    "jannis": {
        "tr": "jannis_data.pkl",
        "tgt": "jannis_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "malonaldehyde": {
        "tr": "md17_malonaldehyde.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "medical": {
        "tr": "medical_data.pkl",
        "tgt": "medical_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {0.0: -1, 1.0: 1},
        "task": "classification",
    },
    "miniboone": {
        "tr": "miniboone_data.pkl",
        "tgt": "miniboone_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {"False": -1, "True": 1},
        "task": "classification",
    },
    "mnist": {
        "tr": "mnist_data.pkl",
        "tgt": "mnist_target.pkl",
        "loading": "pkl",
        "split": 60000,
        "shuffle": False,
        "label_map": {i: 1 if i != 0 else -1 for i in range(10)},
        "task": "classification",
    },
    "naphthalene": {
        "tr": "md17_naphthalene.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "phishing": {
        "tr": "phishing",
        "loading": "libsvm",
        "split": 0.8,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "qm9": {
        "tr": "qm9.mat",
        "loading": "mat",
        "split": 100000,
        "shuffle": True,
        "task": "regression",
    },
    "santander": {
        "tr": "santander_data.pkl",
        "tgt": "santander_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {"False": -1, "True": 1},
        "task": "classification",
    },
    "salicylic": {
        "tr": "md17_salicylic.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "sensit_vehicle": {
        "tr": "combined_scale",
        "tst": "combined_scale.t",
        "loading": "libsvm_multiple",
        "label_map": {1: -1, 2: 1, 3: 1},
        "task": "classification",
    },
    "sensorless": {
        "tr": "Sensorless.scale.tr",
        "tst": "Sensorless.scale.val",
        "loading": "libsvm_multiple",
        "label_map": {i: 1 if i != 1 else -1 for i in range(1, 12)},
        "task": "classification",
    },
    "skin_nonskin": {
        "tr": "skin_nonskin",
        "loading": "libsvm",
        "split": 0.8,
        "label_map": {1: -1, 2: 1},
        "task": "classification",
    },
    "susy": {
        "tr": "SUSY",
        "loading": "libsvm",
        "split": 4500000,
        "shuffle": False,
        "label_map": {0: -1, 1: 1},
        "task": "classification",
    },
    "svhn": {
        "tr": "svhn_data.pkl",
        "tgt": "svhn_target.pkl",
        "loading": "pkl",
        "split": 73257,
        "shuffle": False,
        "label_map": {i: 1 if i != 0 else -1 for i in range(10)},
        "task": "classification",
    },
    "taxi": {
        "tr": "taxi-data/subsampled_data.h5py",
        "loading": "h5py",
        "split": 100000000,
        "shuffle": True,
        "task": "regression",
    },
    "toluene": {
        "tr": "md17_toluene.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "uracil": {
        "tr": "md17_uracil.npz",
        "loading": "npz",
        "split": 0.8,
        "task": "regression",
    },
    "volkert": {
        "tr": "volkert_data.pkl",
        "tgt": "volkert_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "label_map": {i: 1 if i != 0 else -1 for i in range(10)},
        "task": "classification",
    },
    "w8a": {
        "tr": "w8a",
        "tst": "w8a.t",
        "loading": "libsvm_multiple",
        "task": "classification",
    },
    "yearpredictionmsd": {
        "tr": "YearPredictionMSD",
        "tst": "YearPredictionMSD.t",
        "loading": "libsvm_multiple",
        "task": "regression",
    },
    "yolanda": {
        "tr": "yolanda_data.pkl",
        "tgt": "yolanda_target.pkl",
        "loading": "pkl",
        "split": 0.8,
        "task": "regression",
    },
}
MOLECULES = [
    "aspirin",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "salicylic",
    "toluene",
    "uracil",
]
DATA_KEYS = list(DATA_CONFIGS.keys()) + ["synthetic"]
SYNTHETIC_NTR = 10000
SYNTHETIC_NTST = 1000
SYNTHETIC_D = 10

KERNEL_CONFIGS = {
    # l1_laplace kernel
    "cifar10": {"type": "l1_laplace", "sigma": 20},
    "fashion_mnist": {"type": "l1_laplace", "sigma": 20},
    "mnist": {"type": "l1_laplace", "sigma": 20},
    "qm9": {"type": "l1_laplace", "sigma": 5120},
    "svhn": {"type": "l1_laplace", "sigma": 20},
    # matern kernel with nu=5/2
    "aspirin": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "benzene": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "ethanol": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "malonaldehyde": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "naphthalene": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "salicylic": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "toluene": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    "uracil": {"type": "matern", "nu": 5 / 2, "sigma": "sqrt_dim"},
    # rbf kernel
    "a9a": {"type": "rbf", "sigma": 8},
    "acsincome": {"type": "rbf", "sigma": "median"},
    "airlines": {"type": "rbf", "sigma": "median"},
    "cadata": {"type": "rbf", "sigma": "median"},
    "click_prediction": {"type": "rbf", "sigma": "median"},
    "cod_rna": {"type": "rbf", "sigma": "median"},
    "comet_mc": {"type": "rbf", "sigma": "median"},
    "connect_4": {"type": "rbf", "sigma": "median"},
    "covtype_binary": {"type": "rbf", "sigma": 0.1},
    "creditcard": {"type": "rbf", "sigma": "median"},
    "diamonds": {"type": "rbf", "sigma": "median"},
    "higgs": {"type": "rbf", "sigma": 3.8},
    "hls4ml": {"type": "rbf", "sigma": "median"},
    "ijcnn1": {"type": "rbf", "sigma": 0.5},
    "jannis": {"type": "rbf", "sigma": "median"},
    "medical": {"type": "rbf", "sigma": "median"},
    "miniboone": {"type": "rbf", "sigma": 5},
    "phishing": {"type": "rbf", "sigma": "median"},
    "santander": {"type": "rbf", "sigma": 7},
    "sensit_vehicle": {"type": "rbf", "sigma": 3},
    "sensorless": {"type": "rbf", "sigma": 0.8},
    "skin_nonskin": {"type": "rbf", "sigma": "median"},
    "susy": {"type": "rbf", "sigma": 3},
    "taxi": {"type": "rbf", "sigma": 1},
    "volkert": {"type": "rbf", "sigma": "median"},
    "w8a": {"type": "rbf", "sigma": "median"},
    "yearpredictionmsd": {"type": "rbf", "sigma": 7},
    "yolanda": {"type": "rbf", "sigma": "median"},
}

LAMBDA_CONFIGS = {
    "a9a": 3.00e-07,
    "acsincome": 1.00e-06,
    "airlines": 1.00e-06,
    "aspirin": 1.00e-09,
    "benzene": 1.00e-09,
    "cadata": 1.00e-06,
    "cifar10": 1.00e-06,
    "click_prediction": 1.00e-06,
    "cod_rna": 1.00e-06,
    "comet_mc": 1.00e-06,
    "connect_4": 1.00e-06,
    "covtype_binary": 3.80e-07,
    "creditcard": 1.00e-06,
    "diamonds": 1.00e-06,
    "ethanol": 1.00e-09,
    "fashion_mnist": 1.00e-06,
    "higgs": 3.00e-08,
    "hls4ml": 1.00e-06,
    "ijcnn1": 1.00e-06,
    "jannis": 1.00e-06,
    "malonaldehyde": 1.00e-09,
    "medical": 1.00e-06,
    "miniboone": 1.00e-07,
    "mnist": 1.00e-06,
    "naphthalene": 1.00e-09,
    "phishing": 1.00e-06,
    "qm9": 1.00e-08,
    "salicylic": 1.00e-09,
    "santander": 1.00e-06,
    "sensit_vehicle": 1.00e-08,
    "sensorless": 1.00e-08,
    "skin_nonskin": 1.00e-06,
    "susy": 1.00e-06,
    "svhn": 1.00e-06,
    "taxi": 2.00e-07,
    "toluene": 1.00e-09,
    "uracil": 1.00e-09,
    "volkert": 1.00e-06,
    "w8a": 1.00e-06,
    "yearpredictionmsd": 2.00e-06,
    "yolanda": 1.00e-06,
}
