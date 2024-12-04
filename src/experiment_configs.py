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
    "synthetic": {"type": "rbf", "sigma": 1},
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
    "synthetic": 1.00e-06,
    "taxi": 2.00e-07,
    "toluene": 1.00e-09,
    "uracil": 1.00e-09,
    "volkert": 1.00e-06,
    "w8a": 1.00e-06,
    "yearpredictionmsd": 2.00e-06,
    "yolanda": 1.00e-06,
}

PERFORMANCE_TIME_CONFIGS = {
    "phishing": 1800,
    "cadata": 1800,
    "a9a": 1800,
    "sensorless": 1800,
    "w8a": 1800,
    "ijcnn1": 1800,
    "diamonds": 1800,
    "jannis": 1800,
    "volkert": 1800,
    "cod_rna": 1800,
    "cifar10": 1800,
    "medical": 1800,
    "connect_4": 1800,
    "fashion_mnist": 1800,
    "mnist": 1800,
    "sensit_vehicle": 1800,
    "svhn": 1800,
    "synthetic": 1800,
    "miniboone": 3600,
    "qm9": 3600,
    "uracil": 3600,
    "santander": 3600,
    "aspirin": 3600,
    "skin_nonskin": 3600,
    "creditcard": 3600,
    "salicylic": 3600,
    "naphthalene": 3600,
    "yolanda": 3600,
    "toluene": 3600,
    "yearpredictionmsd": 3600,
    "ethanol": 7200,
    "covtype_binary": 7200,
    "benzene": 7200,
    "comet_mc": 7200,
    "hls4ml": 7200,
    "malonaldehyde": 7200,
    "airlines": 10800,
    "acsincome": 10800,
    "click_prediction": 10800,
    "susy": 10800,
    "higgs": 10800,
    "taxi": 43200,
}

LOG_TEST_ONLY = {
    "phishing": False,
    "cadata": False,
    "a9a": False,
    "sensorless": False,
    "w8a": False,
    "ijcnn1": False,
    "diamonds": False,
    "jannis": False,
    "volkert": False,
    "cod_rna": False,
    "cifar10": False,
    "medical": False,
    "connect_4": False,
    "fashion_mnist": False,
    "mnist": False,
    "sensit_vehicle": False,
    "svhn": False,
    "synthetic": False,
    "miniboone": False,
    "qm9": False,
    "uracil": False,
    "santander": False,
    "aspirin": False,
    "skin_nonskin": False,
    "creditcard": False,
    "salicylic": False,
    "naphthalene": False,
    "yolanda": False,
    "toluene": False,
    "yearpredictionmsd": False,
    "ethanol": False,
    "covtype_binary": False,
    "benzene": False,
    "comet_mc": False,
    "hls4ml": False,
    "malonaldehyde": False,
    "airlines": False,
    "acsincome": False,
    "click_prediction": False,
    "susy": True,
    "higgs": True,
    "taxi": True,
}

FALKON_INDUCING_POINTS_GRID = [10000, 20000, 50000, 100000, 200000]
