DATA_DIR = "./data/"
DATA_CONFIGS = {
    "a9a": {"tr": "a9a", "tst": "a9a.t", "loading": "libsvm"},
    "acsincome": {
        "tr": "acsincome_data.pkl",
        "tgt": "acsincome_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "airlines": {
        "tr": "airlines_data.pkl",
        "tgt": "airlines_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "aspirin": {"tr": "md17_aspirin.npz", "loading": "npz", "split": 0.8},
    "benzene": {"tr": "md17_benzene2017.npz", "loading": "npz", "split": 0.8},
    "cadata": {"tr": "cadata", "loading": "libsvm", "split": 0.8},
    "click_prediction": {
        "tr": "click_prediction_data.pkl",
        "tgt": "click_prediction_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "cod_rna": {"tr": "cod-rna", "tst": "cod-rna.t", "loading": "libsvm"},
    "comet_mc": {
        "tr": "comet_mc_data.pkl",
        "tgt": "comet_mc_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "connect_4": {"tr": "connect-4", "loading": "libsvm", "split": 0.8},
    "covtype_binary": {
        "tr": "covtype.libsvm.binary.scale",
        "loading": "libsvm",
        "split": 0.8,
    },
    "creditcard": {
        "tr": "creditcard_data.pkl",
        "tgt": "creditcard_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "diamonds": {
        "tr": "diamonds_data.pkl",
        "tgt": "diamonds_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "ethanol": {"tr": "md17_ethanol.npz", "loading": "npz", "split": 0.8},
    "higgs": {"tr": "HIGGS", "loading": "libsvm", "split": 10500000},
    "hls4ml": {
        "tr": "hls4ml_data.pkl",
        "tgt": "hls4ml_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "ijcnn1": {"tr": "ijcnn1.tr", "tst": "ijcnn1.t", "loading": "libsvm"},
    "jannis": {
        "tr": "jannis_data.pkl",
        "tgt": "jannis_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "malonaldehyde": {"tr": "md17_malonaldehyde.npz", "loading": "npz", "split": 0.8},
    "medical": {
        "tr": "medical_data.pkl",
        "tgt": "medical_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "miniboone": {
        "tr": "miniboone_data.pkl",
        "tgt": "miniboone_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "mnist": {
        "tr": "mnist_data.pkl",
        "tgt": "mnist_target.pkl",
        "loading": "pkl",
        "split": 60000,
    },
    "naphthalene": {"tr": "md17_naphthalene.npz", "loading": "npz", "split": 0.8},
    "phishing": {"tr": "phishing", "loading": "libsvm", "split": 0.8},
    "qm9": {"tr": "homo.mat", "loading": "mat", "split": 100000},
    "santander": {
        "tr": "santander_data.pkl",
        "tgt": "santander_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "salicylic": {"tr": "md17_salicylic.npz", "loading": "npz", "split": 0.8},
    "sensit_vehicle": {
        "tr": "combined_scale",
        "tst": "combined_scale.t",
        "loading": "libsvm",
    },
    "sensorless": {
        "tr": "Sensorless.scale.tr",
        "tst": "Sensorless.scale.val",
        "loading": "libsvm",
    },
    "skin_nonskin": {"tr": "skin_nonskin", "loading": "libsvm", "split": 0.8},
    "susy": {"tr": "SUSY", "loading": "libsvm", "split": 4500000},
    "synthetic": {},
    "taxi": {
        "tr": "taxi-data/subsampled_data.h5py",
        "loading": "h5py",
        "split": 100000000,
    },
    "toluene": {"tr": "md17_toluene.npz", "loading": "npz", "split": 0.8},
    "uracil": {"tr": "md17_uracil.npz", "loading": "npz", "split": 0.8},
    "volkert": {
        "tr": "volkert_data.pkl",
        "tgt": "volkert_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
    "w8a": {"tr": "w8a", "tst": "w8a.t", "loading": "libsvm"},
    "yearpredictionmsd": {
        "tr": "YearPredictionMSD",
        "tst": "YearPredictionMSD.t",
        "loading": "libsvm",
    },
    "yolanda": {
        "tr": "yolanda_data.pkl",
        "tgt": "yolanda_target.pkl",
        "loading": "pkl",
        "split": 0.8,
    },
}
DATA_KEYS = list(DATA_CONFIGS.keys())
SYNTHETIC_NTR = 10000
SYNTHETIC_NTST = 1000
SYNTHETIC_D = 10
