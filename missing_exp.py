MISSING = [
    # missing weight decay and dropout for no noise experiements
    {"name": "base_sgd_dp",    "noise": 0,   "optimizer": "sgd",  "dropout": 0.2, "weight_decay": False},
    {"name": "base_sgd_wd",    "noise": 0,   "optimizer": "sgd",  "dropout": 0.0, "weight_decay": True},
    {"name": "base_sgd_dp_wd",    "noise": 0,   "optimizer": "sgd",  "dropout": 0.2, "weight_decay": True},

    {"name": "base_sgd_dp",    "noise": 0,   "optimizer": "adam",  "dropout": 0.2, "weight_decay": False},
    {"name": "base_sgd_wd",    "noise": 0,   "optimizer": "adam",  "dropout": 0.0, "weight_decay": True},
    {"name": "base_sgd_dp_wd",    "noise": 0,   "optimizer": "adam",  "dropout": 0.2, "weight_decay": True},

    #missing SDG wieght decay
    {"name": "0.1_sgd_wd",    "noise": 0.1,   "optimizer": "sgd",  "dropout": 0.0, "weight_decay": True},
    {"name": "0.2_sgd_wd",    "noise": 0.2,   "optimizer": "sgd",  "dropout": 0.0, "weight_decay": True},
    {"name": "0.5_sgd_wd",    "noise": 0.5,   "optimizer": "sgd",  "dropout": 0.0, "weight_decay": True},
    {"name": "1.0_sgd_wd",    "noise": 1.0,   "optimizer": "sgd",  "dropout": 0.0, "weight_decay": True},
    ]

