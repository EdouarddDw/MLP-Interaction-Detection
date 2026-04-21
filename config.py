EXPERIMENTS = [

    # ── No noise: baselines only ──────────────────────────────────────────────
    {"name": "base_adam",   "noise": 0,   "optimizer": "adam", "dropout": 0.0, "weight_decay": None},
    {"name": "base_sgd",    "noise": 0,   "optimizer": "sgd",  "dropout": 0.0, "weight_decay": None},

    # ── Noise 0.1 SD ──────────────────────────────────────────────────────────
    {"name": "0.1_adam",         "noise": 0.1, "optimizer": "adam", "dropout": 0.0, "weight_decay": None},
    {"name": "0.1_sgd",          "noise": 0.1, "optimizer": "sgd",  "dropout": 0.0, "weight_decay": None},
    {"name": "0.1_adam_dp",      "noise": 0.1, "optimizer": "adam", "dropout": 0.2, "weight_decay": None},
    {"name": "0.1_adam_wd",      "noise": 0.1, "optimizer": "adam", "dropout": 0.0, "weight_decay": "L2"},
    {"name": "0.1_adam_dp_wd",   "noise": 0.1, "optimizer": "adam", "dropout": 0.2, "weight_decay": "L2"},
    {"name": "0.1_sgd_dp",       "noise": 0.1, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": None},
    {"name": "0.1_sgd_dp_wd",    "noise": 0.1, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": "L2"},

    # ── Noise 0.2 SD ──────────────────────────────────────────────────────────
    {"name": "0.2_adam",         "noise": 0.2, "optimizer": "adam", "dropout": 0.0, "weight_decay": None},
    {"name": "0.2_sgd",          "noise": 0.2, "optimizer": "sgd",  "dropout": 0.0, "weight_decay": None},
    {"name": "0.2_adam_dp",      "noise": 0.2, "optimizer": "adam", "dropout": 0.2, "weight_decay": None},
    {"name": "0.2_adam_wd",      "noise": 0.2, "optimizer": "adam", "dropout": 0.0, "weight_decay": "L2"},
    {"name": "0.2_adam_dp_wd",   "noise": 0.2, "optimizer": "adam", "dropout": 0.2, "weight_decay": "L2"},
    {"name": "0.2_sgd_dp",       "noise": 0.2, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": None},
    {"name": "0.2_sgd_dp_wd",    "noise": 0.2, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": "L2"},

    # ── Noise 0.5 SD ──────────────────────────────────────────────────────────
    {"name": "0.5_adam",         "noise": 0.5, "optimizer": "adam", "dropout": 0.0, "weight_decay": None},
    {"name": "0.5_sgd",          "noise": 0.5, "optimizer": "sgd",  "dropout": 0.0, "weight_decay": None},
    {"name": "0.5_adam_dp",      "noise": 0.5, "optimizer": "adam", "dropout": 0.2, "weight_decay": None},
    {"name": "0.5_adam_wd",      "noise": 0.5, "optimizer": "adam", "dropout": 0.0, "weight_decay": "L2"},
    {"name": "0.5_adam_dp_wd",   "noise": 0.5, "optimizer": "adam", "dropout": 0.2, "weight_decay": "L2"},
    {"name": "0.5_sgd_dp",       "noise": 0.5, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": None},
    {"name": "0.5_sgd_dp_wd",    "noise": 0.5, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": "L2"},

    # ── Noise 1.0 SD ──────────────────────────────────────────────────────────
    {"name": "1.0_adam",         "noise": 1.0, "optimizer": "adam", "dropout": 0.0, "weight_decay": None},
    {"name": "1.0_sgd",          "noise": 1.0, "optimizer": "sgd",  "dropout": 0.0, "weight_decay": None},
    {"name": "1.0_adam_dp",      "noise": 1.0, "optimizer": "adam", "dropout": 0.2, "weight_decay": None},
    {"name": "1.0_adam_wd",      "noise": 1.0, "optimizer": "adam", "dropout": 0.0, "weight_decay": "L2"},
    {"name": "1.0_adam_dp_wd",   "noise": 1.0, "optimizer": "adam", "dropout": 0.2, "weight_decay": "L2"},
    {"name": "1.0_sgd_dp",       "noise": 1.0, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": None},
    {"name": "1.0_sgd_dp_wd",    "noise": 1.0, "optimizer": "sgd",  "dropout": 0.2, "weight_decay": "L2"},
]
