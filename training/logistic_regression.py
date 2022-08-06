from hyperopt import hp
from sklearn.linear_model import LogisticRegression

from training.hpo import tune


def tune_lr(num_trials: int, path_to_data: str):
    model = LogisticRegression
    search_space = {
        "C": hp.lognormal("C", 0, 1),
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42,
    }

    tune(model, search_space, num_trials, path_to_data, tag="Logistic Regression")
