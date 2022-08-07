from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier

from training.hpo import tune  # pylint: disable=import-error


def tune_rf(num_trials: int, path_to_data: str):
    model = RandomForestClassifier
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
        "random_state": 42,
        "n_jobs": -1,
    }

    tune(model, search_space, num_trials, path_to_data, tag="Random Forest")
