import os
import pickle

import numpy as np
import mlflow
from hyperopt import STATUS_OK, Trials, tpe, fmin
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("viabill-experiment")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def tune(model, search_space: dict, num_trials: int, path_to_data: str, tag: str):
    X_train, y_train = load_pickle(os.path.join(path_to_data, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(path_to_data, "valid.pkl"))

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", tag)
            mlflow.log_params(params)
            candidate = model(**params)
            candidate.fit(X_train, y_train)
            y_pred = candidate.predict(X_val)
            roc_auc = roc_auc_score(y_val, y_pred)
            mlflow.log_metric("roc_auc_score", roc_auc)

        return {"loss": -1 * roc_auc, "status": STATUS_OK}

    rstate = np.random.default_rng(17)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )
