import os
import shutil
import logging

import mlflow
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from utils.io_utils import dump_pickle, load_pickle

logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "viabill-experiment"


def construct_samples() -> tuple:
    X_train, y_train = load_pickle("data/train.pkl")
    X_val, y_val = load_pickle("data/valid.pkl")
    X_test, y_test = load_pickle("data/test.pkl")

    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])
    return (X_train, y_train, X_test, y_test)


def run():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.roc_auc_score DESC"],
    )[0]

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, "BestModel")
    client.transition_model_version_stage(
        name="BestModel", version=model_version.version, stage="Production"
    )

    model = mlflow.sklearn.load_model("models:/BestModel/production")

    X_train, y_train, X_test, y_test = construct_samples()

    model.fit(X_train, y_train)
    logging.info(
        "ROC AUC score on test sample is %f",
        roc_auc_score(y_test, model.predict(X_test)),
    )

    dump_pickle(model, os.path.join("inference", "model.pkl"))
    shutil.copyfile("Pipfile", "inference/Pipfile")
    shutil.copyfile("Pipfile.lock", "inference/Pipfile.lock")
    shutil.copyfile("data/scaler.pkl", "inference/scaler.pkl")
    shutil.copyfile("data/ohe.pkl", "inference/ohe.pkl")


if __name__ == "__main__":
    run()
