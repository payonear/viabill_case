import logging
import argparse

from training import random_forest, logistic_regression
from utils.data_utils import prepare_training_samples

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_query",
        default="./utils/sample_query.txt",
        help="the location where txt with SQL query for sample creation is located.",
    )
    parser.add_argument(
        "--samples_dir",
        default="./data/",
        help="the location where prepared samples for experiment will be stored.",
    )

    parser.add_argument(
        "--num_trials",
        default=20,
        help="number of tuning rounds for each model.",
    )

    args = parser.parse_args()

    # prepare samples
    prepare_training_samples(args.path_to_query, args.samples_dir)

    # run models tuning job
    logistic_regression.tune_lr(args.num_trials, args.samples_dir)
    random_forest.tune_rf(args.num_trials, args.samples_dir)
