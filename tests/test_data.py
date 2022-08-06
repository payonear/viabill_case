import json

import pandas as pd

from utils.data_utils import (
    add_features,
    split_sample,
    connect_to_db,
    create_sample,
    remove_outliers,
    determine_default,
)

with open("tests/test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data = pd.DataFrame.from_dict(test_data)


def test_connect_to_db():
    path_to_db = "data/viabill.db"
    _, _, tables_in_db = connect_to_db(path_to_db)
    # make sure the database is not empty
    assert len(tables_in_db) > 0


def test_create_sample():
    path_to_query = "tests/test_sql_query.txt"
    sample = create_sample(path_to_query)
    assert sample.shape[0] > 0


def test_determine_default():
    data = {"paytmentStatus4": [0, 1, 1, 0, 2, 1, 2]}
    df = pd.DataFrame.from_dict(data)
    df["default"] = determine_default(df)
    assert all(df["default"].values == [0, 0, 0, 0, 1, 0, 1])


def test_remove_outliers():
    no_outliers = remove_outliers(test_data)

    assert no_outliers.shape[0] == 4
    assert no_outliers[no_outliers.age < 18].shape[0] == 0
    assert no_outliers[no_outliers.income < 1000].shape[0] == 0
    assert no_outliers[no_outliers.price < 20].shape[0] == 0


def test_split_sample():
    X_train, X_val, X_test, y_train, y_val, y_test = split_sample(
        test_data, val_size=0.2, test_size=0.2
    )
    assert X_train.shape[0] == y_train.shape[0] == 6
    assert X_val.shape[0] == y_val.shape[0] == 2
    assert X_test.shape[0] == y_test.shape[0] == 2


def test_add_features():
    with_features = add_features(test_data)
    assert all(with_features["defaulted_earlier"] == [0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    assert all(with_features["late_earlier"] == [0, 1, 1, 1, 0, 1, 1, 1, 0, 0])
