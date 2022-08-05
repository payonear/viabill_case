import pandas as pd

from utils.data_utils import connect_to_db, create_sample, determine_default


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
