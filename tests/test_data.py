from utils.data_utils import connect_to_db


def test_connect_to_db():
    path_to_db = "data/viabill.db"
    _, _, tables_in_db = connect_to_db(path_to_db)
    # make sure the database is not empty
    assert len(tables_in_db) > 0
