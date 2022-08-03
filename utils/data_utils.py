import logging

import sqlalchemy
from sqlalchemy import inspect

logging.basicConfig(level=logging.DEBUG)


def connect_to_db(path_to_db="data/viabill.db"):
    sqlite_path_to_db = f"sqlite:///{path_to_db}"
    engine = sqlalchemy.create_engine(sqlite_path_to_db)
    logging.info('SQL engine is created.')

    inspector = inspect(engine)
    schemas = inspector.get_schema_names()
    tables_in_db = inspector.get_table_names(schema=schemas[0])
    logging.info(
        "%d tables found inside database %s",
        len(tables_in_db),
        path_to_db.split('/')[-1],
    )

    return engine, inspector, tables_in_db
