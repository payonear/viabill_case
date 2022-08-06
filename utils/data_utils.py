import logging

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import inspect
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def connect_to_db(path_to_db: str = "data/viabill.db") -> tuple:
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

    return (engine, inspector, tables_in_db)


def create_sample(path_to_query: str) -> pd.DataFrame:
    engine, _, _ = connect_to_db()
    with open(path_to_query, "r", encoding="utf-8") as file:
        doc = file.readlines()
    query = "".join(doc)
    df = pd.read_sql(query, con=engine)
    return df


def prepare_training_sample(path_to_query: str) -> tuple:
    df = create_sample(path_to_query)
    df["default"] = determine_default(df)
    df = add_features(df)
    df = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_sample(df)
    return (X_train, X_val, X_test, y_train, y_val, y_test)


def preprocess_data(df: pd.DataFrame, scaler=None, ohe=None) -> tuple:
    categorical = ["sex", "defaulted_earlier", "late_earlier"]
    numerical = ["price", "income", "age"]
    if not scaler or not ohe:
        scaler = StandardScaler()
        ohe = OneHotEncoder(drop="first")

        scaler.fit(df[numerical])
        ohe.fit(df[categorical])
    df_copy = df.copy()
    df_copy[numerical] = scaler.transform(df_copy[numerical])
    ohe_categ_cols = list(ohe.get_feature_names_out())
    df_copy[ohe_categ_cols] = ohe.transform(df_copy[categorical])
    leave_columns = numerical + ohe_categ_cols + ["transactionID", "default"]
    df_copy = df_copy[leave_columns]
    df_copy.set_index("transactionID", inplace=True)
    df_copy.sort_index(inplace=True)
    return (df_copy, scaler, ohe)


def determine_default(df: pd.DataFrame) -> np.array:
    return np.where(df["paytmentStatus4"] == 2, 1, 0)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy[df_copy["age"] >= 18]
    df_copy = df_copy[df_copy["income"] >= 1000]
    df_copy = df_copy[df_copy["price"] >= 20]
    return df_copy


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # presorting for proper expanding
    df_copy = df.copy()
    df_copy.sort_values(["customerID", "transactionID"], inplace=True)
    df_copy.reset_index(drop=True, inplace=True)

    # define whether customer defaulted earlier
    df_copy["num_defs"] = (
        df_copy.groupby(["customerID"]).default.expanding().sum().values
    )
    df_copy["defaulted_earlier"] = np.where(
        df_copy.groupby(["customerID"]).num_defs.shift(1, fill_value=0) > 0, 1, 0
    )

    # define whether customer was late with payments earlier
    pstatuses = [
        "paytmentStatus1",
        "paytmentStatus2",
        "paytmentStatus3",
        "paytmentStatus4",
    ]
    df_copy["late"] = np.where(df_copy[pstatuses].max(axis=1) > 0, 1, 0)
    df_copy["num_lates"] = df_copy.groupby(["customerID"]).late.expanding().sum().values
    df_copy["late_earlier"] = np.where(
        df_copy.groupby(["customerID"]).num_lates.shift(1, fill_value=0) > 0, 1, 0
    )

    return df_copy


def split_sample(
    df: pd.DataFrame, val_size: float = 0.3, test_size: float = 0.2
) -> tuple:
    test = df.iloc[-int(df.shape[0] * test_size) :]
    train_val = df.iloc[: -int(df.shape[0] * test_size)]
    val = train_val.iloc[-int(df.shape[0] * val_size) :]
    train = train_val.iloc[: -int(df.shape[0] * val_size)]

    X_train, y_train = train.drop(["default"], axis=1), train["default"]
    X_val, y_val = val.drop(["default"], axis=1), val["default"]
    X_test, y_test = test.drop(["default"], axis=1), test["default"]
    return (X_train, X_val, X_test, y_train, y_val, y_test)
