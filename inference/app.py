import pickle

import pandas as pd
from flask import Flask, jsonify, request


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


scaler = load_pickle("scaler.pkl")
ohe = load_pickle("ohe.pkl")
model = load_pickle("model.pkl")


def preprocess_data(df: pd.DataFrame) -> tuple:
    categorical = ["sex", "defaulted_earlier", "late_earlier"]
    numerical = ["price", "income", "age"]

    df_copy = df.copy()
    df_copy[numerical] = scaler.transform(df_copy[numerical])
    ohe_categ_cols = list(ohe.get_feature_names_out())
    df_copy[ohe_categ_cols] = ohe.transform(df_copy[categorical])
    leave_columns = numerical + ohe_categ_cols + ["transactionID"]
    df_copy = df_copy[leave_columns]
    df_copy.set_index("transactionID", inplace=True)
    df_copy.sort_index(inplace=True)
    return (df_copy, scaler, ohe)


def predict(application):
    application = pd.DataFrame(application, index=[0])
    application = preprocess_data(application[0])
    preds = model.predict_proba(application)
    return float(preds[0][1])


app = Flask("default-prediction")


@app.route("/app", methods=["POST"])
def predict_endpoint():
    application = request.get_json()
    transactionID = application["transactionID"]

    pred = predict(application)

    result = {"default_prob": pred, "transactionID": transactionID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
