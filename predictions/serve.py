import pickle
from flask import Flask, request, jsonify
import requests
import os

MODEL_PATH = os.getenv("MODEL_PATH", "./models/model_C_1_0.bin")
VERSION = os.getenv("VERSION", "N/A")

with open(MODEL_PATH, "rb") as f_in:
    dv, model = pickle.load(f_in)

def prepare_features(customer):
    # turn into dictionary if not already so, mind formats that the model expects
    customer = dv.transform([customer])
    return customer

def predict(customer) :
    y_pred = model.predict_proba(customer)[0, 1]
    return float(y_pred)

app = Flask('churn')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    customer = request.get_json()
    X = prepare_features(customer)
    y_pred = predict(X)
    result = {
        "prediction": {
            "churn_probability": y_pred
        },
        "version": VERSION
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)