from flask import Flask, jsonify
import pandas as pd
import json

app = Flask(__name__)

@app.route('/api/prices')
def get_prices():
    data = pd.read_csv('data/BrentOilPrices.csv')
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/metrics')
def get_model_metrics():
    
    with open('data/metrics.json', 'r') as file:
        metrics = json.load(file)
    return metrics