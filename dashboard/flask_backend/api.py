from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/api/prices')
def get_prices():
    data = pd.read_csv('data/BrentOilPrices.csv')
    return jsonify(data.to_dict(orient='records'))