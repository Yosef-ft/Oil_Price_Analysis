import time
from flask import Flask

app = Flask(__name__)

@app.route('/test')
def get_current_time():
    return {'time': time.time()}