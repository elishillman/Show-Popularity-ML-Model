from model import predictShow
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    pred = predictShow(9.0,10000,'HBO Max',10,1000000,'current',2.0,'Drama')
    return render_template('index.html')
