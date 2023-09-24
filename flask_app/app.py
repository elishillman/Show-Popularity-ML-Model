from .model import predictShow
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    rating = float(request.form['imdb_rating'])
    votes = int(request.form['imdb_votes'])
    service = request.form['streaming_service']
    comments = int(request.form['reddit_comments'])
    followers = int(request.form['twitter_followers'])
    status = request.form['status']
    runtime = int(request.form['runtime'])
    genre = request.form['genre']
    prediction = predictShow(rating, votes, service, comments, followers, status, runtime, genre)
    
    return render_template('result.html', prediction=prediction)