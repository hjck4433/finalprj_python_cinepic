from flask import Flask, Response, request, jsonify
from routes.theater import get_theater_data
from routes.movies import send_movie_data
from routes.get_recommendation import calculate_recommendations
from routes.elastic import index_movies
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins=['http://localhost:8111'])

@app.route('/api/theater',methods=['GET'])
def theater_data():
    result = get_theater_data()
    return Response(result, content_type='application/json; charset=utf-8')

@app.route('/api/movies', methods=['GET'])
def movie_data():
    result = send_movie_data()
    return Response(result, content_type='application/json; charset=utf-8')

@app.route('/api/recommendation', methods=['POST'])
def movie_recs():
    user_preferences = request.json

    result = calculate_recommendations(user_preferences)

    return Response(result, content_type='application/json; charset=utf-8')

@app.route('/api/index/movies', methods=['GET'])
def index_movies_route():
    result = index_movies()
    return jsonify({"message": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

