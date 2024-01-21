from flask import Flask, Response
from routes.theater import get_theater_data
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, origins=['http://localhost:8111'])

@app.route('/api/theater',methods=['GET'])
def theater_data():
    result = get_theater_data()
    return Response(result, content_type='application/json; charset=utf-8')


if __name__ == '__main__':
    app.run(debug=True)

