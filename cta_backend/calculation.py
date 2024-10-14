from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


@app.route('/api/submit', methods=['POST'])
def receive_data():
    data = request.json
    print(data)
    return jsonify({'message': 'Data received successfully'}), 200


@app.route('/api/data', methods=['GET'])
def get_data():
    y = np.random.rand(100)
   
    points1 = list(map(lambda idx: {"x": idx[0], "y": idx[1]}, enumerate( np.random.rand(100)) ))
    points2 = list(map(lambda idx: {"x": idx[0], "y": idx[1]}, enumerate( np.random.rand(100)) ))
    data = [
        {"id": "series1", "data":points1},
        {"id": "series2", "data": points2}
    ]
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)