from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
from useful_functions import model_predict_cnn, model_predict_transformers
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return jsonify("Hola")


@app.route('/predict', methods=['POST'])
def upload():
    print(request.json)
    imagefile = request.json
    req = requests.post('http://localhost:4000/images/sendFile', json=imagefile)

    base64_data = req.content

    image = Image.open(io.BytesIO(base64_data))

    preds = model_predict_cnn(image)
    predsTransformers = model_predict_transformers(image)
    print('Transformers:', predsTransformers)

    accuracy_cnn = float(np.max(preds, axis=1)[0])
    accuracy_cnn = str(round(accuracy_cnn * 100 , 2))
    accuracy_cnn = accuracy_cnn + ' %'

    accuracy_transformers = float(np.max(preds, axis=1)[0])
    accuracy_transformers = str(round(accuracy_transformers * 100 , 2))
    accuracy_transformers = accuracy_transformers + ' %'

    return jsonify({
        'prediction_cnn': accuracy_cnn,
        'prediction_transformers': accuracy_transformers
    })


if __name__ == '__main__':
    app.run(port=8000,debug=True)
