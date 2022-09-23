from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
from useful_functions import model_predict_cnn, model_predict_transformers
from PIL import Image
import io
from threading import Thread, current_thread

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

    Cnn_Thread = Thread(target=model_predict_cnn, args=(image,))    
    Thread_Transformers = Thread(target=model_predict_transformers, args=(image,))

    Cnn_Thread.start()
    print("ESTE ES EL PRIMERO:", current_thread().name)
    Thread_Transformers.start()
    print("ESTE ES EL SEGUNDO:", current_thread().name) 

    Cnn_Thread.join()
    Thread_Transformers.join()

    preds_cnn = model_predict_cnn(image)

    preds_transformers = model_predict_transformers(image)

    accuracy_cnn_raw = float(np.max(preds_cnn, axis=1)[0])
    accuracy_cnn = str(round(accuracy_cnn_raw * 100 , 2))
    accuracy_cnn = accuracy_cnn + ' %'

    accuracy_transformers_raw = float(np.max(preds_transformers, axis=1)[0])
    accuracy_transformers = str(round(accuracy_transformers_raw * 100 , 2))
    accuracy_transformers = accuracy_transformers + ' %'

    accuracy_average = (accuracy_cnn_raw + accuracy_transformers_raw)/2
    accuracy_average = str(round(accuracy_average * 100 , 2))
    accuracy_average = accuracy_average + ' %'

    return jsonify({
        'prediction_cnn': accuracy_cnn,
        'prediction_transformers': accuracy_transformers,
        'prediction_average': accuracy_average
    })


if __name__ == '__main__':
    app.run(port=8000,debug=True)
