from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from useful_functions import model_predict_cnn, model_cnn

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return jsonify("Hola")


@app.route('/predict', methods=['POST'])
def upload():
    imagefile = request.form['file']
    print(imagefile)
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    preds = model_predict_cnn(image_path, model_cnn)

    accuracy = float(np.max(preds, axis=1)[0])
    accuracy = str(round(accuracy * 100 , 2))
    accuracy = accuracy + ' %'

    response = {'prediction: ' : accuracy}


    return jsonify(response)


if __name__ == '__main__':
    app.run(port=8000,debug=True)
