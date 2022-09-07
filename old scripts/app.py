# Recibir la imagen y guardarla
# Ponerle los filtros necesarios
# Mandarla a predecir


from flask import Flask, request
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re


img_size = 100

app = Flask(__name__)

MODEL_PATH = 'model-015.model'

model = load_model(MODEL_PATH)


label_dict = {0: 'Covid19 Negative', 1: 'Covid19 Positive'}


def preprocess(img):

    img = np.array(img)

    if (img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray/255
    resized = cv2.resize(gray, (img_size, img_size))
    reshaped = resized.reshape(1, img_size, img_size)
    return reshaped


@app.route('/', methods=['GET'])
def helloWorld():
    return "Hello"


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    prediction = preprocess(imagefile)
    result = np.argmax(prediction, axis=1)[0]
    accuracy = float(np.max(prediction, axis=1)[0])

    label = label_dict[result]

    print(prediction, result, accuracy)

    response = {'prediction': {'result': label, 'accuracy': accuracy}}

    return jsonify(response)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
