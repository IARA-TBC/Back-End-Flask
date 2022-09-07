from cv2 import resize
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = './model/TBC_CNN_Fusion.h5'

model = load_model(MODEL_PATH)


'''label_dict = {0: 'Covid19 Negative', 1: 'Covid19 Positive'}

def image(path):
    img = cv2.imread(path) 
    new_arr = cv2.resize(img, (100, 100))
    new_arr = np.array(new_arr)

    if (new_arr.ndim == 3):
        gray = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = new_arr

    gray = gray/255
    resized = cv2.resize(gray, (100, 100))
    reshaped = resized.reshape(1, 100, 100)
    
    return reshaped'''

def image(path):
    img = cv2.imread(path)
    resize = tf.image.resize(img, (256, 256))
    return np.expand_dims(resize/255, 0)



def model_predict(img_path, model):
    preds = model.predict(image(img_path))
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Hoa"


@app.route('/predict', methods=['POST'])
def upload():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    preds = model_predict(image_path, model)

    result = np.argmax(preds, axis=1)[0]
    accuracy = float(np.max(preds, axis=1)[0])

    ###label = label_dict[result]

    ####print(preds, result, accuracy)

    #response = {'prediction': {'result': label, 'accuracy': accuracy}}

    response = {'prediction':  accuracy}


    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

