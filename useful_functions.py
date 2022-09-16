import numpy as np
import io
import keras.models
import tensorflow as tf
from skimage import transform
from PIL import Image
from keras.models import load_model
from transformers import TFViTForImageClassification
import transformers

MODEL_PATH_CNN = './models/TBC_CNN_Fusion.h5'

model_cnn = load_model(MODEL_PATH_CNN)

with open("./models/config/ConfigTrain.json") as json_file:
    json_config = json_file.read()
transformer = keras.models.model_from_json(json_config)
transformer.load_weights('./models/WeigthsTrain.h5')

def image_cnn(image):
    np_image = np.array(image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def image_transformers(image):
    image = image.convert("RGB")
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    img = np.moveaxis(image, -1, 0)
    img = np.expand_dims(img, axis=0)
    return img

def model_predict_cnn(image):
    preds = model_cnn.predict(image_cnn(image))
    return preds

def model_predict_transformers(image):
    #Arreglar el predict
    preds = transformers.predict(image_transformers(image))
    return preds