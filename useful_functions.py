import numpy as np
import io
import tensorflow as tf
from skimage import transform
from PIL import Image
from keras.models import load_model

MODEL_PATH_CNN = './models/TBC_CNN_Fusion.h5'
#MODEL_PATH_TRANSFORMERS = './models/WeightsInfer.h5'

model_cnn = load_model(MODEL_PATH_CNN)
#model_Transformers = load_model(MODEL_PATH_TRANSFORMERS)

def image_cnn(path):
    np_image = Image.open(path)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def image_transformers(path):
    file = Image.open(io.BytesIO(path)).convert("RGB")
    file = file.resize((224, 224))
    file = np.array(file) / 255.0
    file = np.moveaxis(file, -1, 0)
    return file

def model_predict_cnn(img_path, model):
    preds = model.predict(image_cnn(img_path))
    return preds

def model_predict_transformers(img_path, model):
    preds = model.predict(image_transformers(img_path))
    return preds