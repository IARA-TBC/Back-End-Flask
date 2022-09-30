from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
from useful_functions import model_predict_transformers, model_predict_cnn
from Threads import CustomThreadCnn, CustomThreadTransformers
from PIL import Image
import pydicom as PDCM
import io
from threading import Thread, current_thread
from ImageConverter.imageConverter import Dicom_to_Image
    
app = Flask(__name__)
CORS(app)

#Just a test route
@app.route('/', methods=['GET'])
def index():
    # Main page
    return jsonify("Hola")



#Main route
@app.route('/predict_jpg', methods=['POST'])
def predict_jpg_img():
    print(request.json)
    #Recibo el json enviado por node que contiene la ruta de la imagen
    imagefile = request.json

    #Hago una petición post a una ruta del back-end que recibe una ruta de una imagen y devuelve la misma. 
    #Por supuesto, en este caso estoy enviando la ruta de la imagen antes recibida
    req = requests.post('http://localhost:4000/images/sendFile', json=imagefile)

    #Recibo la imagen en base64
    base64_data = req.content

    #Abro convierto la información a bytes y la abro con pillow
    image = Image.open(io.BytesIO(base64_data))

    #Le indico a un thread que targetee a la función que permite al modelo de cnn predecir el resultado de la imagen
    #Además le paso como paraámetro la imagen
    
    '''Cnn_Thread = Thread(target=model_predict_cnn, args=(image,))

    Cnn_Thread.start()
    print("ESTE ES EL PRIMERO:", current_thread().name)

    #Le indico a un thread que targetee a la función que permite al modelo de transformers predecir el resultado de la imagen
    #Además le paso como paraámetro la imagen    
    Thread_Transformers = Thread(target=model_predict_transformers, args=(image,))

    #Abro el thread de cnn

    #Abro el thread de transformers
    Thread_Transformers.start()
    print("ESTE ES EL SEGUNDO:", current_thread().name) 

    #Cierro el thread de cnn
    Cnn_Thread.join()
    #Cierro el thread de transformers
    Thread_Transformers.join()



    preds_cnn = model_predict_cnn(image)

    preds_transformers = model_predict_transformers(image)
    '''    
    thread_cnn = CustomThreadCnn(image)
    thread_cnn.start()


    thread_transformers = CustomThreadTransformers(image)
    thread_transformers.start()


    thread_cnn.join()
    thread_transformers.join()

    preds_cnn = thread_cnn.preds_cnn
    print(preds_cnn)
    print(thread_cnn.name)
    preds_transformers = thread_transformers.preds_transformers
    print(preds_transformers)
    print(thread_transformers.name)

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
        'prediction_average': accuracy_average,
    })


@app.route('/predict_dicom', methods=['POST'])
def predict_dicom_img():
    print(request.json)
    imagefile = request.json

    req = requests.post('http://localhost:4000/images/sendFile', json=imagefile)
    base64_data = req.content

    dicom_encoding = PDCM.read_file(io.BytesIO(base64_data))
    dicom_image_path = Dicom_to_Image(dicom_encoding)
    image = Image.open(dicom_image_path, mode='r')

    #image_to_send = open(dicom_encoding)

    files = {'file': open(dicom_image_path, 'rb')}
    image_req = requests.post('http://localhost:4000/images/saveImageRoute', files=files)
    print(image_req.json()['path'])
    new_path = image_req.json()['path']

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
        'prediction_average': accuracy_average,
        'new_path': new_path
    })


if __name__ == '__main__':
    app.run(port=8000,debug=True)
