from useful_functions import model_cnn, image_cnn, image_transformers, model_predict_cnn
from threading import Thread
import keras.models
from PIL import Image
from time import sleep

image = Image.open('0003.jpg')
# Inicializo una clase
class CustomThreadCnn(Thread):
    # Creo que un constructor que me permite inicializar las variables que voy a utilizar
    def __init__(self, image):

        #Le indico que cuando se ejecute la clase se abra otro thread
        Thread.__init__(self)
        #Defino las variables que voy a utilizar
        #Self.image va a ser igual a la imagen que me pasen
        self.image = image
        #Self.preds_cnn va a ser igual a la predicción del modelo de cnn
        self.preds_cnn = model_cnn.predict(image_cnn(self.image))
    
    
    #Funcion que voy a ejecutar en otro thread
    def model_predict_cnn(self):
        #Devuelvo la predicción del modelo de cnn 
        return self.preds_cnn



with open("./models/config/ConfigTrain.json") as json_file:
    json_config = json_file.read()
transformer = keras.models.model_from_json(json_config)
transformer.load_weights('./models/WeigthsTrain.h5')

class CustomThreadTransformers(Thread):
    # constructor
    def __init__(self, image):

        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.image = image
        self.preds_transformers = transformer.predict(image_transformers(self.image))
    
    
    # function executed in a new thread
    def model_predict_transformers(self):
        return self.preds_transformers

