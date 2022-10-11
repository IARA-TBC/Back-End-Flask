from useful_functions import model_cnn, image_cnn, image_transformers, model_predict_cnn
from threading import Thread
import keras.models
from PIL import Image, ImageFile
from time import sleep

ImageFile.LOAD_TRUNCATED_IMAGES = True

image = Image.open('0003.jpg')

# Inicializo una clase
class CustomThreadCnn(Thread):
    # Creo que un constructor que me permite inicializar las variables que voy a utiliza
    def __init__(self, image):
        #Le indico que cuando se ejecute la clase se abra otro thread
        Thread.__init__(self)
        #Defino las variables que voy a utilizar
        #Self.image va a ser igual a la imagen que me pasen
        #
        self.image = image
        self.value = None
 
    #Funcion que voy a ejecutar en otro thread
    def run(self):
        # block for a moment
        #sleep(1)
        # store data in an instance variable
        self.value = model_cnn.predict(image_cnn(self.image))
 

with open("./models/config/ConfigTrain.json") as json_file:
    json_config = json_file.read()
transformer = keras.models.model_from_json(json_config)
transformer.load_weights('./models/WeigthsTrain.h5')

class CustomThreadTransformers(Thread):
    # constructor
    def __init__(self, image):
        #Le indico que cuando se ejecute la clase se abra otro thread
        Thread.__init__(self)
        #Defino las variables que voy a utilizar
        #Self.image va a ser igual a la imagen que me pasen
        #
        self.image = image
        self.value = None
 
    #Funcion que voy a ejecutar en otro thread
    def run(self):
        # block for a moment
        #sleep(1)
        # store data in an instance variable
        self.value = transformer.predict(image_transformers(self.image))

