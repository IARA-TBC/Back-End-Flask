from useful_functions import model_cnn, image_cnn, image_transformers
from time import sleep
from threading import Thread, current_thread
from PIL import Image
import keras.models
 
# custom thread
'''
class CustomThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None
 
    # function executed in a new thread
    def run(self):
        # block for a moment
        sleep(1)
        # store data in an instance variable
        self.value = 'Hello from a new thread'
 
# create a new thread
thread = CustomThread()
# start the thread
thread.start()
# wait for the thread to finish
thread.join()
# get the value returned from the thread
data = thread.value
print(data)
'''

'''
class CustomThreadCnn():
    # constructor
    def __init__(self, image):

        # set a default value
        self.image = image
        self.preds_cnn = None

        t = Thread(target=self.model_predict_cnn)
        t.start()
        t.join()
        
    
    # function executed in a new thread
    def model_predict_cnn(self):
        self.preds_cnn = model_cnn.predict(image_cnn(self.image))
        #self.preds_cnn = preds_cnn
        #print(preds_cnn)
        #return  preds_cnn


# custom thread
class CustomThreadCnn2():
    # constructor
    def __init__(self, image):

        # set a default value
        self.image = image
        self.preds_cnn = None

        t = Thread(target=self.model_predict_cnn)
        t.start()
        
    
    # function executed in a new thread
    def model_predict_cnn(self):
        self.preds_cnn = model_cnn.predict(image_cnn(self.image))
        #self.preds_cnn = preds_cnn
        #print(preds_cnn)
        #return  preds_cnn



thread = CustomThreadCnn2(image)

thread.join()

data = thread.image
data2 = thread.preds_cnn
print(data)
print(data2) 
  
# start the thread
thread.start()
# wait for the thread to finish
thread.join()

'''
image = Image.open('0000.jpg')
# custom thread
class CustomThreadCnn(Thread):
    # constructor
    def __init__(self, image):

        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.image = image
        self.preds_cnn = model_cnn.predict(image_cnn(self.image))
    
    
    # function executed in a new thread
    def model_predict_cnn(self):
        print("Este es el primer thread para cnn",  current_thread().name)
        return self.preds_cnn
        #self.preds_cnn = preds_cnn
        #print(preds_cnn)
        #return  preds_cnn


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
        print("Este es el segunda thread para thread",  current_thread().name)
        return self.preds_transformers
        #self.preds_cnn = preds_cnn
        #print(preds_cnn)
        #return  preds_cnn

thread_cnn = CustomThreadCnn(image)
thread_cnn.start()


thread_transformers = CustomThreadTransformers(image)
thread_transformers.start()


thread_cnn.join()
thread_transformers.join()

preds_cnn = thread_cnn.preds_cnn
print(preds_cnn)
preds_transformers = thread_transformers(image)
print(preds_transformers)




'''class Employee:

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'Employee', 60000)


# SuperFastPython.com
# example of returning a value from a thread
from time import sleep
from threading import Thread

# custom thread
class CustomThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None

    # function executed in a new thread
    def run(self):
        # block for a moment
        sleep(1)
        # store data in an instance variable
        self.value = 'Hello from a new thread'

# create a new thread
thread = CustomThread()
# start the thread
thread.start()
# wait for the thread to finish
thread.join()
# get the value returned from the thread
data = thread.value
print(data)
'''