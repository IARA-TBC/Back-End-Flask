# SuperFastPython.com
# example of returning multiple value from a thread
from time import sleep
from threading import Thread


# custom thread
'''class CustomThread(Thread):
    # constructor
    def __init__(self, image):
        # execute the base constructor
        Thread.__init__(self)
        # set a default values
        self.image = None
        self.value2 = None
        self.value3 = None
 
    # function executed in a new thread
    def run(self, image):
        # block for a moment
        sleep(1)
        # store data in an instance variable
        self.image = 'Hello from a new thread'
        self.value2 = 99
        self.value3 = False
 
# create a new thread
thread = CustomThread()
# start the thread
thread.start()
# wait for the thread to finish
thread.join()
# report all values returned from a thread
print(thread.value1)
print(thread.value2)
print(thread.value3)

# SuperFastPython.com
# example of returning a value from a thread
'''
'''
from time import sleep
from threading import Thread
 
# function executed in a new thread
def task():
    # block for a moment
    sleep(1)
    # correctly scope the global variable
    global data
    # store data in the global variable
    data = 'Hello from a new thread'
 
# define the global variable
data = None
# create a new thread
thread = Thread(target=task)
# start the thread
thread.start()
# wait for the thread to finish
thread.join()
# report the global variable
print(data)
'''

# custom thread
class CustomThread:
    # constructor

    def prueba(self, nombre):
        sleep(2)
        print("Hola bro", nombre)


    def __init__(self, nombre):
        # execute the base constructor
        t = Thread(target=self.prueba, args=nombre)
        # set a default value
        t.start()
 

CustomThread("Luis")

