#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Dropout

class NNModel(object):
    def __init__(self):
#        self.model = Sequential()
        self.data = None

    def load_task(self, params):
        """ TODO: load model, data and train the model """
        print "Loaded model with params: " + params
