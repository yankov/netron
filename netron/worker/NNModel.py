from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.models import model_from_json

class NNModel(object):
    def __init__(self):
        self.model = None
        self.data = None

    def load_task(self, model_json, x_train, y_train):
        print "Loading model..."
        self.model = model_from_json(model_json)
        print "Training"
        res = self.model.fit(x_train, y_train, nb_epoch=2)
        print "Complete!"
        return res

