from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

class KerasModel(object):
    def __init__(self, nb_epoch = 10, patience = 5):
        self.model = None
        self.data = None

        # How many epoch to train (if score earlystopping doesn't kick in earlier)
        self.nb_epoch = nb_epoch

        # How many epoch of not improving before earlystopper kicks in
        self.patience = patience

    def run_job(self, model_json, x_train, y_train, nb_epoch=10):
        print "Loading model..."
        self.model = model_from_json(model_json)
        print "Training"
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1, mode='auto')
        res = self.model.fit(x_train, y_train, nb_epoch=self.nb_epoch, verbose=1, callbacks=[earlystopper])
        print "Complete!"
        return res

