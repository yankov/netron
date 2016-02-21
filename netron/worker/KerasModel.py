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

    def run_job(self, model_json, X_train, y_train, X_test, y_test, nb_epoch=10):
        print "Loading model..."
        self.model = model_from_json(model_json)
        if X_test is not None and y_test is not None:
            validation_data = (X_test, y_test)
            monitor = "val_loss"
        else:
            validation_data = None
            monitor = "loss"

        print "Training"
        earlystopper = EarlyStopping(monitor=monitor, patience=self.patience, verbose=1, mode='auto')
        res = self.model.fit(X_train, y_train, nb_epoch=self.nb_epoch, verbose=1,
                             validation_data=validation_data, callbacks=[earlystopper])
        if validation_data:
            print "Evaluating model on a test set"
            score = self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=1)
            print "Test score: %f\n Test accuracy: %f" % (score[0], score[1])
            res.history["val_loss"] = score[0]
            res.history["val_accuracy"] = score[1]
        return res

