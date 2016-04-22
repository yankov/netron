from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from netron.models import KerasModelFactory
from netron.worker import DataLoader
from keras.optimizers import SGD
from pymongo import MongoClient
import json

class KerasModel(object):
    def __init__(self, exp_id, nb_epoch = 10, patience = 5, mongo_uri = "mongodb://localhost:27017/", data_filename = None):
        self.exp_id = exp_id
        self.model = None
        self.data = None
        self.data_loader = None

        # How many epoch to train (if score earlystopping doesn't kick in earlier)
        self.nb_epoch = nb_epoch

        # How many epoch of not improving before earlystopper kicks in
        self.patience = patience

        self.mongo_uri = mongo_uri

        self.factory = KerasModelFactory()

        self.data_loader = DataLoader("http://localhost:8080", self.mongo_uri)

    def run_job(self, params):
        # load data from a numpy archive or s3
        X_train, y_train, X_test, y_test = self.data_loader.load_data(params["data_filename"])

        # Validate model
        model_json = self.factory.build_from_sketch(params, X_train.shape)

        # Wrong network structure, return with a big loss
        if not model_json:
            return 1e6

        # Train model: call KerasModel
        try:
            result = self.train(json.dumps(model_json), X_train, y_train, X_test, y_test)
        except Exception as e:
            print "Exception caught"
            print e
            return 1e6

        # Store results to Mongo (for netron UI)
        self.data_loader.save_result(self.exp_id, result)

        return result.history["val_loss"] if "val_loss" in result.history else 1e6

    def train(self, model_json, X_train, y_train, X_test, y_test, nb_epoch=10):
        print "Loading model..."
        self.model = model_from_json(model_json)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        print self.model.to_json()
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
            score = self.model.evaluate(X_test, y_test, verbose=1)
            print "Test score: %f\n Test accuracy: %f" % (score[0], score[1])
            res.history["val_loss"] = score[0]
            res.history["val_accuracy"] = score[1]
        return res

