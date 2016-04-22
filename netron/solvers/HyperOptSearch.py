from netron.worker import KerasModel
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
import hyperopt.pyll.stochastic
import pymongo
from pymongo import MongoClient
import json
import random

class HyperOptSearch():
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

    def first_layer(self, layer):
        layer["config"]["input_shape"] = self.input_shape
        return layer

    def last_layer(self, layer, activation = None):
        layer["config"]["output_dim"] = self.output_dim
        if activation:
            layer["config"]["activation"] = activation

        return layer

    def dense_space(self, input_shape = None, output_dim = None, activation = None):
        unq = "|" + str(random.randint(1, 99999))

        return  {
            'class_name': 'Dense',
            'config': {
                'output_dim': hp.choice("output_dim" + unq, [32, 64, 128, 256, 512]),
                'activation': hp.choice("activation" + unq, ["linear", "relu", "softmax"])
             }
        }

    def convolution2d_space(self):
        unq = "|" + str(random.randint(1, 99999))
        return {
            'class_name': 'Convolution2D',
            'config': {
                'nb_filter': hp.choice("nb_filter" + unq, [32, 64, 128, 512]),
                'nb_col': 3,
                'nb_row': 3
             }
        }

    def maxpooling2d_space(self):
        return {
            'class_name': 'MaxPooling2D',
            'config': {
                'pool_size': (2,2)
             }
        }

    def flatten_space(self):
        return {
            'class_name':'Flatten',
            'config': {}
        }

    def zeropadding2d_space(self):
        return {
            'class_name': 'ZeroPadding2D',
            'config': {
                'padding': (1, 1)
             }
        }

    def dropout_space(self):
        unq = "|" + str(random.randint(1, 99999))

        return {
           'class_name': 'Dropout',
           'config': {
               'p': hp.choice('p' + unq, [0.25, 0.5])
            }
        }

    def sgd_space(self):
        return {
           'class_name': 'SGD',
           'decay': 1e-6,
           'lr': 0.01,
           'momentum': 0.9,
           'nesterov': True
        }

    def adadelta_space(self):
        return  {
            'class_name': 'Adadelta',
            'config': {
                'lr': 0.1,
                'epsilon': 1e-06,
                'rho': 0.95
             }
        }

    def conv_layers_spaces(self):
        return [self.dense_space(), self.convolution2d_space(), self.maxpooling2d_space(), self.zeropadding2d_space(), self.flatten_space(), self.dropout_space()]

    def conv_model_space(self, n_layers, data_filename):
        layers = [hp.choice('l0', [self.first_layer(self.dense_space()), self.first_layer(self.convolution2d_space())])]
        for i in range(n_layers-2):
            layers.append(hp.choice("l%d" % (i+1), self.conv_layers_spaces()))

        layers.append(hp.choice("l%d" % (n_layers - 1), [self.last_layer(self.dense_space()), self.last_layer(self.convolution2d_space())]))

        # Full model search space for layer size `n_layers`
        return {
            'config': layers,
            'loss': 'categorical_crossentropy',
            'class_name': 'Sequential',
            'class_mode': 'categorical',
            #'optimizer': hp.choice("optimizer", [self.sgd_space()]),
            'data_filename': data_filename
        }

    def create_mongo_trials(self, mongo_uri):
        jobs_col = MongoClient(mongo_uri)["netron"]["hyperopt_jobs"]
        last_job = jobs_col.find({}, {"exp_key": 1}).sort("exp_key", pymongo.DESCENDING).limit(1)
        last_job = list(last_job)
        if len(last_job) > 0:
            exp_key = int(last_job[0]["exp_key"]) + 1
        else:
            exp_key = 0
        print "Current experiment key is %s" % exp_key

        mongo_uri = mongo_uri + 'netron/hyperopt_jobs'
        return exp_key, MongoTrials(mongo_uri, exp_key=exp_key)

    def start_search_server(self, mongo_uri, data_filename, layers_num, max_evals=100, nb_epoch = 10, patience = 5):
        space = self.conv_model_space(layers_num, data_filename)
        exp_key, trials = self.create_mongo_trials(mongo_uri)
        keras_model = KerasModel(exp_key, nb_epoch = nb_epoch, patience = patience, mongo_uri = mongo_uri)
        best = fmin(keras_model.run_job, space, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        print "Done!"
        print best

if __name__ == "__main__":
    print "HyperOpt search. Example of a valid model for mnist."
    h = HyperOptSearch(input_shape=[1,28,28], output_dim=10)
    model = None
    keras_factory = KerasModelFactory()
    while not model:
        model = hyperopt.pyll.stochastic.sample(h.conv_model_space(6))
        model = keras_factory.build_from_sketch(model, [1,28,28])

    print json.dumps(model)
