from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import SGD
import json
import itertools

#TODO: Generalize it so it applies not only to Keras and neural nets.

class NeuralNetParamGrid:
    LAYERS = {
        "Dense": Dense(1).get_config(),
        "Dropout": Dropout(0).get_config()
    }

    OPTIMIZERS = {
        "SGD": SGD().get_config()
    }

    def __init__(self, params_grid, grid_walker):
        # Grid with a set a possible parameters and values that they may take.
        # Example: {"a": [1,2,3], "b": ["sin", "cos", "tan"]}
        self.params_grid = params_grid

        # Walker is a object that generates samples given a grid of possible parameters.
        # Simple example of walker that can be passed is ParameterGrid or ParameterSampler
        # from scikitlearn.
        self.grid_walker = grid_walker

    def create_network_structures(self, layers, layer_nums):
        """Returns all combinations of given set of layers with given set of sizes"""

        arch = [list(itertools.product(layers, repeat=i)) for i in layer_nums]
        # Flatten the list
        return [b for a in arch for b in a]


    def create_flat_layers_grid(self, layer_types, input_dim, output_dim):
        """Creates a grid of all possible parameters names for every layer
        for a given network structure. Prepends a layer number to a parameter name. """

        flat_params_grid = {}
        for i, layer_type in enumerate(layer_types):
            matched_keys = set(self.LAYERS[layer_type].keys()) & set(self.params_grid["layer_params"])
            layer_params = dict([(str(i) + ":" + key, self.params_grid["layer_params"][key]) for key in matched_keys])

            # Input size of the first layer should be always the same
            if i == 0:
                layer_params["0:input_dim"] = [input_dim]

            # Output size of the last layer should be always the same
            if i == len(layer_types) - 1:
                layer_params[str(i) + ":output_dim"] = [output_dim]

            flat_params_grid.update(layer_params)

        return flat_params_grid

    def create_flat_optimizer_grid(self, optimizer_name):
        """Creates a grid of all possible parameter values for a given optimizer"""

        flat_params_grid = {}
        matched_keys = set(self.OPTIMIZERS[optimizer_name].keys()) & set(self.params_grid["optimizer_params"])
        optimizer_params = dict([("optimizer:" + key, self.params_grid["optimizer_params"][key]) for key in matched_keys])
        flat_params_grid.update(optimizer_params)

        return flat_params_grid

    def fold_params(self, params):
        """Un-flattens the dictionary of parameters so that parameters at each layer
        can be accessed using index:
        Input: {'0:activation': 'tanh', '0:output_dim': 128}
        Output:{'0' : {'activation': 'tanh', 'output_dim': 128}}"""

        model_params = {"optimizer":{}}
        for k in params:
            i, param = k.split(":")
            if i == "optimizer":
                model_params[i][param] = params[k]
            else:
                i = int(i)
                if i not in model_params:
                    model_params[i] = {}
                model_params[i][param] = params[k]

        return model_params

    def create_model(self, empty_model, layers, nn_params):
        model_json = json.loads(empty_model.to_json())

        for i, layer_type in enumerate(layers):
            model_json["layers"].append(self.LAYERS[layer_type].copy())
            for k in nn_params[i]:
                model_json["layers"][i][k] = nn_params[i][k]

        model_json["optimizer"] = nn_params["optimizer"]

        # It makes sense only to use one loss per experiment
        model_json["loss"] = self.params_grid["loss"][0]
        model_json["class_mode"] = "categorical"
        return model_json

    def generate_models(self, input_dim, output_dim):
        empty_model = Sequential()
        for layers in self.create_network_structures(self.params_grid["layers"], self.params_grid["layer_nums"]):
            flat_params_grid = self.create_flat_layers_grid(layers, input_dim, output_dim)
            for optimizer_name in self.params_grid["optimizers"]:
                flat_grid = flat_params_grid.copy()
                flat_grid.update(self.create_flat_optimizer_grid(optimizer_name))
                for params in self.grid_walker(flat_grid):
                    nn_params = self.fold_params(params)
                    yield self.create_model(empty_model, layers, nn_params)


