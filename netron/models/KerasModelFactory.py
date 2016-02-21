from keras.layers.core import *
from keras.layers.convolutional import *
from keras.optimizers import *
from keras.models import Sequential
import json

class KerasModelFactory:
    LAYERS = {
        "Dense": {"config": Dense(1).get_config(), "first_ok": True, "last_ok": True, "input_ndim":2, "output_ndim":2},
        "Dropout": {"config": Dropout(0).get_config(), "first_ok": False, "last_ok": False, "input_ndim":-1, "output_ndim":-1},
        "Convolution2D": {"config": Convolution2D(1,1,1).get_config(), "first_ok": True, "last_ok": False, "input_ndim":4, "output_ndim":4},
        "MaxPooling2D": {"config": MaxPooling2D().get_config(), "first_ok": False, "last_ok": False, "input_ndim":4, "output_ndim":4},
        "Flatten": {"config": Flatten().get_config(), "first_ok": False, "last_ok": False, "input_ndim":-1, "output_ndim":2},
    }

    OPTIMIZERS = {
        "SGD": SGD().get_config(),
        "Adadelta": Adadelta().get_config()
    }

    # Layers that cannot be chained to themselves.
    # Example: Dropout -> Dropout
    NO_DUP_LAYERS = set(["Flatten", "Dropout"])

    def __init__(self):
        self.empty_model = Sequential()

    def get_layer_config(self, layer_type):
        return self.LAYERS[layer_type]["config"]

    def get_layer_params(self, layer_type):
        return self.get_layer_config(layer_type).keys()

    def get_optimizer_config(self, optimizer_type):
        return self.OPTIMIZERS[optimizer_type]

    def get_optimizer_params(self, optimizer_type):
        return self.get_optimizer_config(optimizer_type).keys()

    def create_model(self, layers, nn_params, loss_type):
        model_json = json.loads(self.empty_model.to_json())

        for i, layer_type in enumerate(layers):
            # Default config for a layer
            model_json["layers"].append(self.LAYERS[layer_type]["config"].copy())

            # Set values from a grid
            if i in nn_params:
                for k in nn_params[i]:
                    model_json["layers"][i][k] = nn_params[i][k]

        model_json["optimizer"] = nn_params["optimizer"]

        # It makes sense only to use one loss per experiment
        model_json["loss"] = loss_type
        model_json["class_mode"] = "categorical"
        return model_json


    def fix_or_skip(self, layers, input_shape):
        """Validates the network's structure and attempts to fix it if possible"""

        # Skip if input_shape of the data doesn't match the input ndim of the first layer
        if len(input_shape) > self.LAYERS[layers[0]]["input_ndim"]:
            return False
        # Skip this structure if this type of layer cannot be the first layer
        if not self.LAYERS[layers[0]]["first_ok"]:
            return False

        # Skip this structure if this type of layer cannot be the last layer
        if not self.LAYERS[layers[-1]]["last_ok"]:
            return False

        dup_keys = set()
        for i in range(1, len(layers)):
            layer_conf = self.LAYERS[layers[i]]
            prev_layer_conf = self.LAYERS[layers[i-1]]

            # Some type of layers cannot duplicate like Dropout->Dropout.
            if layers[i] == layers[i-1]:
                dup_keys.add(layers[i])

            # Flatten out input for 2D layers.
            if layer_conf["input_ndim"] == 2 and  (prev_layer_conf["output_ndim"] != layer_conf["input_ndim"]):
                layers = list(layers)
                layers.insert(i, "Flatten")
                layers = tuple(layers)
            # If previous layer can have any output ndim find ndim of the layer before that and check if it matches
            # input ndim of the current layer.
            elif layer_conf["input_ndim"] != -1 and prev_layer_conf["output_ndim"] == -1:
                j = i - 2
                while j > 0:
                    if self.LAYERS[layers[j]]["output_ndim"] == -1:
                        j -= 1
                    elif self.LAYERS[layers[j]]["output_ndim"] != layer_conf["input_ndim"]:
                        return False
                    else:
                        break
            # Skip, if current layer input ndim doesn't match previous layer output ndim.
            elif layer_conf["input_ndim"] != -1 and (prev_layer_conf["output_ndim"] != layer_conf["input_ndim"]):
                return False

        # Skip if there are not allowed duplicate layers (Dropout->Dropout)
        if self.NO_DUP_LAYERS & dup_keys:
            return False

        return layers

