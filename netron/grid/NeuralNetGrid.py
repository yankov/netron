class NeuralNetGrid:
    """Creates a parameter grid (search space) for a given network structure"""

    def __init__(self, params_grid, model_factory):
        # Grid with a set a possible parameters and values that they may take.
        # Example: {"a": [1,2,3], "b": ["sin", "cos", "tan"]}
        self.params_grid = params_grid
        self.model_factory = model_factory

    def create_flat_layers_grid(self, layer_types, input_shape, output_dim):
        """Creates a grid of all possible parameters names for every layer
        for a given network structure. Prepends a layer number to a parameter name. """

        flat_params_grid = {}
        for i, layer_type in enumerate(layer_types):
            matched_keys = set(self.model_factory.get_layer_params(layer_type)) & set(self.params_grid["layer_params"])
            layer_params = dict([(str(i) + ":" + key, self.params_grid["layer_params"][key]) for key in matched_keys])

            # Input shape of the first layer should be always the same
            if i == 0:
                layer_params["0:input_shape"] = [input_shape]

            # Output size of the last layer should be always the same
            if i == len(layer_types) - 1:
                layer_params[str(i) + ":output_dim"] = [output_dim]

            flat_params_grid.update(layer_params)

        return flat_params_grid

    def create_flat_optimizer_grid(self, optimizer_name):
        """Creates a grid of all possible parameter values for a given optimizer"""

        flat_params_grid = {}
        matched_keys = set(self.model_factory.get_optimizer_params(optimizer_name)) & set(self.params_grid["optimizer_params"])
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


