simple_params_grid = {"model": ["Sequential"],
                      "layers": ["Dense"],
                      "layer_nums": range(1, 4),
                      "optimizers": ["SGD"],
                      "loss": ["mean_absolute_error"],
                      "layer_params":
                          {"output_dim": [32, 64, 128, 256],
                           "activation": ["tanh", "linear", "sigmoid"]},
                           "optimizer_params":
                               {"decay": [1.1e+29],
                                "lr": [-4.555e+28],
                                "momentum":[-1.03e+34],
                                "name": ["SGD"],
                                "nesterov":[False]}}


