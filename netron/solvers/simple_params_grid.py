simple_params_grid = \
    {"model": ["Sequential"],
     "layers": ["Dense"],
     "layer_nums": range(1, 5),
     "optimizers": ["SGD"],
     "loss": ["mean_absolute_error"],
     "layer_params":
        {"output_dim": [32, 64, 128, 256, 512],
         "activation": ["tanh", "linear", "sigmoid", "relu"],
         "p":[0.25, 0.5]
        },
         "optimizer_params":
            {"decay": [1e-6],
             "lr": [0.1],
             "momentum":[0.9],
             "name": ["SGD"],
             "nesterov":[True]}}


