from netron.solvers import Solver
from netron.grid import NeuralNetGrid
from sklearn.grid_search import ParameterGrid
import itertools

class GridSearch(Solver):
    def __init__(self, params_grid, input_shape, output_dim, model_type, data_filename):
        self.model_factory = self.get_model_factory(model_type)
        self.grid = NeuralNetGrid(params_grid, self.model_factory)
        self.models = self.generate_models(input_shape, output_dim)
        super(GridSearch, self).__init__(model_type, data_filename)

    def create_network_structures(self, layers, layers_num, input_shape):
        """Returns all combinations of given set of layers with given set of sizes"""
        for i in layers_num:
            for net_struct in itertools.product(layers, repeat=i):
                fixed_net_struct = self.model_factory.fix_or_skip(net_struct, input_shape)
                if fixed_net_struct:
                    yield fixed_net_struct
                else:
                    print "skipping invalid structure: %s" % "->".join(net_struct)
                    continue

    def generate_models(self, input_shape, output_dim):
        loss_type = self.grid.params_grid["loss"][0]
        for layers in self.create_network_structures(self.grid.params_grid["layers"], self.grid.params_grid["layer_nums"], input_shape):
            print "Current network: %s" % "->".join(layers)
            flat_params_grid = self.grid.create_flat_layers_grid(layers, input_shape, output_dim)
            for optimizer_name in self.grid.params_grid["optimizers"]:
                flat_grid = flat_params_grid.copy()
                flat_grid.update(self.grid.create_flat_optimizer_grid(optimizer_name))
                for params in ParameterGrid(flat_grid):
                    nn_params = self.grid.fold_params(params)
                    yield self.model_factory.create_model(layers, nn_params, loss_type)

# Example.
if __name__ == "__main__":
    job_stream = GridSearch(simple_params_grid, 1, 1, "keras", "sin_data.npz")
    for i in range(5):
        print "Model #" + str(i)
        print "=" * 10
        print job_stream.get_new_job(worker_id = 1)
