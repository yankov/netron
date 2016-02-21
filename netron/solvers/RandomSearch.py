from netron.solvers import Solver
from netron.grid import NeuralNetGrid
from sklearn.grid_search import ParameterSampler, ParameterGrid
import random
import itertools

class RandomSearch(Solver):
    # If we sample more than this number of already seen networks
    # consecutively, then skip to the next network size
    STRUCT_DUP_THRESHOLD = 100

    def initialize(self, params_sample_size, structure_sample_size):
        self.params_sample_size = params_sample_size
        self.structure_sample_size = structure_sample_size
        self.seen_structures = set()

    def random_product(self, *args, **kwds):
        "Random selection from itertools.product(*args, **kwds)"
        pools = map(tuple, args) * kwds.get('repeat', 1)
        return tuple(random.choice(pool) for pool in pools)

    def create_network_structures(self, layers, layers_num, input_shape):
        """Returns all combinations of given set of layers with given set of sizes"""
        for i in layers_num:
            j = 0
            dups = 0
            while j < self.structure_sample_size:
                net_struct = self.random_product(layers, repeat=i)
                fixed_net_struct = self.model_factory.fix_or_skip(net_struct, input_shape)
                struct_hash = hash(fixed_net_struct)
                if fixed_net_struct:
                    if struct_hash not in self.seen_structures:
                        self.seen_structures.add(struct_hash)
                        j += 1
                        dups = 0
                        yield fixed_net_struct
                    else:
                        print "Skipping structure that has already been trained"
                        dups += 1
                        if dups > self.STRUCT_DUP_THRESHOLD:
                            # We probably seen all structures with this size, so skip to the next size.
                            break
                else:
                    #print "skipping invalid structure: %s" % "->".join(net_struct)
                    continue

    def generate_models(self, input_shape, output_dim):
        loss_type = self.grid.params_grid["loss"][0]
        for layers in self.create_network_structures(self.grid.params_grid["layers"], self.grid.params_grid["layer_nums"], input_shape):
            print "Current network: %s" % "->".join(layers)
            flat_params_grid = self.grid.create_flat_layers_grid(layers, input_shape, output_dim)
            for optimizer_name in self.grid.params_grid["optimizers"]:
                flat_grid = flat_params_grid.copy()
                flat_grid.update(self.grid.create_flat_optimizer_grid(optimizer_name))

                n_samples = min(self.params_sample_size, len(ParameterGrid(flat_grid)))
                for params in ParameterSampler(flat_grid, n_samples):
                    nn_params = self.grid.fold_params(params)
                    yield self.model_factory.create_model(layers, nn_params, loss_type)

# Example.
if __name__ == "__main__":
    import sys

    job_stream = RandomSearch(sys.argv[1], [1, 28, 28], 10, "keras", "sin_data.npz", 1, 5)
    for i in range(5):
        print "Model #" + str(i)
        print "=" * 10
        print job_stream.get_new_job(worker_id = 1)
