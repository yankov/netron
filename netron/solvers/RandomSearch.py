from netron.solvers import *
from sklearn.grid_search import ParameterSampler

class RandomSearch(Solver):
    def __init__(self, params_grid, input_dim, output_dim, n_samples, model_type, data_filename):
        self.grid = NeuralNetParamGrid(params_grid, self.random_walker(n_samples))
        self.models = self.grid.generate_models(input_dim, output_dim)
        super(RandomSearch, self).__init__(model_type, data_filename)

    def random_walker(self, n_samples):
        def f(params):
            return ParameterSampler(params, n_samples)

        return f

# Example.
if __name__ == "__main__":
    job_stream = RandomSearch(simple_params_grid, 1, 1, 3, "keras", "sin_data.npz")
    for i in range(5):
        print "Model #" + str(i)
        print "=" * 10
        print job_stream.get_new_job(worker_id = 1)
