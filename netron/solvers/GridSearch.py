from netron.solvers import *
from sklearn.grid_search import ParameterGrid

class GridSearch(Solver):
    def __init__(self, params_grid, input_dim, output_dim, model_type, data_filename):
        self.grid = NeuralNetParamGrid(params_grid, ParameterGrid)
        self.models = self.grid.generate_models(input_dim, output_dim)
        super(GridSearch, self).__init__(model_type, data_filename)

# Example.
if __name__ == "__main__":
    job_stream = GridSearch(simple_params_grid, 1, 1, "keras", "sin_data.npz")
    for i in range(5):
        print "Model #" + str(i)
        print "=" * 10
        print job_stream.get_new_job(worker_id = 1)
