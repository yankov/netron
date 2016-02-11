from netron.solvers import Solver

class GridSearch(Solver):
    def get_new_job(self, worker_id):
        return '{"model_type": "neural_net", "parameters": "some model params"}'


