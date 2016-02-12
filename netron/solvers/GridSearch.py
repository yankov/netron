from netron.solvers import Solver
import tornado
import json
import os

class GridSearch(Solver):
    def get_new_job(self, worker_id):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/model_example.json") as f:
            model = f.read()
        job = {"model_type": "neural_net", "model": model}
        return tornado.escape.json_encode(job)


