from netron.solvers import *
import tornado
import json
import os

class DummySearch(Solver):
    """Basic example of a searcher. Just returns same parameters
    each time.
    """
    def __init__(self, *args):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/model_example.json") as f:
            self.model_params = f.read()
        super(DummySearch, self).__init__(*args)

    def get_new_job(self, worker_id):
        job = self.create_job(self.model_params)
        return job.to_json()


