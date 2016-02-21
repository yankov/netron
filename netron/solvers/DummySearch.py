from netron.solvers import *
import json
import os

class DummySearch(Solver):
    """Basic example of a searcher. Just returns same parameters each time."""

    def initialize(self):
        with open(os.path.dirname(os.path.abspath(__file__)) + "/model_example.json") as f:
            self.model_params = f.read()

    def generate_models(self, input_shape, output_shape):
        job = self.create_job(self.model_params)
        for i in range(10):
            yield job.to_json()


