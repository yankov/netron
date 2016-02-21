import tornado
import json
import uuid
from netron.models import KerasModelFactory
from netron.grid import *

class Job:
    def __init__(self, experiment_id, model_type, model_params, data_filename, refresh_data = False):
        self.experiment_id = experiment_id
        self.model_type = model_type
        self.model_params = model_params
        self.data_filename = data_filename
        self.refresh_data = refresh_data
        self.status = "created"

    def to_json(self):
        return json.dumps(self.__dict__)

class Solver(object):
    def __init__(self, grid_file_path, input_shape, output_dim, model_type, data_filename, *args):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_type = model_type
        self.data_filename = data_filename
        self.experiment_id = str(uuid.uuid4())
        self.params_grid = self.load_params_grid(grid_file_path)
        self.model_factory = self.get_model_factory(model_type)
        self.grid = NeuralNetGrid(self.params_grid, self.model_factory)
        self.models = self.generate_models(input_shape, output_dim)
        self.initialize(*args)

    def initialize(self, *args):
        """This to be overloaded in a child class if additional logic in constructor needed"""
        pass

    def load_params_grid(self, grid_file_path):
        with open(grid_file_path) as f:
            return json.loads(f.read())

    def create_job(self, model_params, refresh_data = False):
        return Job(self.experiment_id, self.model_type, model_params, self.data_filename, refresh_data)

    def get_new_job(self, worker_id):
        try:
            job = self.create_job(next(self.models))
            return job.to_json()
        except StopIteration:
            return '{"wait":"True", "experiment_id": "%s"}' % self.experiment_id

    def get_model_factory(self, model_type):
        if model_type == "keras":
            return KerasModelFactory()
        else:
            raise ValueError("%s is not supported. Only Keras models are supported right now." % model_type)

    def generate_models(input_shape, output_shape):
        """This method should be overloaded in a child class.
        Yields models that will be loaded by workers"""
        pass
