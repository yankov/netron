import tornado
import json

class Job:
    def __init__(self, model_type, model_params, data_filename, refresh_data = False):
        self.model_type = model_type
        self.model_params = model_params
        self.data_filename = data_filename
        self.refresh_data = refresh_data
        self.status = "created"

    def to_json(self):
        return tornado.escape.json_encode(self.__dict__)

class Solver(object):
    def __init__(self, model_type, data_filename):
        self.model_type = model_type
        self.data_filename = data_filename

    def create_job(self, model_params, refresh_data = False):
        return Job(self.model_type, model_params, self.data_filename, refresh_data)
