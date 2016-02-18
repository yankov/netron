import tornado
import json
import uuid

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
    def __init__(self, model_type, data_filename):
        self.model_type = model_type
        self.data_filename = data_filename
        self.experiment_id = str(uuid.uuid4())

    def create_job(self, model_params, refresh_data = False):
        return Job(self.experiment_id, self.model_type, model_params, self.data_filename, refresh_data)

    def get_new_job(self, worker_id):
        try:
            job = self.create_job(next(self.models))
            return job.to_json()
        except StopIteration:
            return '{"wait":"True", "experiment_id": "%s"}' % self.experiment_id

