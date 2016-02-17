from netron.worker import KerasModel
import tornado
from tornado.httpclient import HTTPClient, AsyncHTTPClient
from tornado.httpclient import HTTPError
from tornado.ioloop import IOLoop
from tornado import gen
import numpy as np
from pymongo import MongoClient
import os
import socket
import json
import datetime

class Worker(object):
    # Time before retrying a request to server in seconds
    POLL_INTERVAL = 10

    def __init__(self, api_url, mongo_server, mongo_port = 27017):
        self.url = api_url
        self.name = socket.gethostname()
        self.status = "idle"
        self.http_client = AsyncHTTPClient()
        self.models = {"keras": KerasModel()}
        self.data_files = {}
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
        self.mongo_client = MongoClient(mongo_server, mongo_port)
        self.db = self.mongo_client['netron']
        self.experiments_col = self.db["experiments"]

    @gen.coroutine
    def load_data(self, filename, refresh):
        if filename in self.data_files and not refresh:
            raise gen.Return(self.data_files[filename])

        print "Getting data from the server"
        result = yield self.http_client.fetch(self.url + "/data/" + filename)
        data_file = os.path.join(self.data_path, filename)
        with open(data_file, "w") as f:
            f.write(result.body)
        data = np.load(data_file)
        self.data_files[filename] = [data["x_train"], data["y_train"]]

        raise gen.Return(self.data_files[filename])


    @gen.coroutine
    def get_new_job(self):
        try:
            response = yield self.http_client.fetch(self.url + "/worker/" + self.name + "/job")

            # Why the hell I have to decode it 2 times? If I don't, after first decode it's
            # still a string.
            job = json.loads(tornado.escape.json_decode(response.body.decode('utf-8')))

            if job["model_type"] not in self.models:
                raise ValueError("Only the following models are supported right now: " + ", ".join(self.models.keys()))

            x_train, y_train = yield self.load_data(job["data_filename"], job["refresh_data"])
            result = self.models[job["model_type"]].run_job(json.dumps(job["model_params"]), x_train, y_train)
            self.save_result(job["experiment_id"], result)

            # Get a new job from server
            IOLoop.current().call_later(1, lambda: self.get_new_job())

        except HTTPError as e:
            print("Error: " + str(e))
            IOLoop.current().call_later(self.POLL_INTERVAL, lambda: self.get_new_job())
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Error: " + str(e))
            IOLoop.current().call_later(self.POLL_INTERVAL, lambda: self.get_new_job())

    def save_result(self, experiment_id, result):
        """ Saves the results of training to MongoDB """
        self.experiments_col.insert({
            "experiment_id": experiment_id,
            "loss": result.history["loss"],
            "params": result.params,
            "model_params": json.loads(result.model.to_json()),
            "created_at": datetime.datetime.utcnow()})

    def close(self):
        self.http_client.close()


if __name__ == "__main__":
    worker = Worker("http://localhost:8080", mongo_server = "localhost", mongo_port = 27017)
    worker.get_new_job()
    IOLoop.current().start()
