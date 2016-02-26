# -*- coding: utf-8 -*-
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
import boto3

class Worker(object):
    # Seconds before retrying a request to server in seconds
    POLL_INTERVAL = 1

    # Seconds to wait when there are no new jobs
    NO_JOB_WAIT= 60

    def __init__(self, api_url, mongo_uri, mongo_port = 27017, **kwargs):
        self.url = api_url
        self.name = socket.gethostname()
        self.status = "idle"
        self.http_client = AsyncHTTPClient()
        self.models = {"keras": KerasModel(**kwargs)}
        self.data_files = {}
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client['netron']
        self.experiments_col = self.db["experiments"]
        self.start_time = datetime.datetime.now()

    def json_decode(self, body):
        json_body = json.loads(body.decode("utf-8"))
        # This needs to be called two times on escaped json string. ¯\_(ツ)_/¯
        if not isinstance(json_body, dict):
            return json.loads(json_body)

        return json_body

    @gen.coroutine
    def download_file_from_server(self, filename):
        result = yield self.http_client.fetch(self.url + "/data/" + filename)
        data_file = os.path.join(self.data_path, filename)
        with open(data_file, "w") as f:
            f.write(result.body)
        data = np.load(data_file)
        raise gen.Return(data)

    def download_file_from_s3(self, filename, refresh):
        path = filename.split("//")[1].split("/")
        bucket = path[0]
        filename = "/".join(path[1:])
        data_file = os.path.join(self.data_path, path[-1])
        if os.path.isfile(data_file) and not refresh:
            return np.load(data_file)

        key = boto3.resource('s3').Object(bucket, filename).get()
        with open(data_file, "w") as f:
            for chunk in iter(lambda: key['Body'].read(4096), b''):
                f.write(chunk)

        return np.load(data_file)

    @gen.coroutine
    def load_data(self, filename, refresh):
        if filename in self.data_files and not refresh:
            raise gen.Return(self.data_files[filename])

        print "Getting data from the server"
        if "s3://" in filename:
            data = self.download_file_from_s3(filename, refresh)
        else:
            data = yield self.download_file_from_server(filename)

        self.data_files[filename] = [data["X_train"], data["y_train"]]

        # Add a test set if it's there
        if "X_test" in data and "y_test" in data:
            self.data_files[filename].extend([data["X_test"], data["y_test"]])
        else:
            self.data_files[filename].extend([None, None])

        raise gen.Return(self.data_files[filename])


    @gen.coroutine
    def get_new_job(self):
        try:
            response = yield self.http_client.fetch(self.url + "/worker/" + self.name + "/job")
            job = self.json_decode(response.body)

            # no jobs, wait 1 min before next request.
            if "wait" in job:
                print "Experiment %s is done! There are no new jobs yet." % job["experiment_id"]
                IOLoop.current().call_later(self.NO_JOB_WAIT, lambda: self.get_new_job())
                return

            # if job type is unsupported (only keras supported at the moment)
            if job["model_type"] not in self.models:
                raise ValueError("Only the following models are supported right now: " + ", ".join(self.models.keys()))

            # load data from a numpy archive with a given name
            X_train, y_train, X_test, y_test = yield self.load_data(job["data_filename"], job["refresh_data"])

            # train the model
            result = self.models[job["model_type"]].run_job(json.dumps(job["model_params"]), X_train, y_train, X_test, y_test)
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
            "val_loss": result.history["val_loss"] if "val_loss" in result.history else None,
            "val_accuracy": result.history["val_accuracy"] if "val_accuracy" in result.history else None,
            "params": result.params,
            "model_params": json.loads(result.model.to_json()),
            "created_at": datetime.datetime.utcnow()})

    def close(self):
        self.http_client.close()


if __name__ == "__main__":
    worker = Worker("http://localhost:8080", mongo_uri = "mongodb://localhost:27017/", nb_epoch = 10, patience = 5)
    worker.get_new_job()
    IOLoop.current().start()
