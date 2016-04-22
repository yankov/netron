# -*- coding: utf-8 -*-
from netron.worker import KerasModel
from netron.worker import DataLoader
import tornado
from tornado.httpclient import HTTPClient, AsyncHTTPClient
from tornado.httpclient import HTTPError
from tornado.ioloop import IOLoop
from tornado import gen
import numpy as np
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

    def __init__(self, api_url, mongo_uri, **kwargs):
        self.url = api_url
        self.name = socket.gethostname()
        self.http_client = AsyncHTTPClient()
        self.models = {"keras": KerasModel(**kwargs)}
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
        self.data_loader = DataLoader(api_url, mongo_uri)

    def json_decode(self, body):
        json_body = json.loads(body.decode("utf-8"))
        # This needs to be called two times on escaped json string. ¯\_(ツ)_/¯
        if not isinstance(json_body, dict):
            return json.loads(json_body)

        return json_body

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

            # TODO: clean it up! terrible, I know..
            model = self.models[job["model_type"]]
            model.exp_id = job["experiment_id"]
            job["model_params"]["data_filename"] = job["data_filename"]
            result = model.run_job(job["model_params"])

            # Get a new job from server
            IOLoop.current().call_later(1, lambda: self.get_new_job())

        except HTTPError as e:
            print("Error: " + str(e))
            IOLoop.current().call_later(self.POLL_INTERVAL, lambda: self.get_new_job())
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Error: " + str(e))
            IOLoop.current().call_later(self.POLL_INTERVAL, lambda: self.get_new_job())

    def close(self):
        self.http_client.close()


