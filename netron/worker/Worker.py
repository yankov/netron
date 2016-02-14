from netron.worker import NNModel
import tornado
from tornado.httpclient import HTTPClient, AsyncHTTPClient
from tornado.httpclient import HTTPError
from tornado.ioloop import IOLoop
from tornado import gen
import os
import socket
import json
import numpy as np

class Worker(object):
    # Time before retrying a request to server in seconds
    POLL_INTERVAL = 10

    def __init__(self, server, port):
        self.server = server
        self.port = port
        self.url = "http://{0}:{1}".format(server, port)
        self.name = socket.gethostname()
        self.status = "idle"
        self.http_client = AsyncHTTPClient()
        self.models = {"neural_net": NNModel()}
        self.data_files = {}
        self.data_path = os.path.join(os.path.dirname(__file__), "data")

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
            x_train, y_train = yield self.load_data(job["data_filename"], job["refresh_data"])

            if job["model_type"] == "neural_net":
                if "neural_net" not in self.models:
                    self.models["neural_net"] = NNModel()

                result = self.models["neural_net"].run_job(job["model_params"], x_train, y_train)

                # TODO: Store the results possibly here

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


if __name__ == "__main__":
    worker = Worker("localhost", 8080)
    worker.get_new_job()
    IOLoop.current().start()
