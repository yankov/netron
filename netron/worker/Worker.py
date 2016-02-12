from netron.worker import NNModel import tornado
from tornado.httpclient import HTTPClient, AsyncHTTPClient
from tornado.httpclient import HTTPError
from tornado.ioloop import IOLoop
from tornado import gen
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

    @gen.coroutine
    def get_new_job(self):
        try:
            response = yield self.http_client.fetch(self.url + "/worker/" + self.name + "/job")

            # Why the hell I have to decode it 2 times? If I don't, after first decode it's
            # still a string.
            task = json.loads(tornado.escape.json_decode(response.body.decode('utf-8')))

            if task["model_type"] == "neural_net":
                if "neural_net" not in self.models:
                    self.models["neural_net"] = NNModel()

                x_train = np.random.rand(10000, 1) * 20 - 10
                result = self.models["neural_net"].load_task(task["model"], x_train, np.sin(x_train))

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
