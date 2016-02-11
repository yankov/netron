from netron.worker import NNModel
import tornado
from tornado.httpclient import HTTPClient, AsyncHTTPClient
from tornado.httpclient import HTTPError
from tornado.ioloop import IOLoop
from tornado import gen
import socket
import json

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
            task = tornado.escape.json_decode(response.body)

            #if task["model_type"] == "neural_net":
            #    if "neural_net" not in self.models:
            #        self.models["neural_net"] = NNModel()
            #    result = self.models["neural_net"].load_task(task["parameters"])

            result = self.models["neural_net"].load_task(task)
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
