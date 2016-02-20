from netron.server import JobManager
from netron.solvers import DummySearch, GridSearch, RandomSearch, simple_params_grid
from netron.server import TrainStats
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
import tornado
import json
import os

class MainRequestHandler(RequestHandler):
    def get(self):
        self.write("Server is ok")

class JobHandler(RequestHandler):
    def initialize(self, job_manager):
        self.job_manager = job_manager

    def get(self, worker_id):
        job = self.job_manager.get_new_job(worker_id)
        self.write(json.dumps(job))

    def post(self, worker_id):
        if self.request.body:
            body = json.loads(self.request.body)
            ret = self.job_manager.save_results(worker_id, body)
            self.finish('ok')
        else:
            self.finish("error: no data")

class StatsHandler(RequestHandler):
    def initialize(self):
        self.stats = TrainStats()

    def get(self, experiment_id = None):
        if not experiment_id:
            self.render("index.html", **self.stats.get_all_experiments())
        else:
            self.render("experiment.html", **self.stats.get_stats(experiment_id))

class JobHTTPServer(object):
    def __init__(self, port, job_manager):
        self.job_manager = job_manager
        self.static_path = os.path.join(os.path.dirname(__file__), "static")

        self.routes = Application(
        [
            (r"/", MainRequestHandler),
            (r"/worker/(.*)/job", JobHandler, {"job_manager": job_manager}),
            (r"/stats", StatsHandler),
            (r"/stats/(.*)", StatsHandler),
            (r"/data/(.*)", tornado.web.StaticFileHandler, {'path': self.static_path})
            ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path= self.static_path)

        self.port = port

    def start(self):
        server = HTTPServer(self.routes)
        server.listen(self.port)
        IOLoop.current().start()

# Example
if __name__ == "__main__":
    job_manager = JobManager(solver = RandomSearch(simple_params_grid, 1, 1, 10, "keras", "mnist_train.npz"))
    #job_manager = JobManager(solver = GridSearch(simple_params_grid, 1, 1, "keras", "sin_data.npz"))
    server = JobHTTPServer(8080, job_manager)
    server.start()
