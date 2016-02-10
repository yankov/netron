from netron.server import JobManager
from netron.solvers import GridSearch
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
import json

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

class ResultsHandler(RequestHandler):
    def get(self):
        self.write("Stat page with the results of training")

class JobHTTPServer(object):
    def __init__(self, port, job_manager):
        self.job_manager = job_manager

        self.routes = Application([
            (r"/", MainRequestHandler),
            (r"/worker/(.*)/job", JobHandler, {"job_manager": job_manager}),
            (r"/results", ResultsHandler)
        ])

        self.port = port

    def start(self):
        server = HTTPServer(self.routes)
        server.listen(self.port)
        IOLoop.current().start()

# Example
if __name__ == "__main__":
    job_manager = JobManager(solver = GridSearch())
    server = JobHTTPServer(8080, job_manager)
    server.start()
