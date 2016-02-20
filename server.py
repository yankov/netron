from netron.server import JobHTTPServer
from netron.server import JobManager
from netron.solvers import DummySearch, GridSearch, RandomSearch, simple_params_grid
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, required=True, help="Input dimensions for a first layer (dimensions of the training sample)")
parser.add_argument('--output_dim', type=int, required=True, help="Output dimensions for a last layer (number of predicting classes for example)")
parser.add_argument('--data', required=True, help="Name of the npz file with X_train and y_train")
parser.add_argument('--solver', required=True, help="A solver to search through a parameter space: Currently only RandomSearch or GridSearch")
parser.add_argument('--grid', required=True, help="A json file with a parameter grid. Example: simple_params_grid.json")
parser.add_argument('--port', type=int, default=8080, required=False, help="A port server should be listening on. Default is 8080.")
args = parser.parse_args()


with open(args.grid) as f:
    params_grid = json.loads(f.read())

if args.solver == "GridSearch":
    solver = GridSearch(params_grid, args.input_dim, args.output_dim, "keras", args.data)
elif args.solver == "RandomSearch":
    solver = RandomSearch(params_grid, args.input_dim, args.output_dim, 10, "keras", args.data)
else:
    raise ValueError("This solver is not supported. Only possible values for --solver right now are GridSearch or RandomSearch")


job_manager = JobManager(solver)
print "Started a server with %s solver and %s dataset" % (args.solver, args.data)

server = JobHTTPServer(args.port, job_manager)
server.start()
