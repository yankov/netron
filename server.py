from netron.server import JobHTTPServer
from netron.server import JobManager
from netron.solvers import DummySearch, GridSearch, RandomSearch, simple_params_grid
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_shape', required=True, help="Input shape for a first layer (dimensions of the training sample)")
parser.add_argument('--output_dim', type=int, required=True, help="Output dimensions for a last layer (number of predicting classes for example)")
parser.add_argument('--data', required=True, help="Name of the npz file with X_train and y_train")
parser.add_argument('--solver', required=True, help="A solver to search through a parameter space: Currently only RandomSearch or GridSearch")
parser.add_argument('--grid', required=True, help="A json file with a parameter grid. Example: simple_params_grid.json")
parser.add_argument('--port', type=int, default=8080, required=False, help="A port server should be listening on. Default is 8080.")
parser.add_argument('--params_sample_size', type=int, required=False, help="Only for RandomSearch: parameter sample size per network structure.")
parser.add_argument('--structure_sample_size', type=int, required=False, help="Only for RandomSearch: network structure sample size per given number of layers.")
args = parser.parse_args()

input_shape = [int(dim) for dim in args.input_shape.split(",")]

if args.solver == "GridSearch":
   solver = GridSearch(args.grid, input_shape, args.output_dim, "keras", args.data)
elif args.solver == "RandomSearch":
    if not args.params_sample_size or not args.structure_sample_size:
        raise ValueError("--params_sample_size  and --structure_sample_size must be used with RandomSearch")
    solver = RandomSearch(args.grid, input_shape, args.output_dim, "keras", args.data, args.params_sample_size,
                          args.structure_sample_size)
else:
    raise ValueError("This solver is not supported. Only possible values for --solver right now are GridSearch or RandomSearch")


job_manager = JobManager(solver)
print "Started a server with %s solver and %s dataset" % (args.solver, args.data)

server = JobHTTPServer(args.port, job_manager)
server.start()
