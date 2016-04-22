from netron.server import JobHTTPServer
from netron.server import JobManager
from netron.solvers import DummySearch, GridSearch, RandomSearch, HyperOptSearch, simple_params_grid
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_shape', required=True, help="Input shape for a first layer (dimensions of the training sample)")
parser.add_argument('--output_dim', type=int, required=True, help="Output dimensions for a last layer (number of predicting classes for example)")
parser.add_argument('--data', required=True, help="Name of the npz file with X_train and y_train")
parser.add_argument('--solver', required=True, help="A solver to search through a parameter space: Currently only RandomSearch or GridSearch")
parser.add_argument('--grid', required=False, help="A json file with a parameter grid. Example: simple_params_grid.json")
parser.add_argument('--port', type=int, default=8080, required=False, help="A port server should be listening on. Default is 8080.")
parser.add_argument('--params_sample_size', type=int, required=False, help="Only for RandomSearch: parameter sample size per network structure.")
parser.add_argument('--structure_sample_size', type=int, required=False, help="Only for RandomSearch: network structure sample size per given number of layers.")
parser.add_argument('--mongo_uri', required=False, default="mongodb://localhost:27017/", help="MongoDB connection string URI.")
parser.add_argument('--layers_num', required=False, help="Number of layers for neural networks.")
parser.add_argument('--max_evals', required=False, default=1e6, help="Max number of samples to train")
parser.add_argument('--nb_epoch', required=False, default=10, type=int, help="Max mumber of epoch per job.")
parser.add_argument('--patience', required=False, default=5, type=int, help="Max mumber of epoch without improvement (EarlyStopper).")

args = parser.parse_args()

input_shape = [int(dim) for dim in args.input_shape.split(",")]

# TODO: cleanup repetative code
print "Starting a server with %s solver and %s dataset" % (args.solver, args.data)
if args.solver == "GridSearch":
    solver = GridSearch(args.grid, input_shape, args.output_dim, "keras", args.data)
    job_manager = JobManager(solver)
    server = JobHTTPServer(args.port, job_manager, args.mongo_uri)
    server.start()
elif args.solver == "RandomSearch":
    if not args.params_sample_size or not args.structure_sample_size:
        raise ValueError("--params_sample_size  and --structure_sample_size must be used with RandomSearch")
    solver = RandomSearch(args.grid, input_shape, args.output_dim, "keras", args.data, args.params_sample_size,
                          args.structure_sample_size)
    job_manager = JobManager(solver)
    server = JobHTTPServer(args.port, job_manager, args.mongo_uri)
    server.start()
elif args.solver == "HyperOpt":
    h = HyperOptSearch(input_shape=input_shape, output_dim=args.output_dim)
    h.start_search_server(args.mongo_uri, args.data, int(args.layers_num), args.max_evals, args.nb_epoch, args.patience)
else:
    raise ValueError("This solver is not supported. Only possible values for --solver right now are GridSearch or RandomSearch")

