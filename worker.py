from netron.worker import Worker
from netron.worker import HyperOptWorker
from tornado.ioloop import IOLoop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--server', required=False, help="URL of the netron server. Example: http://localhost:8080.")
parser.add_argument('--mongo_uri', required=False, default="mongodb://localhost:27017/", help="MongoDB connection string URI.")
parser.add_argument('--nb_epoch', required=False, default=10, type=int, help="Max mumber of epoch per job.")
parser.add_argument('--patience', required=False, default=5, type=int, help="Max mumber of epoch without improvement (EarlyStopper).")
parser.add_argument('--type', required=False, help="Type of worker. Only used for HyperOpt right now.")
args = parser.parse_args()

if args.type == "HyperOpt":
    print "Starting HyperOpt worker. Press Ctrl+C to stop."
    worker = HyperOptWorker(args.mongo_uri, {"poll_interval":1, "workdir":"tmp", "reserve_timeout":120})
    worker.run()
else:
    print "Starting a worker. Press Ctrl+C to stop."
    worker = Worker(args.server, args.mongo_uri, exp_id = None, nb_epoch = args.nb_epoch, patience = args.patience)
    worker.get_new_job()
    IOLoop.current().start()


