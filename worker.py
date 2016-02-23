from netron.worker import Worker
from tornado.ioloop import IOLoop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--server', required=True, help="URL of the netron server. Example: http://localhost:8080.")
parser.add_argument('--mongo_uri', required=False, default="mongodb://localhost:27017/", help="MongoDB connection string URI.")
parser.add_argument('--nb_epoch', required=True, type=int, help="Max mumber of epoch per job.")
parser.add_argument('--patience', required=True, type=int, help="Max mumber of epoch without improvement (EarlyStopper).")
args = parser.parse_args()

worker = Worker(args.server, args.mongo_uri, nb_epoch = args.nb_epoch, patience = args.patience)
worker.get_new_job()
print "Started a worker. Press Ctrl+C to stop."
IOLoop.current().start()


