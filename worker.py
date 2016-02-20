from netron.worker import Worker
from tornado.ioloop import IOLoop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--server', required=True, help="URL of the netron server. Example: http://localhost:8080.")
parser.add_argument('--mongo_server', required=False, default="localhost", help="Hostname where you MongoDB is.")
parser.add_argument('--mongo_port', required=False, type=int, default=27017, help="Port your MongoDB is listening on.")
parser.add_argument('--nb_epoch', required=True, type=int, help="Max mumber of epoch per job.")
parser.add_argument('--patience', required=True, type=int, help="Max mumber of epoch without improvement (EarlyStopper).")
args = parser.parse_args()

worker = Worker(args.server, args.mongo_server , args.mongo_port, nb_epoch = args.nb_epoch, patience = args.patience)
worker.get_new_job()
print "Started a worker. Press Ctrl+C to stop."
IOLoop.current().start()


