import os
import boto3
import urllib2
import numpy as np
from pymongo import MongoClient
import datetime
import json

class DataLoader:
    '''Responsible for loading datasets and storing results'''

    def __init__(self, url, mongo_uri = None):
        self.url = url
        self.data_path = os.path.join(os.path.dirname(__file__), "data")
#        self.data_path = "/home/ubuntu/netron/netron/worker/data"
        self.data_files = {}
        self.mongo_uri = mongo_uri
        self.mongo_col = None

    def init_mongo_conn(self):
        self.mongo_client = MongoClient(self.mongo_uri)
        self.db = self.mongo_client['netron']
        self.mongo_col = self.db["experiments"]

    def save_result(self, exp_id, result):
        if not self.mongo_col:
            self.init_mongo_conn()

        """ Saves the results of training to MongoDB """
        self.mongo_col.insert({
            "experiment_id": exp_id,
            "loss": result.history["loss"],
            "val_loss": result.history["val_loss"] if "val_loss" in result.history else None,
            "val_accuracy": result.history["val_accuracy"] if "val_accuracy" in result.history else None,
            "params": result.params,
            "model_params": json.loads(result.model.to_json()),
            "created_at": datetime.datetime.utcnow()})

    def download_file_from_server(self, filename):
        result = urllib2.urlopen(self.url + "/data/" + filename)
        data_file = os.path.join(self.data_path, filename)
        with open(data_file, "w") as f:
            f.write(result.read())
        data = np.load(data_file)
        return data

    def download_file_from_s3(self, filename):
        path = filename.split("//")[1].split("/")
        bucket = path[0]
        filename = "/".join(path[1:])
        data_file = os.path.join(self.data_path, path[-1])
        if os.path.isfile(data_file):
            return np.load(data_file)

        key = boto3.resource('s3').Object(bucket, filename).get()
        with open(data_file, "w") as f:
            for chunk in iter(lambda: key['Body'].read(4096), b''):
                f.write(chunk)

        return np.load(data_file)

    def load_data(self, filename):
        if filename in self.data_files:
            return self.data_files[filename]

        print "Getting data from the server"
        if "s3://" in filename:
            data = self.download_file_from_s3(filename)
        else:
            data = self.download_file_from_server(filename)

        self.data_files[filename] = [data["X_train"], data["y_train"]]

        # Add a test set if it's there
        if "X_test" in data and "y_test" in data:
            self.data_files[filename].extend([data["X_test"], data["y_test"]])
        else:
            self.data_files[filename].extend([None, None])

        return self.data_files[filename]

