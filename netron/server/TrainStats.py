import pymongo
import numpy as np
import json

class TrainStats:
    def __init__(self, mongo_server = "localhost", mongo_port = 27017):
       self.mongo_client = pymongo.MongoClient(mongo_server, mongo_port)
       self.db = self.mongo_client["netron"]
       self.experiments_col = self.db["experiments"]

    def get_all_experiments(self):
        cur = self.experiments_col.find({}).sort('created_at', pymongo.DESCENDING)
        exp_ids = set()
        experiments = []
        # stupid way to get distinct ids.
        for exp in cur:
            if exp["experiment_id"] not in exp_ids:
                exp_ids.add(exp["experiment_id"])
                experiments.append(exp)

        return {"experiments": experiments}

    def get_stats(self, experiment_id):
        stats = {"experiment_id": experiment_id}

        cur = self.experiments_col.find({"experiment_id": experiment_id})
        models = [row for row in cur]

        # If there was a validation set, sort by val_loss, otherwise by loss
        if models[0]["val_loss"]:
            models.sort(key = lambda x: x["val_loss"] if not np.isnan(x["val_loss"]) else 100)
        else:
            # model['loss'] is an array and some values can be NaN. Handling them here
            # in a trivial way. Must be cleaned up.
            models.sort(key = lambda x:min([100] + [loss for loss in x["loss"] if not np.isnan(loss)]))

        top_models = models[:10]
        for model in top_models:
            losses = [100] + [loss for loss in model["loss"] if not np.isnan(loss)]
            model["mean_loss"] = np.mean(losses)
            model["min_loss"] = min(losses)
            model["model_params"] = json.dumps(model["model_params"])

        stats["count"] = len(models)
        stats["models"] = top_models

        return stats

