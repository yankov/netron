import pymongo
import numpy as np

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
        models.sort(key = lambda x:min(x["loss"]))
        top_models = models[:10]

        stats["count"] = len(models)
        stats["models"] = top_models #[min(model["loss"]) for model in top5_models]
        stats["np"] = np

        return stats

