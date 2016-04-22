from hyperopt.mongoexp import MongoJobs, MongoWorker
import pymongo
import sys

class HyperOptWorker:
    def __init__(self, mongo_uri, options):
        self.options = options
        self.mj = MongoJobs.new_from_connection_str(mongo_uri + "netron/hyperopt_jobs")
        last_job = self.mj.db["hyperopt_jobs"].find({}, {"exp_key": 1}).sort("exp_key", pymongo.DESCENDING).limit(1)
        last_job = list(last_job)
        if len(last_job) > 0:
            self.exp_key = last_job[0]["exp_key"]
        else:
            self.exp_key = "0"

    def run(self):
        print "Starting a worker for exp_key = %s" % str(self.exp_key)

        mworker = MongoWorker(self.mj,
                float(self.options["poll_interval"]),
                workdir=self.options["workdir"],
                exp_key=self.exp_key)

        while True:
            res = mworker.run_one(reserve_timeout=float(self.options["reserve_timeout"]))


if __name__ == "__main__":
    worker = HyperOptWorker("mongodb://localhost:27017/", {"poll_interval":1, "workdir":None, "reserve_timeout":360})
    worker.run()
