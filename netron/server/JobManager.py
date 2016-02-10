
class JobManager(object):
    def __init__(self, solver):
        self.solver = solver
        self.workers = {}
        self.errors = {}
        self.jobs_done = 0
        self.jobs_failed = 0
        self.jobs_pending = 0
        self.jobs_total =0

    def get_new_job(self, worker_id):
        job = self.solver.get_new_job(worker_id)
        self.jobs_total += 1
        self.jobs_pending += 1
        return job

    def save_results(self, worker_id, data):
        # TODO: save results in the database
        self.jobs_done += 1
        self.jobs_pending -= 1

        return 200


