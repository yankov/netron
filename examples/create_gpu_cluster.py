from netron.cloud import AWSCluster
import time

# This script creates and then destroys a 2-node GPU cluster in AWS.

cluster = AWSCluster()

# Create 2 GPU instances at max price of $0.1/hr.
cluster.create_spot_instances(max_spot_price = 0.1, instance_count = 1, bootstrap_script = "examples/bootstrap_worker.sh")
req = cluster.describe_spot_requests()

while "open" in req:
    time.sleep(1)
    print "Waiting to fullfill %d spot requests." % req["open"]
    req = cluster.describe_spot_requests()

print "All spot requests have been fulfilled!\nInstance running: %d" % cluster.live_instances_count()
print "Terminating ... "

# Clearing spot requests, just in case.
cluster.cancel_all_spot_requests()

# Terminate instances
#cluster.terminate_all_instances()

# If script fails before this line, make sure there are no un-terminated instances left.
print "Ok!"





