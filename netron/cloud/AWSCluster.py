# -*- coding: utf-8 -*-

import os
from boto3.session import Session
import base64

class AWSCluster():
    AWS_CREDENTIALS_PATH = "~/.aws/credentials"
    DEFAULT_REGION = "us-east-1"
    AMI_NAME = "neuro_worker"
    KEY_NAME = "neuro_keys"

    def __init__(self, aws_credentials_path = AWS_CREDENTIALS_PATH,
                 default_region = DEFAULT_REGION,
                 ami_name = AMI_NAME,
                 key_name = KEY_NAME):

        self.default_region = default_region
        self.aws_credentials_path = aws_credentials_path
        self.key_name = key_name
        self.secret_key, self.access_key = self.read_credentials(aws_credentials_path)

        self.session = {}
        self.create_session(default_region)

        self.ec2 = { default_region: self.session[default_region].resource('ec2') }
        self.ec2_client = { default_region: self.session[default_region].client('ec2') }
        self.s3 = { default_region: self.session[default_region].resource('s3') }

        self.AMI_NAME = ami_name

    def read_credentials(self, aws_credentials_path):
        """ Just reads the standard AWS credentials file """

        with open(os.path.expanduser(aws_credentials_path)) as f:
            creds = f.readlines()

        return creds[1].split(" = ")[1].rstrip(), creds[2].split(" = ")[1].rstrip()

    def create_session(self, region):
        self.session[region] = Session(aws_access_key_id=self.access_key,
                                       aws_secret_access_key=self.secret_key,
                                       region_name = region)

    def get_ami_id(self, ami_name, region = DEFAULT_REGION):
        amis = self.ec2_client[region].describe_images(Filters=[{'Name':'tag:Name', 'Values':[ami_name]}])
        return amis['Images'][0]['ImageId']

    def live_instances(self, region = DEFAULT_REGION):
        return self.ec2[region].instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])

    def live_instances_count(self, region = DEFAULT_REGION):
        return len(list(self.live_instances(region)))

    def create_spot_instances(self, max_spot_price, instance_count, instance_type = "g2.2xlarge",
                              region = DEFAULT_REGION, bootstrap_script = "examples/bootstrap_worker.sh"):
        with open(bootstrap_script) as f:
            script = base64.b64encode(f.read())

        specs = {
            'ImageId': self.get_ami_id(self.AMI_NAME, region),
            'KeyName': self.key_name,
            'SecurityGroups': ["netron-workers"],
            'UserData': script,
            'IamInstanceProfile': {
                'Name': 'netronWorker'
             },
            'InstanceType': instance_type,
        }

        req = self.ec2_client[region].request_spot_instances(SpotPrice=str(max_spot_price),
                                         InstanceCount = instance_count,
                                         LaunchSpecification = specs)

    def describe_spot_requests(self, region = DEFAULT_REGION):
        reqs = self.ec2_client[region].describe_spot_instance_requests()
        state = {}
        for req in reqs["SpotInstanceRequests"]:
            if req["State"] not in state:
                state[req["State"]] = 0

            state[req["State"]] += 1

        return state

    def cancel_all_spot_requests(self, region = DEFAULT_REGION):
        reqs = self.ec2_client[region].describe_spot_instance_requests()
        req_ids = [req["SpotInstanceRequestId"] for req in reqs["SpotInstanceRequests"]]
        self.ec2_client[region].cancel_spot_instance_requests(SpotInstanceRequestIds=req_ids)

    def terminate_all_instances(self, region = DEFAULT_REGION):
        inst_ids = [inst.id for inst in self.live_instances()]
        self.ec2_client[region].terminate_instances(InstanceIds = inst_ids)
