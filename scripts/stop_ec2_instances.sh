#!/bin/bash

# get the aws token for getting instance id
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
INSTANCE_ID=`curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/instance-id`

# stop the instance
AWS_DEFAULT_REGION=ap-southeast-2 aws ec2 stop-instances --instance-ids $INSTANCE_ID
