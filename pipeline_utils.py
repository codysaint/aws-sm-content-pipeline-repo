#Import libraries
import boto3
import json
import botocore
import time
import os
from random import randint
from functools import wraps
import logging
import sagemaker
import pandas as pd


#Initialise AWS Clients
lambda_client = boto3.client('lambda')
iam_client = boto3.client('iam')
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

iam_desc = 'IAM Policy for Lambda triggering AWS SageMaker Pipeline'
fcn_desc = 'AWS Lambda function for automatically triggering AWS SageMaker Pipeline'

        
# Define IAM Trust Policy for Lambda's role
iam_trust_policy = {
'Version': '2012-10-17',
'Statement': [
  {
    'Effect': 'Allow',
    'Principal': {
      'Service': 'lambda.amazonaws.com'
    },
    'Action': 'sts:AssumeRole'
  }
]
}


#Define function to allow Amazon S3 to trigger AWS Lambda
def allow_s3(fcn_name,bucket_arn,account_num):
    print('Adding permissions to Amazon S3 ...')
    response = lambda_client.add_permission(
            FunctionName=fcn_name,
            StatementId=f"S3-Trigger-Lambda-{int(time.time())}",
            Action='lambda:InvokeFunction',
            Principal= 's3.amazonaws.com',
            SourceArn=bucket_arn,
            SourceAccount=account_num
        )
    print('SUCCESS: Successfully added permissions to Amazon S3!')

        

def add_permissions(name):
    print("Adding permissions to AWS Lambda function's IAM role ...")
    add_execution_role = iam_client.attach_role_policy(
            RoleName=name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
    
    add_execution_role = iam_client.attach_role_policy(
                RoleName=name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
    add_execution_role = iam_client.attach_role_policy(
            RoleName=name,
            PolicyArn='arn:aws:iam::aws:policy/AmazonEventBridgeFullAccess'
        )
    
    print("SUCCESS: Successfully added permissions AWS Lambda function's IAM role!")


def create_role(role_name):
    print('Creating an IAM role for AWS Lambda function ...')
    create_iam_role = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(iam_trust_policy),
        Description=iam_desc
        )
    print('SUCCESS: Successfully created IAM role for AWS Lambda function!')
    time.sleep(10)
    add_permissions(role_name)
    return {
            'arn': create_iam_role['Role']['Arn'],
            'name': create_iam_role['Role']['RoleName']
        }  


def create_lambda(module_name, fcn_name, fcn_code, role_arn):
    print('Creating AWS Lambda function ...')
    new_fcn = lambda_client.create_function(
            FunctionName=fcn_name,
            Runtime='python3.8',
            Role=role_arn,
            Handler=f'{module_name}.lambda_handler',
            Code=dict(ZipFile=fcn_code),
            Description=fcn_desc,
            Timeout=10,
            MemorySize=128,
            Publish=True
        )
    print('SUCCESS: Successfully created AWS Lambda function!')
    return new_fcn['FunctionArn']


def add_notif(bucket, prefix, lambda_fcn_arn):
    print('Initialising Amazon S3 Bucket client ...')
    bucket_notification = s3_resource.BucketNotification(bucket)
    print('SUCCESS: Successfully initilised Amazon S3 Bucket client!')
    print('Setting up notifications on Amazon S3 Bucket')
    setup_notif = bucket_notification.put(
            NotificationConfiguration={
                'LambdaFunctionConfigurations': [
                    {
                        'LambdaFunctionArn': lambda_fcn_arn,
                        'Events': ['s3:ObjectCreated:Put','s3:ObjectCreated:CompleteMultipartUpload'],
                        'Filter': {
                            'Key': {
                                'FilterRules': [
                                    {
                                        'Name': 'suffix',
                                        'Value': '.csv'
                                    },
                                    {
                                        'Name': 'prefix',
                                        'Value': prefix
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        )
    print('SUCCESS: Successfully added notifications to Amazon S3 Bucket!')
    
def create_s3_trigger(fcn_name,bucket,prefix, account_num, lambda_fcn_arn):
    bucket_arn = f"arn:aws:s3:::{bucket}"
    allow_s3(fcn_name,bucket_arn,account_num)
    add_notif(bucket, prefix, lambda_fcn_arn)