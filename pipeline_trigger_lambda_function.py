import json
import boto3
import datetime
import logging
import time

s3 = boto3.resource('s3')
sm = boto3.client('sagemaker')
# Connect to SES (Simple Email Service)
ses = boto3.client('ses')
time_created = datetime.datetime.now()

# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


# check the pipeline execution status
def check_pipeline_status(sm, pipeline_execution_arn):

    # Get the response of latest execution of the pipeline
    pipeline_arn_response = sm.describe_pipeline_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )

    # Extract the status of the latest execution
    status = pipeline_arn_response['PipelineExecutionStatus']

    return status


# send email notification on successful execution of pipeline
def send_email_notification(ses, subject, body, source, recipient):
    # Create email message
    message = {
        'Subject': {
            'Data': subject,
            'Charset': 'UTF-8'
        },
        'Body': {
            'Text': {
                'Data': body,
                'Charset': 'UTF-8'
            }
        }
    }

    try:
        # Send the email
        response = ses.send_email(
            Source=source,
            Destination={
                'ToAddresses': [
                    recipient
                ]
            },
            Message=message
        )

        logger.info('Email sent successfully to %s', recipient)

    except Exception as e:
        logger.exception('Error while sending mail notification: %s', e)
        
    return response


def lambda_handler(event, context):
    
    response = {}
    ses_response=None
    
    try:
        
        pipeline_name = os.environ["PIPELINE_NAME"]

        logger.info(f"Time Lambda created: {time_created} for the pipeline << {pipeline_name} >>")

        #Check version of Boto3 - It must be at least 1.16.55
        logger.info("The version of Boto3 is %s",  boto3.__version__)

        logger.info("Event is %s", event)

        #Get location for where the new data (csv) file was uploaded
        data_bucket = event['Records'][0]['s3']['bucket']['name']
        data_key = event['Records'][0]['s3']['object']['key']

        logger.info("A new file named %s was just uploaded to Amazon S3 in %s", data_key, data_bucket)

        #Start the pipeline execution
        pipeline_exec_response = sm.start_pipeline_execution(
                        PipelineName=pipeline_name,
                        PipelineExecutionDisplayName=f"{data_key.split('/')[-1].replace('_','').replace('.csv','')}",
                        PipelineExecutionDescription=data_key
                        )
        logger.info('Response: %s', pipeline_exec_response)
        
        pipeline_arn = pipeline_exec_response['PipelineExecutionArn']

#         pipeline_status = check_pipeline_status(sm, pipeline_arn)
#         logger.info('Pipeline status: %s', pipeline_status)
        
        response={'statusCode': 200,
                 'msg': pipeline_exec_response}
        
        return response
            
    except Exception as e:
        logger.exception('Exception occurred while lambda execution: %s', e)