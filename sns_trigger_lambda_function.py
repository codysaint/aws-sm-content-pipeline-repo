import json
import boto3
import datetime
import logging
import time
import os

sns = boto3.client('sns')
time_created = datetime.datetime.now()

# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    
    try:
        topic_arn = os.environ['SNS_TOPIC_ARN']
        logger.info(f"\n SNS Trigger Lambda created at: {time_created} for topic << {topic_arn} >>")

        # Parse the alarm event
        sns_event = event
        logger.info("\n\n SNS Event: \n %s", sns_event)      
       
        
#         sns_message = json.loads(sns_event['Message'])
        sns_message=event['detail']
#         logger.info("\n SNS Message: \n %s", sns_message)  
        
        # Extract the relevant details from the alarm event
        alarm_name = sns_message['alarmName']
        alarm_description = sns_message['configuration']['description']
        old_state = sns_message['previousState']['value']
        new_state=sns_message['state']['value']
        reason = sns_message['state']['reason']
        
        # sns subject
        subject = 'CloudWatch Alarm: {}'.format(alarm_name)
        
        # Customize the notification message
#         message = event['message']
        message = 'Alarm Name: {}\n\n'.format(alarm_name)
        message += 'Alarm Description: {}\n\n'.format(alarm_description)
        message += 'Old State: {}\n\n'.format(old_state)
        message += 'New State: {}\n\n'.format(new_state)
        message += 'Reason: {}\n\n'.format(reason)
        
        response = sns.publish(
                        TopicArn=topic_arn,
                        Subject=subject,
                        Message=message
                    )
        
        return response
            
    except Exception as e:
        logger.exception('Exception occurred while execution of lambda trigger for sns: %s', e)