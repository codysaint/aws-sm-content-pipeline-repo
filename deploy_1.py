import os
import subprocess

def upgrade_pip():
    subprocess.call(f'pip install --upgrade pip', shell=True)
    
def install_library(library):
    subprocess.call(f'pip install {library}', shell=True)

def deploy_endpoint(args):
    import boto3
    import sagemaker
    import time
#     from sagemaker import Model, EndpointConfig, Endpoint
    
    from time import gmtime, strftime
    
#     train_image_uri = args.train_image_uri
    model_data = args.model_data
    mdl_name = args.model_name
    sm_role = args.sm_role
    region = args.region
    endpoint_name = args.endpoint_name
    
#     model_version_arn = args.model_version_arn
    inference_image_uri = args.inference_image_uri
    
    print('region: ', region)
    boto_session = boto3.Session(region_name=region)
    sagemaker_boto_client = boto_session.client("sagemaker")
    
#     model=None
    create_model_response=None
    create_endpoint_config_response=None
    create_endpoint_response=None
    endpoint_status=None
    
    arn='arn:aws:iam::791574662255:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole'
    
    model_name = f'{mdl_name}-' + strftime("%Y-%m-%d-%H-%M", gmtime())
    print("Model name : {}".format(model_name))
    
    container_list = [{'Image': inference_image_uri,
                    'ModelDataUrl': model_data
#                        'ModelPackageName': model_version_arn
                      }]

    create_model_response = sagemaker_boto_client.create_model(
        ModelName = model_name[:63],
        ExecutionRoleArn = sm_role,
#         Containers = container_list
        PrimaryContainer={
        "Image": inference_image_uri,
        "ModelDataUrl": model_data,
    },
    )
    
    print("Model arn : {}".format(create_model_response["ModelArn"]))

    # name truncated per sagameker length requirements (63 char max)
    endpoint_config_name = f"{model_name[:45]}-ep-config"
    existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name)[
        "EndpointConfigs"
    ]

    if not existing_configs:
        # Create an endpoint configuration
#         endpoint_config_name = 'DEMO-modelregistry-EndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        print(endpoint_config_name)
        create_endpoint_config_response = sagemaker_boto_client.create_endpoint_config(
            EndpointConfigName = endpoint_config_name,
            ProductionVariants=[{
                'InstanceType':'ml.m5.xlarge',
                'InitialVariantWeight':1,
                'InitialInstanceCount':1,
                'ModelName':model_name[:63],
                'VariantName':'AllTraffic'}])
        

    existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=endpoint_name)[
        "Endpoints"
    ]

    if not existing_endpoints:
#         Create an endpoint
#         endpoint_name = 'DEMO-modelregistry-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        print("EndpointName={}".format(endpoint_name))

            create_endpoint_response = sagemaker_boto_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name)
        
        print("create_endpoint_response: ", create_endpoint_response)
    else:
        create_endpoint_response = sagemaker_boto_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
        
        print("create_endpoint_response: ", create_endpoint_response)


    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = endpoint_info["EndpointStatus"]
    
    max_time = time.time() + 15 * 60  # 15 min

    while time.time() < max_time:
        
        try:
            endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = endpoint_info["EndpointStatus"]

    #         print("endpoint_info: ", endpoint_info)
    #         print("endpoint_status: ", endpoint_status)

            if endpoint_status == 'Creating':
                print(f'Endpoint {endpoint_name} is being created')

                time.sleep(5)
            elif endpoint_status == 'Updating':
                print(f'Endpoint {endpoint_name} is being updated')

                time.sleep(5)
            elif endpoint_status == 'Deleting':
                print(f'Endpoint {endpoint_name} is being deleted')
            elif endpoint_status == 'InService':
                print("\n ========== Endpoint Created Successfully ========== \n")
                print(f'Endpoint {endpoint_name} is in service and ready to serve inferences')
                break
            elif endpoint_status == 'OutOfService':
                print(f'Endpoint {endpoint_name} is running but not ready to serve inferences')
            elif endpoint_status == 'Failed':
                print(f'Endpoint {endpoint_name} has failed and cannot be used')
                break
            else:
                print(f'Endpoint {endpoint_name} status is {endpoint_status}. Please check manually.')
        except Exception as e:
            print(f'Exception occurred while creating endpoint - {e}')


if __name__ == "__main__":
    
    upgrade_pip()
    
    # Install a library
    install_library('sagemaker')
    install_library('boto3')
    install_library('argparse')
    
    import argparse
    from argparse import ArgumentParser

    # Parse argument variables passed via the DeployModel processing step
    parser = ArgumentParser()
#     parser.add_argument("--train_image_uri", type=str)
    parser.add_argument("--model_data", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--sm_role", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--endpoint-name", type=str)
#     parser.add_argument("--model_version_arn", type=str)
    parser.add_argument("--inference_image_uri", type=str)
    
    args = parser.parse_args()

    deploy_endpoint(args)