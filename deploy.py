import os
import subprocess

def install_library(library):
    subprocess.call(f'pip install {library}', shell=True)

def deploy_endpoint(args):    
    import boto3
    boto3.setup_default_session(region_name=os.environ["AWS_REGION"])
    boto_session = boto3.Session(region_name=os.environ["AWS_REGION"])
    
    import sagemaker
    from sagemaker.sklearn import SKLearnModel
    
    model_data = args.model_data
    print("model data: ", model_data)
    sm_role = args.sm_role
    print("sm role: ", sm_role)
    inference_prefix=args.inference_prefix
    # region = args.region
    # print("region: ", region)
    endpoint_name = args.endpoint_name
    
    model=None
    entry_script = os.path.join("/opt/ml/processing/input", f"{inference_prefix}.py")
    
    sagemaker_boto_client = boto_session.client("sagemaker")
    
    # Create a model
    model = SKLearnModel(
                        role=sm_role,
                        model_data=model_data,
                        framework_version="1.0-1",
                        py_version="py3",
                        entry_point=entry_script
                    )
    
    existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_name)[
        "EndpointConfigs"
    ]
    
    if existing_configs:
        print("\n ===== Deleting Endpoint Config ====== \n")
        # Delete the endpoint configuration
        response = sagemaker_boto_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    else:
        print("\n ==== No such config ==== \n")
        pass
    
    existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=endpoint_name)[
        "Endpoints"
    ]
    
    if existing_endpoints:
        print("\n\t ===== Deleting Endpoint ===== \n\t")
        # Delete the endpoint
        response = sagemaker_boto_client.delete_endpoint(EndpointName=endpoint_name)

        # Wait for the endpoint to be deleted
        waiter = sagemaker_boto_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        
        print("\n\t ===== Endpoint deleted successfully ===== \n\t")
        
        print("\n\t ===== Creating new endpoint ===== \n\t")
        
        # deploy model
        deploy_model = model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge', endpoint_name=endpoint_name)
    else:
        print("\n\t ===== NO such Endpoint /// Creating new endpoint ===== \n\t")

        # deploy model
        deploy_model = model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge', endpoint_name=endpoint_name)

    
    print("\n ========== Endpoint Created Successfully ========== \n")



if __name__ == "__main__":
    # Install a library
    install_library('sagemaker')
    install_library('boto3')
    install_library('argparse')
    
    import argparse
    from argparse import ArgumentParser

    # Parse argument variables passed via the DeployModel processing step
    parser = ArgumentParser()
    
    parser.add_argument("--model_data", type=str)
    parser.add_argument("--inference_prefix", type=str)
    parser.add_argument("--sm_role", type=str)
    parser.add_argument("--endpoint_name", type=str)
    
    args = parser.parse_args()
    
    deploy_endpoint(args)
    
    