import os
import subprocess

def install_library(library):
    subprocess.call(f'pip install {library}', shell=True)

def deploy_endpoint(args):    
    import boto3
    boto3.setup_default_session(region_name=os.environ["AWS_REGION"])
    
    import sagemaker
    from sagemaker.sklearn import SKLearnModel
    
#     train_image_uri = args.train_image_uri
    model_data = args.model_data
    print("model data: ", model_data)
    sm_role = args.sm_role
    print("sm role: ", sm_role)
#     region = args.region
#     print("region: ", region)
    endpoint_name = args.endpoint_name
#     entry_script=args.inference_script_uri
#     bucket = args.bucket
    
    model=None
#     arn=' arn:aws:iam::791574662255:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole'
    entry_script = os.path.join("/opt/ml/processing/input", "inference.py")
    
    print("script path exists: ", os.path.exists(entry_script))
    
    print("region: ", os.environ["AWS_REGION"])
    
#     sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=os.environ["AWS_REGION"]))
#     sagemaker_boto_client = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])
    
#     sm_role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
    
        
    # Create a model
    model = SKLearnModel(
                    role=sm_role,
                    model_data=model_data,
                    framework_version="1.0-1",
                    py_version="py3",
                    entry_point=entry_script
                )
#     deploy model
    deploy_model = model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge', endpoint_name=endpoint_name)

    
    print("\n ========== Endpoint Created Successfully ========== \n")



if __name__ == "__main__":
    # Install a library
    install_library('sagemaker')
    install_library('sagemaker-inference')
    install_library('boto3')
    install_library('argparse')
    
    import argparse
    from argparse import ArgumentParser

    # Parse argument variables passed via the DeployModel processing step
    parser = ArgumentParser()
#     parser.add_argument("--train_image_uri", type=str)
    parser.add_argument("--model_data", type=str)
#     parser.add_argument("--model_name", type=str)
    parser.add_argument("--sm_role", type=str)
#     parser.add_argument("--region", type=str)
    parser.add_argument("--endpoint_name", type=str)
#     parser.add_argument("--inference_script_uri", type=str)
#     parser.add_argument("--bucket", type=str)
    
    args = parser.parse_args()
    
    deploy_endpoint(args)
    
    