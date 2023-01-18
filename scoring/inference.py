#inferencing code starts here .....
import os
import json
import numpy as np
from joblib import load
from sagemaker_inference import content_types, decoder
from sagemaker_inference import encoder

# prefix = "/opt/ml/"
# model_dir = os.path.join(prefix, "model")


"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "artwork_content_model.pkl")
    model = load(model_path)
    print("Model Loaded ..\n")
    return model


"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    print('request body: ', request_body)
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
        return request_body
    else:
        raise ValueError("This model only supports application/json input")

        
"""
predict_fn
    input_data: returned array from input_fn above
    model (mch content model) returned model loaded from model_fn above
"""
# return prediction based on loaded model (from the step above) and an input payload
def predict_fn(input_data, model):
    rec = []
    response = {}
    
    print("input data: ", input_data)
    item_id = int(input_data['itemId'])
    
    try:
        rec = [i[0] for i in model[item_id]][:10]
    except Exception as e:
        rec = [type(input_data),str(e)]
        
    response = {'rec': rec}
    
    print('\n response: ', response)
    return response


"""
After invoking predict_fn, the model server invokes output_fn, passing in the return-value from 
predict_fn and the InvokeEndpoint requested response content-type.
"""
def output_fn(prediction, content_type):
    print('\n prediction results \n', prediction)
    return prediction