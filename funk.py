from inference_sdk import InferenceHTTPClient

def roboflow_detect(image, id):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="GeTm2IXviDUjvAQRqDdw"
    )

    result = CLIENT.infer(image, model_id=id)
    return result