"""
Place holders for AWS data operation
"""
import boto3
import json


def upload_dict_to_s3(data: dict, bucket, key):
    """
    Uploads a dictionary to S3
    """
    s3 = boto3.client("s3")
    json_data = json.dumps(data, indent=4)
    s3.put_object(Bucket=bucket, Key=key, Body=json_data)


def get_object(bucket: str, key: str) -> dict:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()
    return json.loads(data)
