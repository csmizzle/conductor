"""
Place holders for AWS data operation
"""
import boto3


def upload_dict_to_s3(data, bucket, key):
    """
    Uploads a dictionary to S3
    """
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=str(data))
