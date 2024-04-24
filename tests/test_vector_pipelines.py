"""
Test vector pipelines
"""
from conductor.pipelines.vectorestores import ApolloPineconeCreateDestroyUpdatePipeline
from tests.vars import TEST_JOB_ID, TEST_PINECONE_INDEX, TEST_RAW_DATA_BUCKET


def test_s3_to_pinecone_apollo() -> None:
    data = ApolloPineconeCreateDestroyUpdatePipeline().update(
        job_id=TEST_JOB_ID,
        index_name=TEST_PINECONE_INDEX,
        source_bucket_name=TEST_RAW_DATA_BUCKET,
    )
    assert data["status"] == "success"
