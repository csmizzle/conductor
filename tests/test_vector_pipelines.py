"""
Test vector pipelines
"""
from conductor.pipelines.vectorestores import (
    ApolloPineconeCreateDestroyUpdatePipeline,
    DiscordPineconeCreateDestroyUpdatePipeline,
    ApifyBulkPineconeCreateDestroyPipeline,
)
from tests.vars import (
    TEST_APOLLO_JOB_ID,
    TEST_PINECONE_INDEX,
    TEST_RAW_DATA_BUCKET,
    TEST_APIFY_BUCKET,
    TEST_DISCORD_JOB_ID,
    TEST_DISCORD_PINECONE_INDEX,
    TEST_RAW_DISCORD_DATA_BUCKET,
)


def test_s3_apollo_to_pinecone() -> None:
    data = ApolloPineconeCreateDestroyUpdatePipeline().update(
        job_id=TEST_APOLLO_JOB_ID,
        index_name=TEST_PINECONE_INDEX,
        source_bucket_name=TEST_RAW_DATA_BUCKET,
    )
    assert data["status"] == "success"


def test_s3_discord_to_pinecone() -> None:
    data = DiscordPineconeCreateDestroyUpdatePipeline().update(
        job_id=TEST_DISCORD_JOB_ID,
        index_name=TEST_DISCORD_PINECONE_INDEX,
        source_bucket_name=TEST_RAW_DISCORD_DATA_BUCKET,
    )
    return data["status"] == "success"


def test_s3_apify_to_pinecone() -> None:
    data = ApifyBulkPineconeCreateDestroyPipeline().build(
        index_name=TEST_PINECONE_INDEX,
        source_bucket_name=TEST_APIFY_BUCKET,
    )
    return data["status"] == "success"
