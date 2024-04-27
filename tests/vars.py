import os
import json


BASEDIR = os.path.dirname(os.path.abspath(__file__))
TEST_PINECONE_INDEX = os.getenv("TEST_PINECONE_INDEX", "unit-test-index")
TEST_APOLLO_JOB_ID = "test_id_1234"
TEST_RAW_DATA_BUCKET = os.getenv("TEST_RAW_DATA_BUCKET", "apollo-testing-bucket")
TEST_ENGAGEMENT_STRATEGIES_BUCKET = os.getenv(
    "TEST_ENGAGEMENT_STRATEGIES_BUCKET", "apollo-testing-bucket"
)
TEST_CREW_PROMPT = "Find CTOs in McLean, VA. Tell me about their backgrounds."
TEST_APOLLO_DATA = json.load(
    open(os.path.join(BASEDIR, "data", "apollo_person_search.json"))
)
TEST_APOLLO_RAW_DATA = json.load(open(os.path.join(BASEDIR, "data", "raw.json")))
TEST_CASSETTES_DIR = os.path.join(BASEDIR, "cassettes")
TEST_GMAIL_CREW_PROMPT = "Create a short email to send to chris.smith@syrinxlabs.com 'Test subject' as the subject and 'Test message' and the message."
TEST_GMAIL_INPUT = {
    "to": ["chris.smith@syrinxlabs.com"],
    "subject": "Test subject",
    "message": "Test message",
    "credentials": os.path.join(BASEDIR, "..", "credentials.json"),
}
TEST_DISCORD_JOB_ID = ""
TEST_DISCORD_PINECONE_INDEX = ""
TEST_RAW_DISCORD_DATA_BUCKET = ""
