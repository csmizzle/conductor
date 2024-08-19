import os
import json
import pathlib

BASEDIR = os.path.dirname(os.path.abspath(__file__))
#### PINECONE
TEST_PINECONE_INDEX = os.getenv("TEST_PINECONE_INDEX", "unit-test-index")
#### APOLLO
TEST_APOLLO_JOB_ID = "test_id_1234"
TEST_RAW_DATA_BUCKET = os.getenv("TEST_RAW_DATA_BUCKET", "apollo-testing-bucket")
TEST_ENGAGEMENT_STRATEGIES_BUCKET = os.getenv(
    "TEST_ENGAGEMENT_STRATEGIES_BUCKET", "apollo-testing-bucket"
)
TEST_APOLLO_DATA = json.load(
    open(os.path.join(BASEDIR, "data", "apollo_person_search.json"))
)
TEST_APOLLO_RAW_DATA = json.load(open(os.path.join(BASEDIR, "data", "raw.json")))
TEST_CASSETTES_DIR = os.path.join(BASEDIR, "cassettes")
#### GMAIL
TEST_GMAIL_CREW_PROMPT = "Create a short email to send to chris.smith@syrinxlabs.com 'Test subject' as the subject and 'Test message' and the message."
TEST_GMAIL_INPUT = {
    "to": ["chris.smith@syrinxlabs.com"],
    "subject": "Test subject",
    "message": "Test message",
    "credentials": os.path.join(BASEDIR, "..", "credentials.json"),
}
#### DISCORD
TEST_DISCORD_JOB_ID = "19fee3dd-539d-40c4-a312-31c9871513b3-1233762364923445309"
TEST_DISCORD_PINECONE_INDEX = os.getenv("TEST_PINECONE_INDEX", "unit-test-index")
TEST_RAW_DISCORD_DATA_BUCKET = os.getenv(
    "TEST_RAW_DISCORD_DATA_BUCKET", "discord-testing-bucket"
)
#### HTML
TEST_HTML_DATA = pathlib.Path(
    os.path.join(BASEDIR, "data", "pinecone.html")
).read_text()
TEST_NEWLINE_HTML_DATA = pathlib.Path(
    os.path.join(BASEDIR, "data", "newline.txt")
).read_text()
#### APIFY
TEST_APIFY_BUCKET = os.getenv("TEST_APIFY_BUCKET", "test-apify-bucket-dev")
#### CREW
TEST_CREW_PROMPT = "Find CTOs in McLean, VA. Tell me about their backgrounds."
#### EMAIL
TEST_EMAIL_CONTEXT = pathlib.Path(
    os.path.join(BASEDIR, "data", "email_context.txt")
).read_text()
#### JSON
TEST_JSON = json.load(open(os.path.join(BASEDIR, "data", "json_data.json")))
#### Text data
TEST_REPORT_RESPONSE = open(os.path.join(BASEDIR, "data", "output.txt")).read()
TEST_COMPLEX_NARRATIVE = open(
    os.path.join(BASEDIR, "data", "complex_narrative.txt")
).read()
TEST_KEY_QUESTIONS_BULLETED = open(
    os.path.join(BASEDIR, "data", "key_questions_bulleted.txt")
).read()
TEST_KEY_QUESTIONS_NARRATIVE = open(
    os.path.join(BASEDIR, "data", "key_questions_narrative.txt")
).read()
### REPORT JSON
REPORT_JSON = json.load(open(os.path.join(BASEDIR, "data", "json_report.json")))
REPORT_V2_JSON = json.load(open(os.path.join(BASEDIR, "data", "test_report_v2.json")))
### CREW RUN
CREW_RUN = json.load(open(os.path.join(BASEDIR, "data", "test_crew_run.json")))
### Graph
GRAPH_JSON = json.load(
    open(os.path.join(BASEDIR, "data", "test_extract_graph_from_report.json"))
)
