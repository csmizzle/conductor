"""
Small MVP CLI Application that does the following
- Takes in a job_id, geography, titles, and industries
- Runs a job with the given inputs
- Logs the job_id, job_name, and agent_query in AWS S3
- Returns the response to the user
"""
import fire
from conductor.runner import ConductorInternalRunner
from conductor.database.aws import upload_dict_to_s3
import uuid
import logging
import os

logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)


def run_conductor(
    job_name: str = None,
    geography: str = None,
    titles: str = None,
    industries: str = None,
):
    if not job_name:
        job_name = "test-job"
    if not geography:
        geography = "Alexandria, VA"
    if titles:
        titles = list(titles)
    else:
        titles = ["Software Engineer", "CTO"]
    if industries:
        industries = list(industries)
    if not industries:
        industries = ["Tech", "Finance"]
    job_id = str(uuid.uuid4())
    data = ConductorInternalRunner.run_from_inputs(
        job_name=job_name,
        job_id=job_id,
        geography=geography,
        titles=titles,
        industries=industries,
    )
    upload_dict_to_s3(
        data=data.model_dump_json(indent=4),
        bucket=os.getenv("CONDUCTOR_S3_BUCKET"),
        key=f"{job_id}/job.json",
    )


if __name__ == "__main__":
    fire.Fire(run_conductor)
