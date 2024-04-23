from celery import Celery
from conductor.pipelines.vectorestores import ApolloPineconeCreateDestroyUpdatePipeline
import logging


logger = logging.getLogger(__name__)
app = Celery(
    "tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)


@app.task
def scrape_page_task(url) -> None:
    logging.info(f"Scraping {url}")


@app.task
def vectorize_apollo_data(job_id: str) -> None:
    logger.info("Vectorizing Apollo data ...")
    pipeline = ApolloPineconeCreateDestroyUpdatePipeline()
    pipeline.update(job_id)
    logger.info("Successfully vectorized Apollo data ...")
