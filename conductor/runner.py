"""
Conductor Job Runner
"""
from conductor.chains import create_conductor_search
from conductor.agents import build_internal_agent
from conductor.models import ConductorJobCustomerResponse, ConductorJobCustomerInput
import logging


logger = logging.getLogger(__name__)


class ConductorInternalRunner:
    """
    Runner for internal conductor jobs
    """

    def __init__(
        self,
        *,
        job_name: str,
        job_id: str,
        geography: str,
        titles: list[str],
        industries: list[str],
    ):
        self.data = ConductorJobCustomerResponse(
            input=ConductorJobCustomerInput(
                job_name=job_name,
                job_id=job_id,
                geography=geography,
                titles=titles,
                industries=industries,
            )
        )

    @classmethod
    def run_from_inputs(
        cls,
        *,
        job_name: str,
        job_id: str,
        geography: str,
        titles: list[str],
        industries: list[str],
    ) -> ConductorJobCustomerResponse:
        runner = cls(
            job_name=job_name,
            job_id=job_id,
            geography=geography,
            titles=titles,
            industries=industries,
        )
        runner.run()
        return runner.data

    def run(self) -> None:
        logger.info(
            f"Running job: {self.data.input.job_name} with job_id: {self.data.input.job_id}"
        )
        agent_query = create_conductor_search(
            job_id=self.data.input.job_id,
            geography=self.data.input.geography,
            titles=self.data.input.titles,
            industries=self.data.input.industries,
        )
        logger.info("Running: " + agent_query)
        self.data.agent_query = agent_query
        agent = build_internal_agent()
        agent_response = agent.invoke({"input": agent_query})
        self.data.response = agent_response["output"]
