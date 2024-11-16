"""
Models for the pipelines module
"""
from typing import Union
from pydantic import BaseModel
from conductor.builder.agent import ResearchTeamTemplate
from conductor.flow import runner
from conductor.reports.builder import models as report_models
from conductor.graph import models as graph_models
from conductor.profiles.models import Company
from conductor.reports.models import ImageSearchResult
from crewai.crews.crew_output import CrewOutput


class ResearchPipelineState(BaseModel):
    """State for the research pipeline"""

    team: Union[ResearchTeamTemplate, None] = None
    # research_team: Union[flow_models.Team, None] = None
    research_team_output: Union[list[CrewOutput], None] = None
    specification: Union[str, None] = None
    # search_team: Union[flow_models.SearchTeam, None] = None
    search_results: Union[list[runner.SearchTeamAnswers], None] = None
    outline: Union[report_models.ReportOutline, None] = None
    profile: Union[Company, None] = None
    report: Union[report_models.Report, None] = None
    graph: Union[graph_models.Graph, None] = None
    image_search_queries: Union[dict[str, graph_models.Relationship], None] = None
    image_search_results: Union[list[ImageSearchResult], None] = None
