import dspy
from pydantic import BaseModel
import concurrent.futures
from functools import partial
from tqdm import tqdm

claude = dspy.LM("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
dspy.configure(lm=claude)


class ResearchAgentTemplate(BaseModel):
    title: str
    research_questions: list[str]


class ResearchTeamTemplate(BaseModel):
    title: str
    agent_templates: list[ResearchAgentTemplate]


class AgentBuilderFromReportSection:
    def __init__(self, team_title: str, section_title: str) -> None:
        self.team_title = team_title
        self.section_title = section_title

    def build(self) -> ResearchAgentTemplate:
        agent_title = dspy.ChainOfThought(
            "research_team_title:str, report_section_title:str -> agent_title:str"
        )
        generated_title = agent_title(
            research_team_title=self.team_title, report_section_title=self.section_title
        ).agent_title
        generated_research_questions = dspy.ChainOfThought(
            "research_team_title:str, agent_title:str -> research_questions:list[str]"
        )
        return ResearchAgentTemplate(
            title=generated_title,
            research_questions=generated_research_questions(
                research_team_title=self.team_title, agent_title=generated_title
            ).research_questions,
        )


def build_from_section(team_title: str, section_title: str) -> ResearchAgentTemplate:
    return AgentBuilderFromReportSection(
        team_title=team_title, section_title=section_title
    ).build()


def build_from_report_sections(
    report_title: str,
    section_titles: list[str],
) -> ResearchTeamTemplate:
    agents = [build_from_section(report_title, title) for title in tqdm(section_titles)]
    return ResearchTeamTemplate(title=report_title, agents=agents)


def build_from_report_sections_parallel(
    report_title: str,
    section_titles: list[str],
) -> ResearchTeamTemplate:
    # build partial function with report title
    build_from_section_partial = partial(build_from_section, report_title)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        agents = list(executor.map(build_from_section_partial, section_titles))
        return ResearchTeamTemplate(title=report_title, agents=agents)
