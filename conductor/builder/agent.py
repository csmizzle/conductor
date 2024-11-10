import dspy
from pydantic import BaseModel
import concurrent.futures
from tqdm import tqdm
from conductor.builder import signatures
from langfuse.decorators import observe

claude = dspy.LM("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
dspy.configure(lm=claude)


class ResearchAgentTemplate(BaseModel):
    title: str
    research_questions: list[str]


class ResearchTeamTemplate(BaseModel):
    title: str
    perspective: str
    agent_templates: list[ResearchAgentTemplate]


class AgentBuilderFromReportSection:
    def __init__(
        self,
        team_title: str,
        section_title: str,
        perspective: str,
    ) -> None:
        self.team_title = team_title
        self.section_title = section_title
        self.perspective = perspective
        self.generate_agent_title = dspy.ChainOfThought(signatures.AgentTitle)
        self.generate_research_questions = dspy.ChainOfThought(
            signatures.ResearchQuestions
        )

    def build(self) -> ResearchAgentTemplate:
        generated_title = self.generate_agent_title(
            research_team_title=self.team_title, report_section_title=self.section_title
        ).agent_title
        generated_research_questions = self.generate_research_questions(
            research_team_title=self.team_title,
            agent_title=generated_title,
            agent_perspective=self.perspective,
        ).research_questions
        return ResearchAgentTemplate(
            title=generated_title,
            perspective=self.perspective,
            research_questions=generated_research_questions,
        )


@observe(as_type="generation")
def build_from_section(
    team_title: str, section_title: str, perspective: str
) -> ResearchAgentTemplate:
    return AgentBuilderFromReportSection(
        team_title=team_title, section_title=section_title, perspective=perspective
    ).build()


@observe(as_type="generation")
def build_from_report_sections(
    team_title: str, section_titles: list[str], perspective: str
) -> ResearchTeamTemplate:
    agents = [
        build_from_section(
            team_title=team_title, section_title=title, perspective=perspective
        )
        for title in tqdm(section_titles)
    ]
    return ResearchTeamTemplate(title=team_title, agents=agents)


@observe(as_type="generation")
def build_from_report_sections_parallel(
    team_title: str,
    section_titles: list[str],
    perspective: str,
) -> ResearchTeamTemplate:
    agents = []
    # build partial function with report title
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for section_title in section_titles:
            futures.append(
                executor.submit(
                    build_from_section,
                    team_title=team_title,
                    section_title=section_title,
                    perspective=perspective,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            agents.append(future.result())
    return ResearchTeamTemplate(
        title=team_title, perspective=perspective, agent_templates=agents
    )
