import dspy
from pydantic import BaseModel
import concurrent.futures

claude = dspy.LM("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
dspy.configure(lm=claude)


class ResearchAgentTemplate(BaseModel):
    title: str
    research_questions: list[str]


class AgentBuilderFromReportSection:
    def __init__(self, section_title: str) -> None:
        self.section_title = section_title

    def build(self) -> ResearchAgentTemplate:
        agent_title = dspy.ChainOfThought(
            "company_report_section_title:str -> agent_title:str"
        )
        generated_title = agent_title(
            company_report_section_title=self.section_title
        ).agent_title
        research_questions = dspy.ChainOfThought(
            "agent_title:str -> company_research_questions:list[str]"
        )
        return ResearchAgentTemplate(
            title=generated_title,
            research_questions=research_questions(
                agent_title=generated_title
            ).company_research_questions,
        )


def build_from_section(section_title: str) -> ResearchAgentTemplate:
    return AgentBuilderFromReportSection(section_title).build()


def build_from_report_sections(
    section_titles: list[str],
) -> list[ResearchAgentTemplate]:
    return [build_from_section(title) for title in section_titles]


def build_from_report_sections_parallel(
    section_titles: list[str],
) -> list[ResearchAgentTemplate]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(build_from_section, section_titles))
