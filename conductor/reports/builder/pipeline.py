"""
Pipeline for report generation
- Build Outline from report sections
- Conduct conversation for each research question
    - Refine research questions based on conversations
    - Collect data that close research gaps
- Combine outline and conversations
- Use the outline for the report to drive sentence generation
    - Source each sentence from the conversation
"""
from conductor.flow.retriever import ElasticRMClient
from conductor.reports.builder.runner import (
    run_team_simulated_conversations,
    refine_team_from_conversations,
)
from conductor.reports.builder.outline import build_outline
from conductor.reports.builder import models
from conductor.reports.builder import conversations
from conductor.builder.agent import ResearchTeamTemplate


class ConversationReportPipeline:
    """
    Create a STORM report using the conversation pipeline
    """

    def __init__(
        self,
        team: ResearchTeamTemplate,
        specification: str,
        section_titles: list[str],
        retriever: ElasticRMClient,
        max_conversation_turns: int = 5,
    ) -> None:
        self.team = team
        self.retriever = retriever
        self.max_conversation_turns = max_conversation_turns
        self.specification = specification
        self.section_titles = section_titles
        # pipeline values
        self.report_outline: models.ReportOutline = None
        self.simulated_conversation: list[conversations.SimulatedConversation] = None
        self.refined_team: ResearchTeamTemplate = None

    def generate_outline(self) -> None:
        """
        Generate an outline for the report based on the specification
        """
        self.report_outline = build_outline(
            specification=self.specification, section_titles=self.section_titles
        )

    def run_simulated_conversations(self) -> None:
        """
        Run simulated conversations for each research question in parallel
        """
        self.simulated_conversation = run_team_simulated_conversations(
            team=self.team,
            retriever=self.retriever,
            max_conversation_turns=self.max_conversation_turns,
        )

    def refine_conversations(self) -> None:
        """
        Refine the research questions based on the simulated conversations
        """
        self.refined_team = refine_team_from_conversations(
            team=self.team, conversations=self.simulated_conversation
        )
