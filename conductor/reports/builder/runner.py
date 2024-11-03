"""
Runners for report generation pipeline
"""
import concurrent.futures
from conductor.flow.retriever import ElasticRMClient
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.reports.builder.conversations import SimulatedConversation
from conductor.reports.builder.models import ResearchAgentConversations, Conversation


class ResearchAgentSimulatedConversationRunner:
    def __init__(
        self,
        agent: ResearchAgentTemplate,
        retriever: ElasticRMClient,
        max_conversation_turns: int = 5,
    ) -> None:
        self.agent = agent
        self.retriever = retriever
        self.max_conversation_turns = max_conversation_turns
        self.simulated_conversation = SimulatedConversation(
            max_conversation_turns=self.max_conversation_turns, retriever=self.retriever
        )

    def run_research_questions(self) -> ResearchAgentConversations:
        """
        Run simulated conversation for each research question in parallel
        """
        conversations: list[Conversation] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for question in self.agent.research_questions:
                futures.append(executor.submit(self.simulated_conversation, question))
            for future in concurrent.futures.as_completed(futures):
                conversations.append(future.result())
        return ResearchAgentConversations(
            agent=self.agent,
            conversations=[
                Conversation(
                    topic=conversation.topic,
                    conversation_history=conversation.conversation_history,
                    question=conversation.refined_question,
                )
                for conversation in conversations
            ],
        )


class ResearchTeamSimulatedConversationRunner:
    """Run simulated conversations for research team"""

    def __init__(
        self,
        team: ResearchTeamTemplate,
        retriever: ElasticRMClient,
        max_conversation_turns: int = 5,
    ) -> None:
        self.team = team
        self.retriever = retriever
        self.max_conversation_turns = max_conversation_turns
        self.simulated_conversation = SimulatedConversation(
            max_conversation_turns=self.max_conversation_turns, retriever=self.retriever
        )

    def run_research_agents(self) -> list[ResearchAgentConversations]:
        """
        Run simulated conversations for each research agent in parallel
        """
        agent_conversations = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for agent in self.team.agents:
                runner = ResearchAgentSimulatedConversationRunner(
                    agent=agent,
                    retriever=self.retriever,
                    max_conversation_turns=self.max_conversation_turns,
                )
                futures.append(executor.submit(runner.run_research_questions))
            for future in concurrent.futures.as_completed(futures):
                agent_conversations.append(future.result())
        return agent_conversations


def run_agent_simulated_conversations(
    agent: ResearchAgentTemplate,
    retriever: ElasticRMClient,
    max_conversation_turns: int = 5,
) -> ResearchAgentConversations:
    """
    Run simulated conversations for research agent
    """
    runner = ResearchAgentSimulatedConversationRunner(
        agent=agent, retriever=retriever, max_conversation_turns=max_conversation_turns
    )
    return runner.run_research_questions()
