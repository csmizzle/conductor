"""
Runners for report generation pipeline
"""
import concurrent.futures
from conductor.flow.retriever import ElasticRMClient
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.reports.builder.conversations import (
    SimulatedConversation,
    SummarizeConversation,
)
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

    def run_team_conversations(self) -> list[ResearchAgentConversations]:
        """
        Run simulated conversations for each research agent in parallel
        """
        agent_conversations = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for agent in self.team.agent_templates:
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


def run_team_simulated_conversations(
    team: ResearchTeamTemplate,
    retriever: ElasticRMClient,
    max_conversation_turns: int = 5,
) -> list[ResearchAgentConversations]:
    """
    Run simulated conversations for research team
    """
    runner = ResearchTeamSimulatedConversationRunner(
        team=team, retriever=retriever, max_conversation_turns=max_conversation_turns
    )
    return runner.run_team_conversations()


def refine_team_from_conversations(
    *,
    initial_team: ResearchTeamTemplate,
    conversations: list[ResearchAgentConversations],
) -> ResearchTeamTemplate:
    """
    Map refined questions from conversations to team
    """
    for research_conversation in conversations:
        for agent in initial_team.agent_templates:
            if agent.title == research_conversation.agent.title:
                new_questions = []
                for conversation in research_conversation.conversations:
                    new_questions.append(conversation.question)
                agent.research_questions = new_questions
    return initial_team


def summarize_conversation(
    conversation: Conversation,
) -> str:
    """
    Summarize conversations
    """
    conversation_summarizer = SummarizeConversation()
    return conversation_summarizer(conversation).summary


def summarize_conversations_parallel(
    conversations: list[Conversation],
) -> list[str]:
    """
    Summarize conversations in parallel
    """
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for conversation in conversations:
            futures.append(executor.submit(summarize_conversation, conversation))
        for future in concurrent.futures.as_completed(futures):
            summaries.append(future.result())
    return summaries


def summarize_team_conversations_parallel(
    team_conversations: list[ResearchAgentConversations],
) -> list[str]:
    """
    Summarize team conversations in parallel
    """
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for research_agent_conversations in team_conversations:
            futures.append(
                executor.submit(
                    summarize_conversations_parallel,
                    research_agent_conversations.conversations,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            summaries.extend(future.result())
    return summaries
