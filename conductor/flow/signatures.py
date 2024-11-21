"""
Agent signatures for research team
The signatures need to be defined to control the searching and aggregation
of data collection efforts using search engines and APIs.
"""
import dspy
from conductor.flow.models import CitedAnswer as CitedAnswerModel
from conductor.flow.models import CitedValue as CitedValueModel


class AgentBackstory(dspy.Signature):
    """
    A simple backstory for an agent that is a skilled researcher using Google-like search engines
    Use the name and research questions to add domain specific context to the backstory
    """

    agent_name: str = dspy.InputField(desc="The name of the agent")
    research_questions: list[str] = dspy.InputField(
        desc="The research questions the agent is tasked with answering"
    )
    backstory: str = dspy.OutputField(desc="The generated agent backstory")


class AgentSearchEngineResearchGoal(dspy.Signature):
    """
    The goal of the agent is to collect data from search engines and APIs
    Use the name and research questions to add domain specific context to the goal
    """

    agent_name: str = dspy.InputField(desc="The name of the agent")
    research_questions: list[str] = dspy.InputField(
        desc="The research questions the agent is tasked with answering"
    )
    search_engine_research_goal: str = dspy.OutputField(
        desc="The generated search engine research goal"
    )


# research
class ResearchTaskDescription(dspy.Signature):
    """
    The task description should be generated from the agent's role, research question, goal, and backstory to drive the collection of data
    using Google-like search engine queries.
    """

    agent_role = dspy.InputField(desc="The role of the agent in the research team")
    agent_research_question = dspy.InputField(
        desc="The research question the agent is tasked with collecting data for"
    )
    agent_goal = dspy.InputField(desc="The goal of the agent in the research team")
    agent_backstory = dspy.InputField(
        desc="The backstory of the agent in the research team"
    )
    task_description = dspy.OutputField(
        desc="The generated search engine research task description"
    )


# search
class SearchTaskDescription(dspy.Signature):
    """Generate a vector database search task description that will drive retrieval of data"""

    agent_role = dspy.InputField(desc="The role of the agent in the search team")
    agent_research_question = dspy.InputField(
        desc="The research question the agent is tasked with searching for"
    )
    agent_goal = dspy.InputField(desc="The goal of the agent in the search team")
    agent_backstory = dspy.InputField(
        desc="The backstory of the agent in the search team"
    )
    task_description = dspy.OutputField(
        desc="The generated vector database search description"
    )


class SearchTaskExpectedOutput(dspy.Signature):
    """Expected output for the vector database search task"""

    agent_role = dspy.InputField(desc="The role of the agent in the search team")
    agent_research_question = dspy.InputField(
        desc="The research question the agent is tasked with searching for"
    )
    agent_goal = dspy.InputField(desc="The goal of the agent in the search team")
    agent_backstory = dspy.InputField(
        desc="The backstory of the agent in the search team"
    )
    expected_output = dspy.OutputField(
        desc="Output of the search task with source citations"
    )


class CitedAnswer(dspy.Signature):
    question: str = dspy.InputField(desc="The question to be answered")
    documents: list[str] = dspy.InputField(
        desc="The documents used to generate the answer"
    )
    answer: CitedAnswerModel = dspy.OutputField(desc="The answer to the question")


class CitedValue(dspy.Signature):
    """Best value for a question"""

    question: str = dspy.InputField(desc="The question to be answered")
    documents: list[str] = dspy.InputField(
        desc="The documents used to generate the value"
    )
    value: CitedValueModel = dspy.OutputField(desc="Best value to the question")


class DescriptionSpecification(dspy.Signature):
    """
    You are build a query from a value name, description, and specification.
    The value name is the name of a data field you need to search for.
    The description is the description of the data field.
    The specification is the specification is context for the query. The query should only return results that match the specification.
    """

    value_name: str = dspy.InputField(desc="The name of the value to search for")
    description: str = dspy.InputField(desc="The description of the value")
    specification: str = dspy.InputField(desc="The specification of the value")
    specified_retrieval_question: str = dspy.OutputField(
        desc="The specified retrieval question"
    )


class QuestionHyde(dspy.Signature):
    """
    Take a question and create a hypothetical document that would answer the question.
    Only use the elements of the question to create the document.
    """

    question: str = dspy.InputField(desc="The question to be answered")
    document: str = dspy.OutputField(
        desc="The hypothetical document that would answer the question"
    )


class ExtractValue(dspy.Signature):
    """
    Distill an answer to an answer into a value that would fit into a database.
    Use the question to help understand which value to extract.
    """

    question: str = dspy.InputField(desc="The question to be answered")
    answer: str = dspy.InputField(desc="The answer to the question")
    value: str = dspy.OutputField(desc="The extracted value")


class CompanySearchQuestions(dspy.Signature):
    """
    Generate analytical research questions for company research
    Use the perspective of the analyst to generate the queries
    The questions should help a novice analyst understand the company like a subject matter expert
    """

    company_name: str = dspy.InputField(desc="The name of the company")
    perspective: str = dspy.InputField(desc="The perspective of the analyst")
    search_queries: list[str] = dspy.OutputField(desc="The generated search queries")
