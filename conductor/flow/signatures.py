"""
Agent signatures for research team
The signatures need to be defined to control the searching and aggregation
of data collection efforts using search engines and APIs.
"""
import dspy

# configure dspy
llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)


# research
class ResearchTaskDescription(dspy.Signature):
    """Generate a search engine research task description that will drive open source data collection"""

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


class ResearchTaskExpectedOutput(dspy.Signature):
    """A confirmation of the different types of data the agent collected"""

    agent_role = dspy.InputField(desc="The role of the agent in the research team")
    agent_research_question = dspy.InputField(
        desc="The research question the agent is tasked with collecting data for"
    )
    agent_goal = dspy.InputField(desc="The goal of the agent in the research team")
    agent_backstory = dspy.InputField(
        desc="The backstory of the agent in the research team"
    )
    expected_output = dspy.OutputField(
        desc="A confirmation of the different types of data the agent collected"
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
