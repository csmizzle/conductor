"""
Builder signatures for agents
"""
import dspy


class AgentTitle(dspy.Signature):
    """
    Generate an agent title based on the research team title and report section title
    """

    research_team_title: str = dspy.InputField(prefix="Research Team Title: ")
    report_section_title: str = dspy.InputField(prefix="Report Section Title: ")
    agent_title: str = dspy.OutputField(prefix="Agent Title: ")


class ResearchQuestions(dspy.Signature):
    """
    Generate research questions for an agent based on the research team title, agent perspective, and agent title
    Perspectives are user or analyst input that should be used to generate tailored research questions
    """

    research_team_title: str = dspy.InputField(prefix="Research Team Title: ")
    agent_perspective: str = dspy.InputField(prefix="Agent Perspective: ")
    agent_title: str = dspy.InputField(prefix="Agent Title: ")
    research_questions: list[str] = dspy.OutputField(prefix="Research Questions: ")


class Perspective(dspy.Signature):
    """
    Generate a perspective for an agent based on the research team title and report section title
    """

    research_team_title: str = dspy.InputField(prefix="Research Team Title: ")
    report_section_title: str = dspy.InputField(prefix="Report Section Title: ")
    perspective: str = dspy.OutputField(prefix="Agent Perspective: ")
