"""
Test flow components
"""
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.flow.models import Team
from conductor.flow.builders import (
    build_agent,
    build_agent_from_template,
    build_agent_search_task,
    build_agent_search_tasks,
    build_agent_search_tasks_parallel,
    build_agents_search_tasks_parallel,
    build_team,
    build_team_from_template,
    build_search_team_from_template,
)
from conductor.flow.research import (
    ResearchAgentFactory,
    ResearchQuestionAgentSearchTaskFactory,
)
from conductor.flow.search import SearchAgentFactory, SearchTaskFactory
from conductor.flow.team import (
    ResearchTeamFactory,
)
from conductor.flow.utils import (
    build_organization_determination_crew,
)
from conductor.flow.specify import (
    TaskSpecification,
    specify_research_team,
    specify_search_team,
)
from conductor.flow.flow import (
    ResearchFlow,
    run_flow,
    run_research_and_search,
    RunResult,
)
from conductor.flow.runner import run_search_team, run_team_sequential, run_team
from conductor.flow.rag import ElasticRMClient
from conductor.rag.embeddings import BedrockEmbeddings
from crewai import LLM, Agent, Task
from crewai.crew import CrewOutput
from conductor.crews.rag_marketing import tools
from langtrace_python_sdk import langtrace
from langtrace_python_sdk import with_langtrace_root_span
from elasticsearch import Elasticsearch
import dspy
import os
import json
import agentops


langtrace.init()


def test_research_agent_factory() -> None:
    agent_name = "Company Researcher"
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    builder = ResearchAgentFactory(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
    )
    agent = builder.build()
    assert isinstance(agent, Agent)


def test_build_agent() -> None:
    agent_name = "Company Researcher"
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    agent = build_agent(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
        agent_factory=ResearchAgentFactory,
    )
    assert isinstance(agent, Agent)


def test_build_agent_from_template() -> None:
    template = ResearchAgentTemplate(
        title="Company Researcher",
        research_questions=[
            "What is the company's mission?",
            "What are the company's values?",
        ],
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory, template=template, llm=llm, tools=[]
    )
    assert isinstance(agent, Agent)


def test_build_agent_task_factory() -> None:
    # build agent
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory, template=template, llm=llm, tools=[]
    )
    # build agent search task
    task_builder = ResearchQuestionAgentSearchTaskFactory(
        agent=agent, research_question=research_questions[0]
    )
    task = task_builder.build()
    assert isinstance(task, Task)


def test_build_agent_search_task() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory, template=template, llm=llm, tools=[]
    )
    task = build_agent_search_task(
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        agent=agent,
        research_question=research_questions[0],
    )
    assert isinstance(task, Task)


def test_build_agent_search_tasks() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory, template=template, llm=llm, tools=[]
    )
    tasks = build_agent_search_tasks(
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        agent=agent,
        research_questions=research_questions,
    )
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])


def test_build_agent_search_tasks_with_tools(elasticsearch_test_agent_index) -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory,
        template=template,
        llm=llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=elasticsearch_test_agent_index
            )
        ],
    )
    tasks = build_agent_search_tasks(
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        agent=agent,
        research_questions=research_questions,
    )
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])


def test_build_agent_search_tasks_parallel() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory, template=template, llm=llm, tools=[]
    )
    tasks = build_agent_search_tasks_parallel(
        agent=agent,
        research_questions=research_questions,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
    )
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])


def test_build_agents_search_tasks_parallel() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=ResearchAgentFactory, template=template, llm=llm, tools=[]
    )
    tasks = build_agents_search_tasks_parallel(
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        agents=[agent, agent],
        agent_templates=[template, template],
    )
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])


def test_research_team_factory() -> None:
    title = "Company Research Team"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    builder = ResearchTeamFactory(
        title=title,
        agent_templates=agent_templates,
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        llm=llm,
        tools=tools,
    )
    team = builder.build()
    assert isinstance(team, Team)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])


def test_build_research_team() -> None:
    title = "Company Research Team"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    team = build_team(
        title=title,
        agent_templates=agent_templates,
        llm=llm,
        tools=tools,
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    assert isinstance(team, Team)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])


def test_build_research_team_from_template() -> None:
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title, perspective=perspective, agent_templates=agent_templates
    )
    llm = LLM("openai/gpt-4o")
    tools = []
    team = build_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=tools,
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    assert isinstance(team, Team)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])


def test_organization_determination_crew(elasticsearch_test_agent_index) -> None:
    website_url = "https://www.trssllc.com"
    llm = LLM("openai/gpt-4o")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    crew = build_organization_determination_crew(
        website_url=website_url,
        llm=llm,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
    )
    result = crew.kickoff({"website_url": website_url})
    assert isinstance(result, CrewOutput)


def test_task_specification() -> None:
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title, perspective=perspective, agent_templates=agent_templates
    )
    llm = LLM("openai/gpt-4o")
    tools = []
    team = build_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=tools,
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    task = team.tasks[0]
    specifier = TaskSpecification(
        task=task, specification="The company is Thomson Reuters Special Services LLC."
    )
    specified_task = specifier.specify()
    assert isinstance(specified_task, Task)


def test_research_team_specification() -> None:
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title,
        agent_templates=agent_templates,
        perspective=perspective,
    )
    research_team = build_team_from_template(
        team_template=team_template,
        llm=LLM("openai/gpt-4o"),
        tools=[],
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    specified_team = specify_research_team(
        team=research_team,
        specification="The company is Thomson Reuters Special Services LLC.",
    )
    assert isinstance(specified_team, Team)
    assert all([isinstance(agent, Agent) for agent in specified_team.agents])
    assert all([isinstance(task, Task) for task in specified_team.tasks])


@with_langtrace_root_span()
def test_research_flow(elasticsearch_test_agent_index) -> None:
    session = agentops.init(os.getenv("AGENTOPS_API_KEY"))
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    website_url = "https://www.trssllc.com"
    llm = dspy.LM(
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # api_base=litellm_proxy_url
    )
    dspy.configure(lm=llm)
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Structure Researcher",
            research_questions=[
                "Who leads their business divisions?",
                "What are the company divisions?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Social Media Researcher",
            research_questions=[
                "What is the company's social media presence?",
                "What are the company's social media values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title,
        agent_templates=agent_templates,
        perspective=perspective,
    )
    research_team = build_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=elasticsearch_test_agent_index
            )
        ],
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    flow = ResearchFlow(
        research_team=research_team,
        website_url=website_url,
        llm=llm,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
        parallel=True,
    )
    result = run_flow(flow=flow, session_id=session)
    assert isinstance(result, list)
    assert all([isinstance(output, CrewOutput) for output in result])


def test_research_run_async(elasticsearch_cloud_test_research_index) -> None:
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    llm = dspy.LM(
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # api_base=litellm_proxy_url
    )
    dspy.configure(lm=llm)
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # base_url=litellm_proxy_url,
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
    )
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Structure Researcher",
            research_questions=[
                "Who leads their business divisions?",
                "What are the company divisions?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Social Media Researcher",
            research_questions=[
                "What is the company's social media presence?",
                "What are the company's social media values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title,
        agent_templates=agent_templates,
        perspective=perspective,
    )
    research_team = build_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch,
                index_name=elasticsearch_cloud_test_research_index,
            )
        ],
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    # specify team
    specification = "The company is Thomson Reuters Special Services LLC."
    specified_team = specify_research_team(
        team=research_team,
        specification=specification,
    )
    result = run_team(team=specified_team)
    assert isinstance(result, list)
    assert all([isinstance(output, CrewOutput) for output in result])


def test_research_flow_sequential(elasticsearch_test_agent_index) -> None:
    agentops.init(os.getenv("AGENTOPS_API_KEY"))
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    llm = dspy.LM(
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # api_base=litellm_proxy_url
    )
    dspy.configure(lm=llm)
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # base_url=litellm_proxy_url,
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Structure Researcher",
            research_questions=[
                "Who leads their business divisions?",
                "What are the company divisions?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Social Media Researcher",
            research_questions=[
                "What is the company's social media presence?",
                "What are the company's social media values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title,
        agent_templates=agent_templates,
        perspective=perspective,
    )
    research_team = build_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=elasticsearch_test_agent_index
            )
        ],
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    # specify team
    specification = "The company is Thomson Reuters Special Services LLC."
    specified_team = specify_research_team(
        team=research_team,
        specification=specification,
    )
    result = run_team_sequential(team=specified_team)
    assert isinstance(result, list)
    assert all([isinstance(output, CrewOutput) for output in result])


def test_search_agent() -> None:
    agent_name = "Company Researcher"
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    agent = build_agent(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
        agent_factory=SearchAgentFactory,
    )
    assert isinstance(agent, Agent)


def test_search_agent_task() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        agent_factory=SearchAgentFactory, template=template, llm=llm, tools=[]
    )
    task = build_agent_search_task(
        task_factory=SearchTaskFactory,
        agent=agent,
        research_question=research_questions[0],
    )
    assert isinstance(task, Task)


def test_search_team() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    title = "Company Research Team"
    perspective = "Focus on company risks and opportunities for investment"
    specification = "The company is Thomson Reuters Special Services LLC."
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Customer Base Researcher",
            research_questions=[
                "Who are the company's main customers?",
                "What services to they provide to their customers?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title, perspective=perspective, agent_templates=agent_templates
    )
    search_team = build_search_team_from_template(team=team_template)
    specified_search_team = specify_search_team(
        team=search_team,
        perspective=perspective,
        specification=specification,
    )
    answers = run_search_team(
        team=specified_search_team,
        retriever=retriever,
    )
    assert isinstance(answers, list)
    with open("./tests/data/test_search_team_results.json", "w") as f:
        json.dump([answer.model_dump() for answer in answers], f, indent=4)


def test_run_research_and_search(elasticsearch_cloud_test_research_index) -> None:
    # litellm_proxy_url = "http://0.0.0.0:4000"
    llm = dspy.LM(
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=3000,
        # api_base=litellm_proxy_url
    )
    dspy.configure(lm=llm)
    mini = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=3000,
        # base_url=litellm_proxy_url,
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
    )
    url = "https://www.trssllc.com"
    title = "Company Research Team"
    perspective = "Focus on gaps that the company can fill using their social media presence. I suspect they aren't using social media to its full potential."
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Customer Base Researcher",
            research_questions=[
                "Who are the company's main customers?",
                "What services to they provide to their customers?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Social Media Researcher",
            research_questions=[
                "What is the company's social media presence?",
                "What are the company's social media values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(
        title=title, perspective=perspective, agent_templates=agent_templates
    )
    results = run_research_and_search(
        website_url=url,
        research_llm=mini,
        research_team=team_template,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        parallel=False,
    )
    assert isinstance(results, RunResult)
    with open("./tests/data/test_team_results.json", "w") as f:
        json.dump(results.model_dump(), f, indent=4)
