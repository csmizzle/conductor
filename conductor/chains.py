from conductor.prompts import input_prompt
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI


def create_conductor_search(
    job_id: str, geography: str, titles: list[str], industries: list[str]
) -> str:
    """Generate a search input for a conductor job

    Args:
        job_id (str): Conductor job id
        geography (str): Geography to search
        titles (list[str]): Titles to search
        industries (list[str]): Industries to search

    Returns:
        str: agent query
    """
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4-0125-preview", temperature=0),
        prompt=input_prompt,
    )
    response = chain.run(
        job_id=job_id, geography=geography, titles=titles, industries=industries
    )
    return response
