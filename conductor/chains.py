from conductor.prompts import (
    input_prompt,
    apollo_with_job_id_input_prompt,
    apollo_input_prompt,
    apollo_input_structured_prompt,
    gmail_input_prompt,
    html_summary_prompt,
    email_prompt,
    summary_prompt,
    reduce_prompt,
)
from conductor.llms import claude_v2_1
from conductor.parsers import (
    EngagementStrategy,
    HtmlSummary,
    ApolloInput,
    html_summary_parser,
    apollo_input_parser,
)
from langchain.chains.llm import LLMChain
from langchain.chains.summarize.chain import (
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    StuffDocumentsChain,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain import hub
from langchain.docstore.document import Document


@traceable
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


@traceable
def create_engagement_strategy(apollo_people_data: str) -> EngagementStrategy:
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=input_prompt,
    )
    response = chain.invoke({"apollo_people_data": apollo_people_data})
    return response


@traceable
def create_apollo_input_with_job_id(query: str, job_id: str) -> str:
    """
    Extract Apollo input parameters from a general input string
    """
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4-0125-preview", temperature=0),
        prompt=apollo_with_job_id_input_prompt,
    )
    response = chain.invoke({"general_input": query, "job_id": job_id})
    return response


@traceable
def create_apollo_input(query: str) -> str:
    """
    Extract Apollo input parameters from a general input string
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=apollo_input_prompt,
    )
    response = chain.invoke({"general_input": query})
    return response


@traceable
def create_apollo_input_structured(query: str) -> ApolloInput:
    """
    Extract Apollo input parameters from a general input string
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=apollo_input_structured_prompt,
    )
    response = chain.invoke({"query": query})
    return apollo_input_parser.parse(text=response["text"])


@traceable
def create_gmail_input(input_: str) -> str:
    """
    Extract Gmail input parameters from a general input string
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=gmail_input_prompt,
    )
    response = chain.invoke({"general_input": input_})
    return response


@traceable
def create_html_summary(raw: str) -> str:
    """
    Summarize the HTML content of a web page
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=html_summary_prompt,
    )
    response = chain.invoke({"raw": raw})
    return response


@traceable
def create_email_from_context(tone: str, context: str, sign_off: str) -> str:
    """
    Create an email from a context
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=email_prompt,
    )
    response = chain.invoke({"tone": tone, "context": context, "sign_off": sign_off})
    return response


@traceable
def get_parsed_html_summary(content: str) -> HtmlSummary:
    """
    Run html_chain and get parsed summary
    """
    response = create_html_summary(content)
    html_summary = html_summary_parser.parse(response["text"])
    html_summary.content = content
    return html_summary


@traceable
def summarize(query: str, content: str) -> str:
    """
    Summarize content
    """
    chain = LLMChain(llm=claude_v2_1, prompt=summary_prompt)
    response = chain.invoke({"question": query, "content": content})
    return response


@traceable
def map_reduce_summarize(contents: list[str]) -> dict:
    # create langchain docs
    docs = []
    for content in contents:
        doc = Document(page_content=content)
        docs.append(doc)
    map_prompt = hub.pull("rlm/map-prompt")
    map_chain = LLMChain(llm=claude_v2_1, prompt=map_prompt)
    reduce_chain = LLMChain(llm=claude_v2_1, prompt=reduce_prompt)
    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    return map_reduce_chain.invoke(split_docs)
