from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from conductor.llms import claude_sonnet
from langchain_elasticsearch import ElasticsearchRetriever
from conductor.rag.embeddings import BedrockEmbeddings
from typing import Dict
import os
import logging


logger = logging.getLogger(__name__)
index_name = os.getenv("ELASTICSEARCH_INDEX")


def vector_query(search_query: str) -> Dict:
    vector = BedrockEmbeddings().embed_query(
        search_query
    )  # same embeddings as for indexing
    return {
        "knn": {
            "field": "vector",
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10,
        }
    }


vector_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=vector_query,
    content_field="text",
    url=os.getenv("ELASTICSEARCH_URL"),
)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Each sentence should have at least one source.
If there are multiple sources for a sentence, separate them like this: [1][2].
Sources should be in the form of urls.
If there is not a url and instead a description of an image, use the description in the source.
Each URL is Sources should be unique and not repeated.
Sources can be repeated if they are used in different sentences.
Only include the sources that are relevant to the question.
Do not say things like "According to the sources" or similar, make the answers sound natural and analytical.

Example Answer with Sources:

Acme Corp is run by John Doe.[1][2] He is the CEO of the company.[1] John has a background in finance and has worked in the industry for over 20 years.[1][2][3]

Sources:
[1] https://abc.com/source1
[2] https://efg.com/source2
[3] https://lmn.com/source3

End of Example Answer

Use the sources below to add citations to your answer in for form of footnotes.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content + "\nSource:" + doc.metadata["_source"]["metadata"]["url"]
        if "url" in doc.metadata["_source"]["metadata"]
        else "None"
        for doc in docs
    )


search_chain = (
    {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | claude_sonnet
    | StrOutputParser()
)


def search(query: str) -> str:
    logger.info("Searching index ...")
    response = search_chain.invoke(input=query)
    logger.info("Search complete ...")
    return response
