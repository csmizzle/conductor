from conductor.retrievers.pinecone_ import (
    create_gpt4_pinecone_apollo_retriever,
    create_gpt4_pinecone_discord_retriever,
    create_fireworks_pinecone_apollo_retriever,
)
from langsmith import unit


@unit
def test_discord_retriever() -> None:
    retriever = create_gpt4_pinecone_discord_retriever()
    assert retriever is not None
    response = retriever.invoke(
        {
            "query": "Do we want venture capital? Do we mention any organizations like that in the channel?"
        }
    )
    assert isinstance(response, dict)
    assert isinstance(response["result"], str)
    assert len(response["result"]) > 0


@unit
def test_apollo_retriever() -> None:
    retriever = create_gpt4_pinecone_apollo_retriever()
    assert retriever is not None
    response = retriever.invoke(
        {"query": "Are there any key players we should know about? Tell me about them."}
    )
    print(response)
    assert isinstance(response, dict)
    assert isinstance(response["result"], str)
    assert len(response["result"]) > 0


@unit
def test_apollo_fireworks_retriever() -> None:
    retriever = create_fireworks_pinecone_apollo_retriever()
    assert retriever is not None
    response = retriever.invoke(
        {"query": "Are there any key players we should know about? Tell me about them."}
    )
    print(response)
    assert isinstance(response, dict)
    assert isinstance(response["result"], str)
    assert len(response["result"]) > 0
