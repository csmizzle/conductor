from conductor.functions.pinecone_ import search_pinecone
from langsmith import unit


@unit
def test_search_pinecone():
    query = "What is conductor? Who are key players in finance who could benefit from conductor?"
    response = search_pinecone(query)
    assert response is not None
    assert isinstance(response, dict)
    assert isinstance(response["text"], str)
