from conductor.rag.embeddings import BedrockEmbeddings


def test_bedrock_embeddings():
    bedrock_embeddings = BedrockEmbeddings()
    embeddings = bedrock_embeddings.embed_query("Hello, world!")
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0
