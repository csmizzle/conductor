from conductor.search import search


def test_search():
    query = "Who runs hivemind captial? Who is their CEO? WHat do you know about the leader? What is his background?"
    result = search(query)
    print(result)
    assert result is not None
    assert len(result) > 0
    assert "Sources:" in result
