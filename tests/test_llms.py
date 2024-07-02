from conductor.llms import claude_sonnet


def test_claude_sonnet():
    response = claude_sonnet.invoke("Tell me a joke")
    assert response is not None
