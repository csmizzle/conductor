from conductor.agents.graph import workflow


def test_workflow():
    inputs = {
        "messages": [
            ("user", "What is Steve Rubley's background?"),
        ]
    }
    graph = workflow.compile()
    output = graph.invoke(inputs)
    assert isinstance(output, dict)
