from conductor.agents.graph import workflow
import pprint


def test_workflow():
    inputs = {
        "messages": [
            ("user", "What is Steve Rubl"),
        ]
    }
    graph = workflow.compile()
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")


test_workflow()
