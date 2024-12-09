from conductor.agents import edges, nodes, tools, state
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


workflow = StateGraph(state.AgentState)
# nodes
workflow.add_node("agent", nodes.agent)
retrieve = ToolNode([tools.get_document])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", nodes.rewrite)
workflow.add_node("generate", nodes.generate)
# edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", tools_condition, {"tools": "retrieve", END: END}
)
workflow.add_conditional_edges(
    "retrieve",
    edges.grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
