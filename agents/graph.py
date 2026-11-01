"""LangGraph state machine with a dependency-free sequential fallback."""
from __future__ import annotations

from functools import partial

from agents.nodes import advisory_node, checker_node, research_node, verifier_node
from agents.state import AnalysisState

ORDER = ("research", "checker", "advisory", "verifier")


class SequentialGraph:
    """Drop-in replacement used when langgraph is not installed."""

    def __init__(self, nodes: dict, config: dict) -> None:
        self.nodes = nodes
        self.config = config

    def invoke(self, state: dict) -> dict:
        current = dict(state)
        max_retries = int(self.config.get("verifier", {}).get("max_retries", 1))
        for _ in range(max_retries + 1):
            for name in ORDER:
                update = self.nodes[name](current) or {}
                current.update(update)
            if not current.get("retry_requested"):
                break
            current["retry_requested"] = False
        return current


def _route_after_verifier(state: dict) -> str:
    return "research" if state.get("retry_requested") else "end"


def _nodes(llm, retriever, config: dict) -> dict:
    return {
        "research": partial(research_node, retriever=retriever, config=config),
        "checker": partial(checker_node, llm=llm, config=config),
        "advisory": partial(advisory_node, llm=llm, config=config),
        "verifier": partial(verifier_node, config=config),
    }


def build_graph(llm, retriever, config: dict, use_langgraph: bool = True):
    nodes = _nodes(llm, retriever, config)
    if not use_langgraph:
        return SequentialGraph(nodes, config)

    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return SequentialGraph(nodes, config)

    try:
        from langgraph.graph import START
    except ImportError:
        START = None

    graph = StateGraph(AnalysisState)
    for name, function in nodes.items():
        graph.add_node(name, function)

    if START is not None:
        graph.add_edge(START, "research")
    else:
        graph.set_entry_point("research")

    graph.add_edge("research", "checker")
    graph.add_edge("checker", "advisory")
    graph.add_edge("advisory", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"research": "research", "end": END},
    )
    return graph.compile()
