"""
Day-5 LangGraph pipeline graph topology.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

# The nodes are imported from automl_agents.nodes
# Ah, we exposed prep_node as prep_node in nodes/__init__.py, let's import it exactly.
from automl_agents.nodes import (
    profiler_node,
    prep_node,
    selector_node,
    trainer_node,
    reporter_node,
)
from automl_agents.schemas import PipelineState, RunConfig

# Build the workflow StateGraph
builder = StateGraph(PipelineState, context_schema=RunConfig)

# Add nodes
builder.add_node("profiler", profiler_node)
builder.add_node("data_prep", prep_node)
builder.add_node("selector", selector_node)
builder.add_node("trainer", trainer_node)
builder.add_node("reporter", reporter_node)

# Add static edges
builder.add_edge(START, "profiler")
builder.add_edge("profiler", "data_prep")
builder.add_edge("data_prep", "selector")
builder.add_edge("selector", "trainer")
builder.add_edge("trainer", "reporter")
builder.add_edge("reporter", END)

# Compile graph
graph = builder.compile()
