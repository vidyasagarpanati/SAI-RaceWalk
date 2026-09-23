"""LangGraph wiring.

The pipeline is linear today, so the graph is a chain with a failure edge to END.
It exists because branching is coming: multi-camera fusion, multi-athlete
comparison, and a per-section narrate/verify repair loop all want real state.
Every node is also runnable on its own via ``racewalk run --only S4``, so nothing
here is required for debugging.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from racewalk.context import Context
from racewalk.contracts import StepFailed
from racewalk.runner import execute_step
from racewalk.runstate import STEP_NAMES


class PipelineState(TypedDict, total=False):
    run_id: str
    completed: list[str]
    failed: str | None
    error: str | None
    stats: dict[str, Any]


def _merge_stats(old: dict, new: dict) -> dict:
    merged = dict(old or {})
    merged.update(new or {})
    return merged


def build_graph(ctx: Context, steps: list[str], force: bool = False):
    from langgraph.graph import END, StateGraph

    graph = StateGraph(PipelineState)

    def make_node(step_id: str):
        def node(state: PipelineState) -> PipelineState:
            try:
                result = execute_step(ctx, step_id, force=force)
            except StepFailed as exc:
                return {"failed": step_id, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                return {"failed": step_id, "error": f"{type(exc).__name__}: {exc}"}
            return {
                "completed": [*state.get("completed", []), step_id],
                "stats": _merge_stats(state.get("stats", {}), {step_id: result.stats}),
            }
        return node

    for step_id in steps:
        graph.add_node(step_id, make_node(step_id))

    def gate(state: PipelineState) -> str:
        return "stop" if state.get("failed") else "go"

    graph.set_entry_point(steps[0])
    for current, nxt in zip(steps, steps[1:]):
        graph.add_conditional_edges(current, gate, {"go": nxt, "stop": END})
    graph.add_edge(steps[-1], END)
    return graph.compile()


def run_graph(ctx: Context, steps: list[str], force: bool = False) -> PipelineState:
    app = build_graph(ctx, steps, force=force)
    final: PipelineState = app.invoke({"run_id": ctx.run_id, "completed": [], "stats": {}})
    return final


def describe(steps: list[str]) -> str:
    return " -> ".join(f"{s}:{STEP_NAMES[s]}" for s in steps)
