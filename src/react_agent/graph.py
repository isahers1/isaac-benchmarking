from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from operator import add
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict


class Configuration(TypedDict):
    delay: int


class BenchmarkMode(Enum):
    """
    Benchmarking scenarios to consider:
    - Single node
    - Multi-node, sequential
    - Multi-node, high concurrency parallel
    """

    SINGLE_NODE = "single"
    SEQUENTIAL_NODES = "sequential"
    PARALLEL_NODES = "parallel"


class Message(TypedDict):
    id: int
    content: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    counter: int = 0
    delay: int = 0
    data_size: int = 1000
    expand: int = 50
    mode: BenchmarkMode = BenchmarkMode.SINGLE_NODE
    messages: Annotated[list[Message], add] = field(default_factory=list)


def create_large_data(state: State) -> str:
    return "a" * state.data_size


async def entry_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """
    Entry node for the graph.

    Convert the mode from string to BenchmarkMode enum.
    """
    return {
        "counter": 0,
        "mode": BenchmarkMode(state.mode) if state.mode else BenchmarkMode.SINGLE_NODE,
        "messages": [{"id": 0, "content": "Hello from entry node!"}],
    }


async def sequential_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    state.counter += 1
    await asyncio.sleep(state.delay)
    return {
        "counter": state.counter,
        "messages": [
            {
                "id": state.counter,
                "content": f"Hello from sequential node {state.counter}! Data: {create_large_data(state)}",
            }
        ],
    }


async def parallel_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    await asyncio.sleep(state.delay)
    return {
        "messages": [
            {
                "id": state["parallel_id"],
                "content": f"Hello from parallel node {state['parallel_id']}! Data: {create_large_data(state)}",
            }
        ]
    }


def should_continue(
    state: State, config: RunnableConfig
) -> Literal["__end__", "sequential_node"]:
    match state.mode:
        case BenchmarkMode.SINGLE_NODE:
            return "__end__"
        case BenchmarkMode.SEQUENTIAL_NODES:
            if state.counter < state.expand:
                return "sequential_node"
            else:
                return "__end__"
        case BenchmarkMode.PARALLEL_NODES:
            return [
                Send("parallel_node", {"parallel_id": id})
                for id in range(1, state.expand + 1)
            ]


builder = StateGraph(State, Configuration)

graph = (
    builder.add_node(entry_node)
    .add_node(sequential_node)
    .add_node(parallel_node)
    .add_edge("__start__", "entry_node")
    .add_conditional_edges("entry_node", should_continue)
    .add_conditional_edges("sequential_node", should_continue)
    .add_edge("parallel_node", "__end__")
    .compile(name="New Graph")
)
