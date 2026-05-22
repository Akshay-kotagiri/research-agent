from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

from src.state import ResearchState
from src.agents import planner_agent, researcher_agent, writer_agent


def _route(state: ResearchState) -> str:
    """Route to the next agent based on state.next_agent."""
    return state.next_agent or END


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", _route, {"researcher": "researcher", END: END})
    graph.add_conditional_edges("researcher", _route, {"writer": "writer", END: END})
    graph.add_conditional_edges("writer", _route, {END: END})

    return graph.compile()


# Module-level compiled graph — import and invoke directly
research_graph = build_graph()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        query = "What are the latest breakthroughs in quantum computing?"
        initial_state = ResearchState(query=query)
        result = await research_graph.ainvoke(initial_state)
        print(result["final_report"])

    asyncio.run(main())
