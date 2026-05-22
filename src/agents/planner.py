from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ResearchState

_llm = ChatOpenAI(model="gpt-4o", temperature=0)

_SYSTEM = """You are a research planner. Given a user query, break it into focused
sub-questions that a researcher can answer one at a time. Return a numbered list of
sub-questions only — no preamble."""


def planner_agent(state: ResearchState) -> dict:
    """Decompose the top-level query into sub-questions."""
    response = _llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=state.query),
        ]
    )
    return {
        "messages": [response],
        "findings": {"plan": response.content},
        "next_agent": "researcher",
    }
