from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ResearchState

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

_SYSTEM = """You are a research writer. Synthesize the research findings into a clear,
well-structured report. Include an executive summary, key findings, and conclusion."""


def writer_agent(state: ResearchState) -> dict:
    """Synthesize gathered research into a final report."""
    research = state.findings.get("research", "")
    prompt = (
        f"Original query: {state.query}\n\n"
        f"Research findings:\n{research}\n\n"
        "Write the final report."
    )

    response = _llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    return {
        "messages": [response],
        "final_report": response.content,
        "next_agent": None,
    }
