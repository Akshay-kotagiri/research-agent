from __future__ import annotations

from typing import Annotated, Any
from dataclasses import dataclass, field

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


@dataclass
class ResearchState:
    """Shared state passed between agents in the research graph."""

    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)

    # The original research query
    query: str = ""

    # Accumulated research findings keyed by source/agent
    findings: dict[str, Any] = field(default_factory=dict)

    # URLs or document identifiers already visited
    visited_sources: list[str] = field(default_factory=list)

    # Final synthesized answer produced by the writer agent
    final_report: str = ""

    # Controls which agent runs next; None lets the graph decide
    next_agent: str | None = None
