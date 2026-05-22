from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from src.state import ResearchState
from src.agents.planner import planner_agent
from src.agents.researcher import researcher_agent
from src.agents.writer import writer_agent
from src.graph import research_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai(content: str, tool_calls: list | None = None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _mock_mcp_client(tools: list | None = None) -> MagicMock:
    mock = MagicMock()
    mock.get_tools = AsyncMock(return_value=tools or [])
    return mock


# ---------------------------------------------------------------------------
# 1. Planner — breaks query into ≥3 sub-tasks
# ---------------------------------------------------------------------------

def test_planner_breaks_query_into_at_least_3_subtasks():
    plan_text = (
        "1. What is quantum computing?\n"
        "2. How do quantum computers work?\n"
        "3. What are the real-world applications?\n"
        "4. What are the current limitations?"
    )
    state = ResearchState(query="Explain quantum computing")

    with patch("src.agents.planner._llm") as mock_llm:
        mock_llm.invoke.return_value = _ai(plan_text)
        result = planner_agent(state)

    plan = result["findings"]["plan"]
    numbered = [line for line in plan.splitlines() if line.strip() and line.strip()[0].isdigit()]
    assert len(numbered) >= 3, f"Expected ≥3 sub-tasks, got {len(numbered)}: {numbered}"
    assert result["next_agent"] == "researcher"
    assert "messages" in result


def test_planner_stores_plan_in_findings():
    plan_text = "1. Sub-q A\n2. Sub-q B\n3. Sub-q C"
    state = ResearchState(query="test query")

    with patch("src.agents.planner._llm") as mock_llm:
        mock_llm.invoke.return_value = _ai(plan_text)
        result = planner_agent(state)

    assert "plan" in result["findings"]
    assert result["findings"]["plan"] == plan_text


# ---------------------------------------------------------------------------
# 2. Researcher — returns findings for each sub-task
# ---------------------------------------------------------------------------

async def test_researcher_returns_findings():
    state = ResearchState(
        query="Explain quantum computing",
        findings={"plan": "1. What is it?\n2. How does it work?\n3. Applications?"},
    )
    research_text = (
        "Quantum computers leverage superposition and entanglement. "
        "They are applied in cryptography, drug discovery, and optimization."
    )

    with patch("src.agents.researcher._llm") as mock_llm, \
         patch("src.agents.researcher.MultiServerMCPClient") as mock_cls:

        mock_cls.return_value = _mock_mcp_client()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=_ai(research_text))
        mock_llm.bind_tools.return_value = bound

        result = await researcher_agent(state)

    assert "research" in result["findings"]
    assert result["findings"]["research"].strip() != ""
    assert result["next_agent"] == "writer"
    assert "messages" in result


async def test_researcher_preserves_existing_findings():
    """Researcher merges its output with the existing findings dict."""
    state = ResearchState(
        query="test",
        findings={"plan": "1. A\n2. B\n3. C"},
    )

    with patch("src.agents.researcher._llm") as mock_llm, \
         patch("src.agents.researcher.MultiServerMCPClient") as mock_cls:

        mock_cls.return_value = _mock_mcp_client()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=_ai("Some research."))
        mock_llm.bind_tools.return_value = bound

        result = await researcher_agent(state)

    assert "plan" in result["findings"], "Existing 'plan' key must be preserved"
    assert "research" in result["findings"]


async def test_researcher_executes_tool_calls_and_feeds_results_back():
    """When the LLM returns tool calls, the researcher runs them and loops."""
    state = ResearchState(
        query="test",
        findings={"plan": "1. What is it?\n2. How does it work?"},
    )

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "quantum computing"},
            "id": "call_abc",
            "type": "tool_call",
        }],
    )
    final_msg = _ai("Here are the synthesized findings from search results.")

    mock_tool = MagicMock()
    mock_tool.name = "web_search"
    mock_tool.ainvoke = AsyncMock(return_value="Result: quantum computers use qubits")

    with patch("src.agents.researcher._llm") as mock_llm, \
         patch("src.agents.researcher.MultiServerMCPClient") as mock_cls:

        mock_cls.return_value = _mock_mcp_client(tools=[mock_tool])
        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=[tool_call_msg, final_msg])
        mock_llm.bind_tools.return_value = bound

        result = await researcher_agent(state)

    mock_tool.ainvoke.assert_called_once()
    assert result["findings"]["research"] == final_msg.content


async def test_researcher_handles_tool_errors_gracefully():
    """A failed tool call should not crash — the error is passed back to the LLM."""
    state = ResearchState(
        query="test",
        findings={"plan": "1. What is it?"},
    )

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "test"},
            "id": "call_err",
            "type": "tool_call",
        }],
    )
    final_msg = _ai("Findings despite the error.")

    mock_tool = MagicMock()
    mock_tool.name = "web_search"
    mock_tool.ainvoke = AsyncMock(side_effect=Exception("403 Forbidden"))

    with patch("src.agents.researcher._llm") as mock_llm, \
         patch("src.agents.researcher.MultiServerMCPClient") as mock_cls:

        mock_cls.return_value = _mock_mcp_client(tools=[mock_tool])
        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=[tool_call_msg, final_msg])
        mock_llm.bind_tools.return_value = bound

        result = await researcher_agent(state)

    assert result["findings"]["research"] == final_msg.content


# ---------------------------------------------------------------------------
# 3. Writer — final report has >100 words
# ---------------------------------------------------------------------------

def test_writer_produces_report_with_more_than_100_words():
    long_report = (
        "Executive Summary: This report examines the latest developments in AI for developers. "
        + " ".join([f"Key finding number {i} shows significant progress in the field." for i in range(1, 20)])
        + " Conclusion: AI tools continue to advance rapidly, transforming software development."
    )

    state = ResearchState(
        query="Top AI tools for developers",
        findings={
            "plan": "1. What tools exist?\n2. How are they used?",
            "research": "GitHub Copilot, Claude Code, and ChatGPT are leading tools.",
        },
    )

    with patch("src.agents.writer._llm") as mock_llm:
        mock_llm.invoke.return_value = _ai(long_report)
        result = writer_agent(state)

    word_count = len(result["final_report"].split())
    assert word_count > 100, f"Expected >100 words in final_report, got {word_count}"
    assert result["final_report"] == long_report


def test_writer_sets_next_agent_to_none_to_signal_end():
    """Writer must set next_agent=None so the graph routes to END."""
    state = ResearchState(
        query="test",
        findings={"plan": "1. A", "research": "Some findings."},
    )

    with patch("src.agents.writer._llm") as mock_llm:
        mock_llm.invoke.return_value = _ai("A short report.")
        result = writer_agent(state)

    assert result["next_agent"] is None


def test_writer_stores_report_in_final_report_key():
    report = "The final synthesized report content goes here."
    state = ResearchState(
        query="test",
        findings={"plan": "1. A", "research": "findings"},
    )

    with patch("src.agents.writer._llm") as mock_llm:
        mock_llm.invoke.return_value = _ai(report)
        result = writer_agent(state)

    assert result["final_report"] == report
    assert "messages" in result


# ---------------------------------------------------------------------------
# 4. Full graph — end-to-end, reaches FINISH
# ---------------------------------------------------------------------------

async def test_full_graph_runs_end_to_end_and_reaches_finish():
    planner_reply = _ai(
        "1. What AI tools exist for developers?\n"
        "2. How are these tools being adopted?\n"
        "3. What are the productivity benefits?"
    )
    researcher_reply = _ai(
        "GitHub Copilot, Claude Code, Cursor, Tabnine, and ChatGPT are the top AI tools. "
        "Adoption is growing rapidly with measurable productivity gains reported by teams."
    )
    long_report = " ".join(
        ["AI tools are transforming software development in 2026."] * 15
    )
    writer_reply = _ai(long_report)

    with patch("src.agents.planner._llm") as mock_planner, \
         patch("src.agents.researcher._llm") as mock_researcher, \
         patch("src.agents.writer._llm") as mock_writer, \
         patch("src.agents.researcher.MultiServerMCPClient") as mock_cls:

        mock_planner.invoke.return_value = planner_reply

        mock_cls.return_value = _mock_mcp_client()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=researcher_reply)
        mock_researcher.bind_tools.return_value = bound

        mock_writer.invoke.return_value = writer_reply

        result = await research_graph.ainvoke(
            ResearchState(query="Top AI tools for developers in 2026")
        )

    assert result["final_report"].strip() != "", "final_report must not be empty"
    assert len(result["final_report"].split()) > 100, "final_report must exceed 100 words"
    assert result["next_agent"] is None, "next_agent must be None — graph must have reached END"


async def test_full_graph_passes_query_through_all_agents():
    """The original query must be visible in the final state."""
    query = "What is the future of quantum computing?"

    with patch("src.agents.planner._llm") as mock_planner, \
         patch("src.agents.researcher._llm") as mock_researcher, \
         patch("src.agents.writer._llm") as mock_writer, \
         patch("src.agents.researcher.MultiServerMCPClient") as mock_cls:

        mock_planner.invoke.return_value = _ai("1. Q1\n2. Q2\n3. Q3")

        mock_cls.return_value = _mock_mcp_client()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=_ai("Research findings."))
        mock_researcher.bind_tools.return_value = bound

        long = " ".join(["word"] * 120)
        mock_writer.invoke.return_value = _ai(long)

        result = await research_graph.ainvoke(ResearchState(query=query))

    assert result["query"] == query
