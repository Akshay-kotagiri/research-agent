from __future__ import annotations

import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.state import ResearchState

_llm = ChatOpenAI(model="gpt-4o", temperature=0)

_SYSTEM = """You are a research agent with access to web search and page fetching tools.
Answer the sub-questions in the plan as thoroughly as possible. Use web_search to find
relevant sources, then fetch_page to read them in detail. Cite sources with URLs."""

_SERVERS_DIR = os.path.join(os.path.dirname(__file__), "..", "mcp_servers")

_MCP_SERVERS = {
    "search": {
        "command": sys.executable,
        "args": [os.path.join(_SERVERS_DIR, "search_server.py")],
        "transport": "stdio",
    },
    "fetch": {
        "command": sys.executable,
        "args": [os.path.join(_SERVERS_DIR, "fetch_server.py")],
        "transport": "stdio",
    },
}


async def researcher_agent(state: ResearchState) -> dict:
    """Gather information for each sub-question using MCP search and fetch tools."""
    plan = state.findings.get("plan", "")
    prompt = f"Research plan:\n{plan}\n\nOriginal query: {state.query}"

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ]
    new_messages = []

    client = MultiServerMCPClient(_MCP_SERVERS)
    tools = await client.get_tools()
    llm_with_tools = _llm.bind_tools(tools)

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool = next(t for t in tools if t.name == tool_call["name"])
            try:
                result = await tool.ainvoke(tool_call["args"])
                content = str(result)
            except (ToolException, Exception) as exc:
                content = f"Tool error: {exc}. Try a different source."
            tool_msg = ToolMessage(
                content=content, tool_call_id=tool_call["id"]
            )
            messages.append(tool_msg)
            new_messages.append(tool_msg)

    return {
        "messages": new_messages,
        "findings": {**state.findings, "research": response.content},
        "next_agent": "writer",
    }
