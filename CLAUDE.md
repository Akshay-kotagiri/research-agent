# Research Agent — Claude Code Guide

## Project Overview

A multi-agent research system built with **LangGraph** (agent orchestration) and **MCP**
(Model Context Protocol, for tool servers). Three specialized agents collaborate to answer
research queries: a planner, a researcher, and a writer.

## Architecture

```
src/
├── __init__.py
├── state.py          # Shared ResearchState dataclass (LangGraph state)
├── graph.py          # LangGraph graph definition + compiled research_graph
├── agents/
│   ├── __init__.py
│   ├── planner.py    # Breaks query into sub-questions
│   ├── researcher.py # Gathers information for each sub-question
│   └── writer.py     # Synthesizes findings into a final report
└── mcp_servers/
    ├── __init__.py
    ├── search_server.py  # MCP server exposing web_search tool
    └── fetch_server.py   # MCP server exposing fetch_page tool
```

### Agent Flow

```
[planner] → [researcher] → [writer] → END
```

Each agent returns `next_agent` in its state update. The `_route` function in
`graph.py` reads this field to decide which node runs next.

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables (create a `.env` file):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Running

```bash
python -m src.graph
```

Or import `research_graph` in your own script:

```python
from src import research_graph, ResearchState

result = research_graph.invoke(ResearchState(query="Your research question here"))
print(result["final_report"])
```

## MCP Servers

The two MCP servers in `src/mcp_servers/` are **stdio-based** and must be registered
with a compatible MCP host (e.g., Claude Desktop, Claude Code) or connected
programmatically via the MCP Python SDK.

- **search_server** — `web_search(query, num_results)`: plug in your preferred search API
  (Brave Search, Serper, Tavily, etc.) by editing the endpoint in `search_server.py`.
- **fetch_server** — `fetch_page(url, max_chars)`: fetches a URL and strips HTML to
  return clean text. No API key required.

## Adding a New Agent

1. Create `src/agents/your_agent.py` with a function `your_agent(state: ResearchState) -> dict`.
2. Export it from `src/agents/__init__.py`.
3. Add a node and edges in `src/graph.py`.
4. Update `_route` or `next_agent` logic so existing agents can hand off to it.

## Key Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Agent graph orchestration |
| `langchain-anthropic` | Claude model integration |
| `mcp` | Model Context Protocol SDK |
| `httpx` | Async HTTP client for MCP servers |
| `beautifulsoup4` | HTML parsing in fetch server |
