from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx

app = Server("search-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the web and return a list of relevant results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]
    num_results = arguments.get("num_results", 5)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": num_results,
            },
        )
        response.raise_for_status()
        data = response.json()

    results = "\n\n".join(
        f"{r['title']}\n{r['url']}\n{r.get('content', '')}"
        for r in data.get("results", [])
    )
    return [TextContent(type="text", text=results or "No results found.")]


class SearchServer:
    """Entry point wrapper for the MCP search server."""

    @staticmethod
    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(SearchServer.run())
