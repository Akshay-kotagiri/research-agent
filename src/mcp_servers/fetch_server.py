from __future__ import annotations

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx
from bs4 import BeautifulSoup

app = Server("fetch-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="fetch_page",
            description="Fetch and extract readable text content from a URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return",
                        "default": 8000,
                    },
                },
                "required": ["url"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "fetch_page":
        raise ValueError(f"Unknown tool: {name}")

    url = arguments["url"]
    max_chars = arguments.get("max_chars", 8000)

    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        response = await client.get(url, headers={"User-Agent": "ResearchAgent/1.0"})
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)[:max_chars]
    return [TextContent(type="text", text=text)]


class FetchServer:
    """Entry point wrapper for the MCP fetch server."""

    @staticmethod
    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(FetchServer.run())
