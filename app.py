from __future__ import annotations

import asyncio

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.state import ResearchState  # noqa: E402 — must come after load_dotenv
from src.graph import research_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Agent")
st.caption("Powered by GPT-4o · Tavily Web Search · LangGraph multi-agent pipeline")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, plan?}

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("plan"):
            with st.expander("📋 Research plan"):
                st.markdown(msg["plan"])

# ---------------------------------------------------------------------------
# Chat input — submits on Enter
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask your research question…"):

    # Show the user's question immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the multi-agent graph and stream intermediate status
    with st.chat_message("assistant"):
        status = st.status("Running research pipeline…", expanded=True)

        async def _run() -> tuple[dict, str]:
            final: dict = {}
            collected_plan = ""
            async for chunk in research_graph.astream(
                ResearchState(query=prompt),
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if node == "planner":
                        collected_plan = update.get("findings", {}).get("plan", "")
                        status.write("**🗂 Planner** — broke query into sub-questions")
                        status.write(f"```\n{collected_plan}\n```")
                    elif node == "researcher":
                        status.write("**🔍 Researcher** — gathered findings from the web")
                    elif node == "writer":
                        status.write("**✍️ Writer** — synthesizing final report…")
                    final.update(update)
            return final, collected_plan

        result, plan_text = asyncio.run(_run())

        status.update(label="Research complete!", state="complete", expanded=False)

        report_text = result.get("final_report", "No report generated.")
        plan_text = plan_text or result.get("findings", {}).get("plan", "")

        st.markdown(report_text)

        if plan_text:
            with st.expander("📋 Research plan"):
                st.markdown(plan_text)

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": report_text,
        "plan": plan_text,
    })
