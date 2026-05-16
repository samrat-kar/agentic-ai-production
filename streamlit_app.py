# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""Streamlit web interface for the Multi-Agent RAG Research Assistant.

Run with:
    streamlit run streamlit_app.py

Supports two research modes:
- Quick Mode  : single RAGAssistant agent, local corpus only (fast).
- Full Research: three-agent CrewAI pipeline with live web search (thorough).

Safety guardrails (src/safety.py) and resilience helpers (src/resilience.py)
are applied before any question reaches the agents.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging setup (file + console)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Sidebar — health status
# ---------------------------------------------------------------------------


def _render_health_status() -> None:
    """Render a compact health status panel in the sidebar."""
    from src.health import check_environment, check_data_directory

    env = check_environment()
    openai_ok = env["OPENAI_API_KEY"] == "ok"
    tavily_ok = env["TAVILY_API_KEY"] == "ok"

    data_dir = st.session_state.get("_health_data_dir", "data")
    data = check_data_directory(data_dir)
    data_ok = data["status"] == "ok"

    def _badge(ok: bool, label: str) -> str:
        colour = "green" if ok else "red"
        icon = "✓" if ok else "✗"
        return f":{colour}[{icon} {label}]"

    st.sidebar.markdown(
        f"{_badge(openai_ok, 'OpenAI key')}  "
        f"{_badge(tavily_ok, 'Tavily key')}  "
        f"{_badge(data_ok, f'{data[\"file_count\"]} docs')}"
    )
    if not openai_ok:
        st.sidebar.warning("OpenAI API key is missing — queries will fail.")
    if not data_ok:
        st.sidebar.info(f"Knowledge base: {data.get('message', data['status'])}")


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------


def _render_sidebar() -> Dict[str, Any]:
    """Render the sidebar and return the user's configuration choices."""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("### API Keys")
    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required for LLM and embeddings.  Leave blank to use the .env value.",
    )
    tavily_key = st.sidebar.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Required for Full Research mode (live web search). Leave blank to use .env.",
    )

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Research Mode")
    mode = st.sidebar.radio(
        "Mode",
        options=["Quick Mode", "Full Research"],
        index=0,
        help=(
            "**Quick Mode** — single agent, local corpus only, no web search (fast).\n\n"
            "**Full Research** — 3-agent CrewAI pipeline with live Tavily web search (thorough, slower)."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced")
    timeout = st.sidebar.slider(
        "Timeout (seconds)",
        min_value=30,
        max_value=300,
        value=120,
        step=10,
        help="Maximum time to wait for a response before aborting.",
    )
    top_k = st.sidebar.slider(
        "RAG top-k chunks",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of local document chunks retrieved per query.",
    )
    data_dir = st.sidebar.text_input(
        "Knowledge base directory",
        value="data",
        help="Path to the folder containing .txt / .md / .csv / .json files.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    _render_health_status()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**About** · "
        "[GitHub](https://github.com/samrat-kar/agentic-ai-production) · "
        "[Docs](README.md) · "
        "[Issues](https://github.com/samrat-kar/agentic-ai-production/issues)"
    )

    return {
        "mode": mode,
        "timeout": timeout,
        "top_k": top_k,
        "data_dir": data_dir,
    }


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, Any]] = []
    if "assistant" not in st.session_state:
        st.session_state.assistant = None


# ---------------------------------------------------------------------------
# Research runners
# ---------------------------------------------------------------------------


def _ensure_assistant(config: Dict[str, Any]) -> None:
    """Initialise RAGAssistant in the main thread and cache it in session_state.

    Must be called from the main Streamlit thread before any timeout-wrapped
    worker runs, because session_state is not accessible from background threads.
    """
    from src.app import RAGAssistant

    if st.session_state.assistant is None:
        with st.spinner("Initialising assistant and ingesting documents…"):
            assistant = RAGAssistant()
            assistant.load_and_ingest(config["data_dir"])
            st.session_state.assistant = assistant


def _run_quick(question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the single-agent RAGAssistant.

    The assistant instance is passed via config["_assistant"] so this function
    never touches st.session_state (which is unsafe inside background threads).
    """
    assistant = config["_assistant"]
    return assistant.query_with_agent(question, n_results=config["top_k"])


def _run_full_crew(question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full 3-agent CrewAI pipeline."""
    from src.crew import build_crew

    crew = build_crew(data_dir=config["data_dir"])
    raw = crew.kickoff(inputs={"question": question})
    return {
        "question": question,
        "answer": str(raw),
        "sources": [],
        "mode": "full_crew",
    }


def _run_research(question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the correct runner based on the selected mode."""
    from src.resilience import run_with_timeout
    from src.safety import filter_output

    runner = _run_quick if config["mode"] == "Quick Mode" else _run_full_crew
    result = run_with_timeout(runner, config["timeout"], question, config)

    if "answer" in result:
        result["answer"] = filter_output(result["answer"])
    return result


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


def _render_result(result: Dict[str, Any]) -> None:
    """Display the research result in the main panel."""
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    mode = result.get("mode", "unknown")

    st.success("Research complete!")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Answer")
        st.markdown(answer if answer else "_No answer returned._")
    with col2:
        st.markdown("### Metadata")
        st.markdown(f"**Mode:** `{mode}`")
        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                st.markdown(f"- `{src}`")

    with st.expander("Raw JSON response"):
        st.json(result)


def _render_history() -> None:
    """Render the collapsible query history."""
    if not st.session_state.history:
        return
    with st.expander(f"Query history ({len(st.session_state.history)} items)", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            ts = entry.get("timestamp", "")
            q = entry.get("question", "")
            mode = entry.get("mode", "")
            elapsed = entry.get("elapsed_s", 0.0)
            st.markdown(f"**{i}.** `{ts}` · *{mode}* · {elapsed:.1f}s")
            st.markdown(f"> {q}")
            st.markdown("---")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    _init_session_state()
    config = _render_sidebar()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("🔬 Multi-Agent RAG Research Assistant")
    st.markdown(
        "Ask a research question. The assistant retrieves evidence from your local "
        "knowledge base (and the live web in Full Research mode) before composing a "
        "grounded, source-cited answer."
    )
    st.markdown("---")

    # ------------------------------------------------------------------
    # Input form
    # ------------------------------------------------------------------
    with st.form("research_form", clear_on_submit=False):
        question_raw = st.text_area(
            "Research question",
            placeholder="e.g. What are the key advances in quantum error correction?",
            height=100,
        )
        submitted = st.form_submit_button("🚀 Run Research", use_container_width=True)

    # ------------------------------------------------------------------
    # Validation + execution
    # ------------------------------------------------------------------
    if submitted:
        from src.safety import InputValidationError, validate_question

        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key is missing. Enter it in the sidebar or set it in `.env`.")
            st.stop()

        if config["mode"] == "Full Research" and not os.getenv("TAVILY_API_KEY"):
            st.warning(
                "Tavily API key is missing. Full Research mode requires it for web search. "
                "Switch to Quick Mode or add the key in the sidebar."
            )

        # Validate question
        try:
            question = validate_question(question_raw)
        except InputValidationError as exc:
            st.error(f"Input error: {exc}")
            st.stop()

        # For Quick Mode, initialise the assistant in the main thread now —
        # session_state is not accessible inside background threads used by
        # run_with_timeout, so this must happen before we dispatch.
        # We then pass the object via config so the worker thread never reads
        # session_state directly.
        if config["mode"] == "Quick Mode":
            try:
                _ensure_assistant(config)
                config["_assistant"] = st.session_state.assistant
            except Exception as exc:
                st.error(f"Failed to initialise assistant: {exc}")
                logger.exception("Assistant initialisation failed.")
                st.stop()

        # Run research
        start = time.time()
        result_placeholder = st.empty()

        with st.spinner(
            f"Running {config['mode']} on: *{question[:80]}{'…' if len(question) > 80 else ''}*"
        ):
            try:
                result = _run_research(question, config)
                elapsed = time.time() - start

                # Store in history
                st.session_state.history.append(
                    {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "question": question,
                        "mode": config["mode"],
                        "elapsed_s": elapsed,
                        "result": result,
                    }
                )
                logger.info(
                    "Research complete in %.1fs | mode=%s | question=%.80s",
                    elapsed,
                    config["mode"],
                    question,
                )

            except TimeoutError as exc:
                st.error(f"Timeout: {exc}")
                logger.error("Timeout for question: %.80s", question)
                st.stop()
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")
                logger.exception("Unexpected error for question: %.80s", question)
                st.stop()

        st.caption(f"Completed in {elapsed:.1f} seconds · mode: {config['mode']}")
        _render_result(result)

    # ------------------------------------------------------------------
    # History panel
    # ------------------------------------------------------------------
    _render_history()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Multi-Agent RAG Research Assistant · "
        "Powered by CrewAI, LangChain & OpenAI · "
        "CC BY-NC-SA 4.0"
    )


if __name__ == "__main__":
    main()
