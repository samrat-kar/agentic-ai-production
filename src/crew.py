# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""CrewAI agent, task, and crew definitions.

Defines three agents (Research Agent, Analyst Agent, Writer Agent), their
tasks, and the sequential crew pipeline that orchestrates the full research
workflow: Research Agent → Analyst Agent → Writer Agent.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process

logger = logging.getLogger(__name__)
from crewai_tools import TavilySearchTool
from langchain_openai import ChatOpenAI

from .tools import build_vectordb, LocalRAGSearchTool, CalculatorTool, SaveReportTool


def build_crew(data_dir: str = "data") -> Crew:
    load_dotenv()

    # LLM (LangChain OpenAI wrapper)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.2)

    # Tools
    vdb = build_vectordb(data_dir=data_dir)
    local_rag_tool = LocalRAGSearchTool(vdb=vdb)
    tavily_tool = TavilySearchTool()  # requires TAVILY_API_KEY
    calc_tool = CalculatorTool()
    save_tool = SaveReportTool()

    # -----------------------------
    # Agents (minimum 3, distinct roles)
    # -----------------------------
    # Research Agent: first stage — evidence gathering only, no conclusions.
    # Tools: TavilySearchTool (live web), LocalRAGSearchTool (local corpus).
    researcher = Agent(
        role="Research Agent",
        goal="Collect reliable information from the web and local knowledge base to answer the user's query.",
        backstory=(
            "You are a meticulous evidence gatherer. Your sole job is to retrieve raw facts "
            "from the live web (via Tavily) and from the local document corpus (via local_rag_search). "
            "You do not draw conclusions — you compile evidence and cite every source. "
            "You never fabricate information; if nothing relevant is found, you say so."
        ),
        llm=llm,
        tools=[tavily_tool, local_rag_tool],
        verbose=True,
        allow_delegation=False,
    )

    # Analyst Agent: second stage — validation and synthesis, not re-research.
    # Tools: CalculatorTool (arithmetic verification), LocalRAGSearchTool (cross-checking claims).
    analyst = Agent(
        role="Analyst Agent",
        goal="Analyse research notes, reconcile conflicts, verify numbers, and produce a clear, evidence-backed answer outline.",
        backstory=(
            "You are a critical analyst. You receive research notes from the Research Agent and "
            "rigorously evaluate them: you identify key conclusions, surface conflicts between sources, "
            "and verify any numeric claims using the calculator tool. When sources disagree you say so "
            "explicitly. You may re-query the local knowledge base to cross-check claims. "
            "Your output is a structured answer outline — not a final report."
        ),
        llm=llm,
        tools=[calc_tool, local_rag_tool],
        verbose=True,
        allow_delegation=False,
    )

    # Writer Agent: third stage — report composition only, no retrieval.
    # Tools: SaveReportTool (writes final Markdown to ./outputs/).
    # Intentionally has no retrieval tools — writes only from Analyst-provided evidence.
    writer = Agent(
        role="Writer Agent",
        goal="Write the final grounded answer in a clean, structured Markdown report with clear sections and sources, then save it.",
        backstory=(
            "You are a precise technical writer. You receive a validated answer outline from the Analyst Agent "
            "and turn it into a well-structured Markdown report. You include a short answer, a detailed "
            "explanation with bullet points, and a Sources section. You write only what is supported by "
            "the provided evidence — you do not introduce new facts or speculation. "
            "You always save the finished report using the save_report tool."
        ),
        llm=llm,
        tools=[save_tool],
        verbose=True,
        allow_delegation=False,
    )

    # -----------------------------
    # Tasks (coordination / communication)
    # -----------------------------
    research_task = Task(
        description=(
            "User question: {question}\n\n"
            "1) Use Tavily web search to find relevant information.\n"
            "2) Use local_rag_search to pull any local evidence from ./data.\n"
            "3) Produce research notes with bullets + a short 'Sources' list.\n"
            "Be concise but evidence-driven."
        ),
        expected_output=(
            "Research notes in bullet points, followed by a 'Sources' section listing URLs and/or local file sources."
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Take the Research Agent's notes and:\n"
            "1) Identify the key conclusions that answer the question.\n"
            "2) Resolve any conflicts between sources — if sources disagree, surface the conflict explicitly.\n"
            "3) If any numbers are mentioned, verify them using the calculator tool.\n"
            "4) Use local_rag_search to cross-check any claims against the local knowledge base if needed.\n"
            "Return an analysis summary + final recommended answer outline."
        ),
        expected_output="Analysis summary and a clear answer outline (sections + key bullets).",
        agent=analyst,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "Write a final markdown report answering: {question}\n\n"
            "Must include:\n"
            "- Short answer\n"
            "- Explanation with bullet points\n"
            "- Sources section\n\n"
            "Then save it using save_report as 'report.md'."
        ),
        expected_output="A saved report path and the final markdown content.",
        agent=writer,
        context=[analysis_task],
    )

    # Crew orchestration
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # clear coordination: research -> analyze -> write
        verbose=True,
    )
    return crew