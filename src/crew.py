from __future__ import annotations

import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
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
    researcher = Agent(
        role="Research Agent",
        goal="Collect reliable information from the web and local knowledge base to answer the user's query.",
        backstory=(
            "You are a careful researcher. You always gather evidence first, "
            "prefer citing sources, and you do not hallucinate."
        ),
        llm=llm,
        tools=[tavily_tool, local_rag_tool],
        verbose=True,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Analyst Agent",
        goal="Analyze research notes, reconcile conflicts, compute any needed numbers, and produce bulletproof conclusions.",
        backstory=(
            "You are an analytical thinker. You validate claims, summarize key points, "
            "and use the calculator tool when math is needed."
        ),
        llm=llm,
        tools=[calc_tool, local_rag_tool],
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer Agent",
        goal="Write the final grounded answer in a clean report format with clear sections and sources.",
        backstory=(
            "You are an excellent technical writer. You produce structured markdown, "
            "include sources, and write only what can be supported."
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
            "2) Resolve any conflicts (if sources disagree, say so).\n"
            "3) If any numbers are mentioned, verify using calculator.\n"
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