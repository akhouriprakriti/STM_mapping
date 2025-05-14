from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Optional
import pandas as pd

from src.agents.stm_parser import STMParser
from src.agents.transformation_agent import TransformAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.lineage_agent import LineageAgent
from src.agents.report_agent import ReportAgent
from src.agents.llm_chat_node import get_llm_chat_response

# Define the shared pipeline state
class STMState(TypedDict):
    raw_df: pd.DataFrame
    stm_path: str
    stm_plan: Optional[list]
    transformed_df: Optional[pd.DataFrame]
    transform_logs: Optional[list]
    validation_report: Optional[pd.DataFrame]
    lineage_log: Optional[pd.DataFrame]
    report_summary: Optional[str]
    report_paths: Optional[dict]
    user_confirmed: Optional[bool]
    llm_suggestion: Optional[str]
    llm_chat_stm: Optional[str]
    llm_chat_transform: Optional[str]
    llm_chat_validate: Optional[str]
    llm_chat_lineage: Optional[str]
    llm_chat_report: Optional[str]

# Core pipeline stage functions
def parse_stm(state): 
    parser = STMParser(state["stm_path"])
    state["stm_plan"] = parser.parse()
    return state

def transform(state): 
    transformer = TransformAgent(state["raw_df"], state["stm_plan"])
    transformed_df, logs = transformer.apply()
    state["transformed_df"] = transformed_df
    state["transform_logs"] = logs
    return state

def validate(state): 
    validator = ValidationAgent(state["transformed_df"], state["stm_plan"])
    state["validation_report"] = validator.validate()
    return state

def lineage(state): 
    lineage_agent = LineageAgent(state["raw_df"], state["transformed_df"], state["stm_plan"])
    state["lineage_log"] = lineage_agent.trace()
    return state

def report(state): 
    reporter = ReportAgent(state["transformed_df"], state["validation_report"], state["lineage_log"])
    state["report_summary"] = reporter.generate_summary()
    state["report_paths"] = reporter.export_csvs()
    return state

# LLM chat nodes after each stage
def chat_stm(state): return get_llm_chat_response("stm", state)
def chat_transform(state): return get_llm_chat_response("transform", state)
def chat_validate(state): return get_llm_chat_response("validate", state)
def chat_lineage(state): return get_llm_chat_response("lineage", state)
def chat_report(state): return get_llm_chat_response("report", state)

# Build interactive graph
def build_full_interactive_graph():
    builder = StateGraph(STMState)

    # Core + LLM chat nodes
    builder.add_node("parse_stm", RunnableLambda(parse_stm))
    builder.add_node("chat_stm", RunnableLambda(chat_stm))
    builder.add_node("transform", RunnableLambda(transform))
    builder.add_node("chat_transform", RunnableLambda(chat_transform))
    builder.add_node("validate", RunnableLambda(validate))
    builder.add_node("chat_validate", RunnableLambda(chat_validate))
    builder.add_node("lineage", RunnableLambda(lineage))
    builder.add_node("chat_lineage", RunnableLambda(chat_lineage))
    builder.add_node("report", RunnableLambda(report))
    builder.add_node("chat_report", RunnableLambda(chat_report))

    # Execution path
    builder.set_entry_point("parse_stm")
    builder.add_edge("parse_stm", "chat_stm")
    builder.add_edge("chat_stm", "transform")
    builder.add_edge("transform", "chat_transform")
    builder.add_edge("chat_transform", "validate")
    builder.add_edge("validate", "chat_validate")
    builder.add_edge("chat_validate", "lineage")
    builder.add_edge("lineage", "chat_lineage")
    builder.add_edge("chat_lineage", "report")
    builder.add_edge("report", "chat_report")
    builder.add_edge("chat_report", END)

    return builder.compile()

# Expose graph
interactive_pipeline_graph = build_full_interactive_graph()
