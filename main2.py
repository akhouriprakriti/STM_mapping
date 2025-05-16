import streamlit as st
import pandas as pd
import os
from io import StringIO
from dotenv import load_dotenv

from src.agents.stm_parser import STMParser
from src.agents.transformation_agent import TransformAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.lineage_agent import LineageAgent
from src.agents.report_agent import ReportAgent
from src.agents.llm_chat_node import  get_llm_chat_response # your updated LLM node

load_dotenv()

st.set_page_config(page_title="🧠 STM Agentic Assistant", layout="wide")
st.title("🤖 Talk to Your Data Pipeline")

# Upload data
stm_file = st.file_uploader("📄 Upload STM Mapping CSV", type="csv")
data_file = st.file_uploader("📊 Upload Raw Data CSV", type="csv")

if not stm_file or not data_file:
    st.warning("Please upload both files to begin.")
    st.stop()

raw_df = pd.read_csv(data_file)
stm_path = "temp_uploaded_stm.csv"
with open(stm_path, "w") as f:
    f.write(StringIO(stm_file.getvalue().decode("utf-8")).getvalue())

# Session state setup
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = None
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {"raw_df": raw_df, "stm_path": stm_path}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

state = st.session_state.chat_state
stage = st.session_state.chat_stage

# Main chat loop
prompt = st.chat_input("💬 Type a command like 'stm processing', 'transform', 'validate', or 'report'")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.chat_history.append(("user", prompt))

    # Trigger STM step
    if "stm" in prompt.lower():
        parser = STMParser(state["stm_path"])
        stm_plan = parser.parse()
        state["stm_plan"] = stm_plan
        state = get_llm_chat_response("stm", state)
        st.session_state.chat_stage = "stm"
        st.session_state.chat_state = state
        st.session_state.chat_history.append(("assistant", state["llm_chat_stm"]))
        st.chat_message("assistant").write(state["llm_chat_stm"])

    # Trigger Transform step
    elif "transform" in prompt.lower():
        transformer = TransformAgent(state["raw_df"], state["stm_plan"])
        transformed_df, logs = transformer.apply()
        state["transformed_df"] = transformed_df
        state["transform_logs"] = logs
        state = get_llm_chat_response("transform", state)
        st.session_state.chat_stage = "transform"
        st.session_state.chat_state = state
        st.session_state.chat_history.append(("assistant", state["llm_chat_transform"]))
        st.chat_message("assistant").write(state["llm_chat_transform"])
        st.dataframe(transformed_df)

    # Trigger Validation step
    elif "validate" in prompt.lower():
        validator = ValidationAgent(state["transformed_df"], state["stm_plan"])
        validation_report = validator.validate()
        state["validation_report"] = validation_report
        state = get_llm_chat_response("validate", state)
        st.session_state.chat_stage = "validate"
        st.session_state.chat_state = state
        st.session_state.chat_history.append(("assistant", state["llm_chat_validate"]))
        st.chat_message("assistant").write(state["llm_chat_validate"])
        st.dataframe(validation_report)

    # Trigger Report step
    elif "report" in prompt.lower():
        reporter = ReportAgent(state["transformed_df"], state["validation_report"], state["lineage_log"])
        summary = reporter.generate_summary()
        state["report_summary"] = summary
        state = get_llm_chat_response("report", state)
        st.session_state.chat_stage = "report"
        st.session_state.chat_state = state
        st.session_state.chat_history.append(("assistant", state["llm_chat_report"]))
        st.chat_message("assistant").write(state["llm_chat_report"])
        st.markdown(summary)

    # Unknown input fallback
    else:
        st.chat_message("assistant").write("🤖 I didn’t understand that. Try typing: `stm processing`, `transform`, `validate`, or `report`.")

# Optional: Display full chat history
with st.expander("🕘 Show full chat history"):
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role.upper()}**: {msg}")
