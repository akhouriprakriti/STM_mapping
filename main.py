import streamlit as st
import pandas as pd
from workflows.langgraph_pipeline import interactive_pipeline_graph
from io import StringIO
import os
from dotenv import load_dotenv

# Load OpenAI API key from environment
load_dotenv()

st.set_page_config(page_title="🤖 STM Agentic Workflow", layout="wide")
st.title("🧠 Interactive STM Pipeline with LLM Assistant")

st.markdown("### Step 1: Upload STM Mapping and Raw Data")
stm_file = st.file_uploader("📄 Upload STM Mapping CSV", type="csv")
data_file = st.file_uploader("📊 Upload Raw Data CSV", type="csv")

if stm_file and data_file:
    raw_df = pd.read_csv(data_file)

    # Save STM file locally for consistency
    stm_path = "data-quality-agent/data/temp_uploaded_stm.csv"
    os.makedirs(os.path.dirname(stm_path), exist_ok=True)
    with open(stm_path, "w") as f:
        f.write(StringIO(stm_file.getvalue().decode("utf-8")).getvalue())

    st.success("✅ Files uploaded successfully!")

    # Trigger full pipeline
    if st.button("🚀 Run Full Pipeline"):
        with st.spinner("Running full STM pipeline with LLM guidance..."):
            initial_state = {
                "raw_df": raw_df,
                "stm_path": stm_path
            }
            result = interactive_pipeline_graph.invoke(initial_state)

        st.success("✅ Pipeline completed with LLM at every step!")

        # Final Summary
        # st.subheader("📊 Report Summary")
        st.markdown(result.get("report_summary", "—"))

        st.subheader("💬 LLM Feedback at Each Step")
        st.markdown("**After STM Parsing:**")
        st.markdown(result.get("llm_chat_stm", "—"))

        # Show Transformed Data
        st.subheader("📄 Transformed Data")
        st.dataframe(result["transformed_df"])

        st.markdown("**After Transformation:**")
        st.markdown(result.get("llm_chat_transform", "—"))

        # Show Validation Report
        st.subheader("🛠 Validation Report")
        st.dataframe(result["validation_report"])

        st.markdown("**After Validation:**")
        st.markdown(result.get("llm_chat_validate", "—"))
        
        # Show Lineage
        st.subheader("🔗 Technical Lineage Log")
        st.dataframe(result["lineage_log"])

        st.markdown("**After Lineage:**")
        st.markdown(result.get("llm_chat_lineage", "—"))

        st.markdown("**After Report Generation:**")
        st.markdown(result.get("llm_chat_report", "—"))

        # Before vs After for Affected Rows
        affected_rows = result["lineage_log"]["row"].unique().tolist()
        if affected_rows:
            st.subheader("🔍 Before vs After (Affected Rows Only)")
            left, right = st.columns(2)
            with left:
                st.markdown("**Original Data (Affected Rows)**")
                st.dataframe(raw_df.loc[affected_rows])
            with right:
                st.markdown("**Transformed Data (Affected Rows)**")
                st.dataframe(result["transformed_df"].loc[affected_rows])