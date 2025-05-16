from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd

def get_llm_chat_response(stage: str, state: dict) -> dict:
    """
    Interactive LLM agent that summarizes and answers based on the current pipeline stage.
    Adds an LLM response into the state as `llm_chat_<stage>`.
    """
    llm = ChatOpenAI(temperature=0.3, model="gpt-4.1-nano")

    if stage == "stm":
        stm_plan = pd.DataFrame(state.get("stm_plan", [])).to_markdown(index=False)
        prompt = f"""
You are reviewing a source-to-target mapping plan (STM).
Here is the mapping configuration provided by the user:

{stm_plan}

Provide insights about whether this STM is complete, logical, or missing any essential rules.
Allow the user to ask questions about specific mappings or make changes interactively.
"""

    elif stage == "transform":
        logs = state.get("transform_logs", [])
        change_summary = "\n".join(logs[:10]) if logs else "No transformation logs found."
        prompt = f"""
The transformation step has been executed.
Here are the first few transformation logs:

{change_summary}

Summarize what happened in this step. Offer to explain specific changes or address user questions.
"""

    elif stage == "validate":
        violations = state.get("validation_report")
        summary = f"{len(violations)} validation issues found across {violations['column'].nunique()} columns." if violations is not None and not violations.empty else "No violations found."
        prompt = f"""
The data was validated against defined rules.
{summary}

Would you like to explore specific issues, view top errors, or suggest additional checks?
"""

    elif stage == "lineage":
        lineage = state.get("lineage_log")
        if lineage is not None and not lineage.empty:
            sample = lineage.head(5).to_markdown(index=False)
            changes = f"Here are some of the changes:\n{sample}"
        else:
            changes = "No lineage differences detected."

        prompt = f"""
Lineage tracking is complete.
{changes}

Would you like to explore changes by row, column, or transformation step?
"""

    elif stage == "report":
        report = state.get("report_summary", "No report summary available.")
        prompt = f"""
Final report is generated. Can you give a detailed report of everything that happened in the pipeline?

Summary:
{report}

Would you like a breakdown of stats, export formats, or specific highlights?
"""

    else:
        prompt = "You are a pipeline assistant. What would you like to ask or review?"

    # Generate response
    response = llm([HumanMessage(content=prompt)]).content
    state[f"llm_chat_{stage}"] = response
    return state