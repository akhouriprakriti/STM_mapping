import pandas as pd
from datetime import datetime
import os

class ReportAgent:
    def __init__(self, transformed_df, validation_df, lineage_df):
        self.transformed_df = transformed_df
        self.validation_df = validation_df
        self.lineage_df = lineage_df
        self.summary = ""

    def generate_summary(self):
        self.summary = "## 📊 Data Quality Report\n"
        self.summary += f"🕒 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Transformation Summary
        self.summary += "### ✅ Transformation Summary\n"
        self.summary += f"- Records Transformed: {len(self.transformed_df)}\n\n"

        # Validation Summary
        self.summary += "### 🔍 Validation Summary\n"
        if self.validation_df.empty:
            self.summary += "- No validation issues found.\n\n"
        else:
            self.summary += f"- Issues Found: {len(self.validation_df)}\n"
            self.summary += f"- Affected Columns: {self.validation_df['column'].nunique()}\n\n"

        # Lineage Summary
        self.summary += "### 🔁 Lineage Summary\n"
        if self.lineage_df.empty:
            self.summary += "- No transformations changed the values.\n"
        else:
            self.summary += f"- Total Changes: {len(self.lineage_df)}\n"
            self.summary += f"- Changed Columns: {self.lineage_df['target_column'].nunique()}\n\n"

        return self.summary

    def export_csvs(self, output_dir="data-quality-agent/data/report"):
        os.makedirs(output_dir, exist_ok=True)

        transformed_path = os.path.join(output_dir, "transformed_data.csv")
        validation_path = os.path.join(output_dir, "validation_report.csv")
        lineage_path = os.path.join(output_dir, "lineage_log.csv")

        self.transformed_df.to_csv(transformed_path, index=False)
        self.validation_df.to_csv(validation_path, index=False)
        self.lineage_df.to_csv(lineage_path, index=False)

        return {
            "transformed_data_path": transformed_path,
            "validation_report_path": validation_path,
            "lineage_log_path": lineage_path
        }
