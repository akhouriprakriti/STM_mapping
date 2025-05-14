import pandas as pd

class LineageAgent:
    def __init__(self, original_df, transformed_df, stm_plan):
        self.original_df = original_df
        self.transformed_df = transformed_df
        self.stm_plan = stm_plan
        self.lineage_log = []

    def trace(self):
        for rule in self.stm_plan:
            src_col = rule["source_column"]
            tgt_col = rule["target_column"]
            transform = rule["transformation"]

            # Ensure both columns exist before comparing
            if src_col not in self.original_df.columns or tgt_col not in self.transformed_df.columns:
                continue

            for idx in self.original_df.index:
                original_value = self.original_df.at[idx, src_col] if idx in self.original_df.index else None
                transformed_value = self.transformed_df.at[idx, tgt_col] if idx in self.transformed_df.index else None

                # Skip if both are NaN or exactly the same
                if pd.isnull(original_value) and pd.isnull(transformed_value):
                    continue
                if str(original_value) == str(transformed_value):
                    continue

                self.lineage_log.append({
                    "row": idx,
                    "source_column": src_col,
                    "target_column": tgt_col,
                    "original_value": original_value,
                    "transformed_value": transformed_value,
                    "transformation_applied": transform
                })

        return pd.DataFrame(self.lineage_log)
