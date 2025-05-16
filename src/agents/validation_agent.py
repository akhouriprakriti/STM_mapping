import pandas as pd
import re

class ValidationAgent:
    def __init__(self, df, stm_plan, id_column="loan_number"):
        self.df = df
        self.stm_plan = stm_plan
        self.violations = []
        self.id_column = id_column

    def validate(self):
        for rule in self.stm_plan:
            col = rule["target_column"]
            expr = rule["validation"]

            if col not in self.df.columns:
                self.violations.append({
                    "column": col,
                    "rule": expr
                })
                continue

            try:
                failed_rows = pd.Series([False] * len(self.df))

                # Basic validations
                if expr == "not null":
                    failed_rows = self.df[col].isnull()

                elif expr == "valid_email":
                    failed_rows = ~self.df[col].astype(str).str.contains(r'^[^@]+@[^@]+\.[^@]+$', na=False)

                elif "length ==" in expr:
                    expected_len = int(expr.split("==")[1].strip())
                    failed_rows = self.df[col].astype(str).str.len() != expected_len

                elif "isdigit" in expr:
                    failed_rows = ~self.df[col].astype(str).str.isdigit()

                elif any(op in expr for op in [">=", "<=", ">", "<", "==", "!="]):
                    failed_rows = ~self.df.eval(f"{col} {expr}")

                elif "and" in expr or "or" in expr:
                    failed_rows = self.df.query(f"not ({col} {expr})").index

                # Collect violations
                if isinstance(failed_rows, pd.Index):
                    for idx in failed_rows:
                        self.violations.append({
                            "column": col,
                            "loan_number": self.df.at[idx, self.id_column],
                            "rule": expr,
                            "row": idx,
                            "value": self.df.at[idx, col]
                        })
                else:
                    for idx in self.df[failed_rows].index:
                        self.violations.append({
                            "column": col,
                            "loan_number": self.df.at[idx, self.id_column],
                            "rule": expr,
                            "row": idx,
                            "value": self.df.at[idx, col]
                        })

            except Exception as e:
                self.violations.append({
                    "column": col,
                    "rule": expr
                })

        return pd.DataFrame(self.violations)