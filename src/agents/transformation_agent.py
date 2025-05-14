import pandas as pd
from datetime import datetime

class TransformAgent:
    def __init__(self, df, stm_plan):
        self.df = df.copy()
        self.stm_plan = stm_plan
        self.transformed_df = pd.DataFrame()
        self.logs = []

    def apply(self):
        for rule in self.stm_plan:
            src = rule["source_column"]
            tgt = rule["target_column"]
            transform = rule["transformation"]

            # Check if source column exists
            if src not in self.df.columns:
                self.logs.append(f"⚠️ Column '{src}' missing in raw data.")
                continue

            series = self.df[src]

            # Transformation logic
            if transform == "pad_10_digits":
                self.transformed_df[tgt] = series.astype(str).str.zfill(10)

            elif transform == "calculate_age":
                today = datetime.today()
                self.transformed_df[tgt] = pd.to_datetime(series, errors="coerce").apply(
                    lambda dob: today.year - dob.year if pd.notnull(dob) else None
                )

            elif transform == "round_2_decimals":
                self.transformed_df[tgt] = pd.to_numeric(series, errors="coerce").round(2)

            elif transform.startswith("format:"):
                fmt = transform.split("format:")[1].strip()
                self.transformed_df[tgt] = pd.to_datetime(series, errors="coerce", dayfirst=True).dt.strftime(fmt)

            elif transform == "trim_lower":
                self.transformed_df[tgt] = series.astype(str).str.strip().str.lower()

            elif transform.startswith("clip:"):
                try:
                    bounds = transform.split(":")[1].split("-")
                    low, high = float(bounds[0]), float(bounds[1])
                    self.transformed_df[tgt] = pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high)
                except Exception:
                    self.logs.append(f"⚠️ Clip bounds parsing failed for '{tgt}'")
                    self.transformed_df[tgt] = series  # fallback

            elif transform == "fill: unknown":
                self.transformed_df[tgt] = series.fillna("unknown")

            elif transform == "divide_by_12":
                self.transformed_df[tgt] = pd.to_numeric(series, errors="coerce") // 12

            else:
                self.transformed_df[tgt] = series
                self.logs.append(f"⚠️ Unknown transformation '{transform}' for column '{src}' — passed through.")

            self.logs.append(f"✅ Applied '{transform}' → {src} → {tgt}")

        return self.transformed_df, self.logs
