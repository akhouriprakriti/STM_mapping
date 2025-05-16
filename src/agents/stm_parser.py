import pandas as pd

class STMParser:
    def __init__(self, stm_path):
        self.stm_path = stm_path
        self.mapping_plan = []

    def parse(self):
        """
        Reads the STM CSV file and returns a structured transformation plan.
        Each entry in the plan is a dictionary with source, target, transformation, validation, and type.
        """
        df = pd.read_csv(self.stm_path)

        for _, row in df.iterrows():
            self.mapping_plan.append({
                "source_column": row["Source Column"],
                "target_column": row["Target Column"],
                "transformation": row["Transformation"],
                "validation": row["Validation"],
                "data_type": row["Data Type"]
            })

        return self.mapping_plan