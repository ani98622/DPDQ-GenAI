import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(0)

# ------------------ Generalization Helper ------------------ #
def generalize_by_range(value, value_ranges):
    for label, (lower, upper) in value_ranges.items():
        if lower <= value <= upper:
            return label
    return "Other"
# ----------------------------------------------------------- #

# ------------------ Dataset Loader ------------------ #
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
# ---------------------------------------------------- #

# ------------------ EDA ------------------ #
def explore_data(df):
    print("Data types:")
    print(df.dtypes)

    print("\nBasic statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFeature distributions:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    print(f"Categorical columns: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"\n{col} value counts:")
        print(df[col].value_counts().head())

    print(f"\nNumerical columns: {len(numerical_cols)}")
    return categorical_cols, numerical_cols
# ---------------------------------------- #

# ------------------ K-Anonymity Implementation ------------------ #
class KAnonymizer:
    def __init__(self, k=2):
        self.k = k
        self.anonymized_data = None
        self.original_data = None
        self.suppressed_records = 0
        self.generalized_columns = set()
        self.equivalence_classes = None

    def fit_transform(self, df, quasi_identifiers, generalization_rules=None):
        self.original_data = df.copy()
        df_anonymized = df.copy()

        if generalization_rules:
            for col, rule in generalization_rules.items():
                if col in df_anonymized.columns:
                    if isinstance(rule, dict):
                        if all(isinstance(v, tuple) for v in rule.values()):
                            df_anonymized[col] = df_anonymized[col].apply(lambda x: generalize_by_range(x, rule))
                        else:
                            df_anonymized[col] = df_anonymized[col].map(rule).fillna("Other")
                        self.generalized_columns.add(col)

        self.equivalence_classes = df_anonymized.groupby(quasi_identifiers)
        equivalence_class_sizes = self.equivalence_classes.size()

        violating_classes = equivalence_class_sizes[equivalence_class_sizes < self.k].index.tolist()

        if violating_classes:
            mask = df_anonymized[quasi_identifiers].apply(tuple, axis=1).isin([tuple(x) for x in violating_classes])
            self.suppressed_records = mask.sum()
            print(f"Suppressing {self.suppressed_records} records to satisfy {self.k}-anonymity")
            df_anonymized = df_anonymized[~mask]

        self.anonymized_data = df_anonymized
        self.total_records = len(df)
        self.remaining_records = len(df_anonymized)
        self.equivalence_class_count = len(df_anonymized.groupby(quasi_identifiers))
        self.reidentification_risk = self.calculate_reidentification_risk(df_anonymized, quasi_identifiers)

        return df_anonymized

    def calculate_reidentification_risk(self, df, quasi_identifiers):
        if df.empty:
            return 0
        eq_class_sizes = df.groupby(quasi_identifiers).size()
        record_eq_class_sizes = df[quasi_identifiers].apply(lambda x: eq_class_sizes[tuple(x)], axis=1)
        record_risks = 1 / record_eq_class_sizes
        avg_risk = record_risks.mean()
        return avg_risk

    def get_stats(self):
        if self.anonymized_data is None:
            return "No anonymization performed yet."

        stats = {
            "original_records": self.total_records,
            "remaining_records": self.remaining_records,
            "suppressed_records": self.suppressed_records,
            "suppression_rate": round(self.suppressed_records / self.total_records * 100, 2),
            "equivalence_class_count": self.equivalence_class_count,
            "avg_equivalence_class_size": round(self.remaining_records / self.equivalence_class_count, 2),
            "generalized_columns": list(self.generalized_columns),
            "avg_reidentification_risk": round(self.reidentification_risk, 6),
            "max_reidentification_probability": round(1 / self.k, 6)
        }
        return stats

# ------------------ L-Diversity Implementation ------------------ #
class LDiversifier:
    def __init__(self, l=2):
        self.l = l
        self.anonymized_data = None
        self.k_anonymized_data = None
        self.suppressed_records = 0
        self.total_records = 0
        self.remaining_records = 0
        self.equivalence_class_count = 0
        self.reidentification_risk = 0

    def fit_transform(self, k_anonymized_df, quasi_identifiers, sensitive_attributes):
        self.k_anonymized_data = k_anonymized_df.copy()
        df_anonymized = self.k_anonymized_data.copy()
        self.total_records = len(df_anonymized)

        # Group by quasi-identifiers
        groups = df_anonymized.groupby(quasi_identifiers)
        violating_indices = []

        # Check each group for L-diversity violation
        for _, group in groups:
            violates = False
            for attr in sensitive_attributes:
                if group[attr].nunique() < self.l:
                    violates = True
                    break
            if violates:
                violating_indices.extend(group.index.tolist())

        # Suppress violating records
        if violating_indices:
            print(f"Suppressing {len(violating_indices)} records to satisfy {self.l}-diversity")
            df_anonymized.drop(index=violating_indices, inplace=True)
            self.suppressed_records = len(violating_indices)

        self.anonymized_data = df_anonymized
        self.remaining_records = len(df_anonymized)
        self.equivalence_class_count = df_anonymized.groupby(quasi_identifiers).ngroups
        self.reidentification_risk = self._calculate_reidentification_risk(df_anonymized, quasi_identifiers)

        return df_anonymized

    def _calculate_reidentification_risk(self, df, quasi_identifiers):
        if df.empty:
            return 0.0

        eq_class_sizes = df.groupby(quasi_identifiers).size()
        record_sizes = df[quasi_identifiers].apply(lambda x: eq_class_sizes[tuple(x)], axis=1)
        risks = 1 / record_sizes
        return risks.mean()

    def get_stats(self):
        if self.anonymized_data is None:
            return "No anonymization performed yet."

        return {
            "k_anonymized_records": self.total_records,
            "remaining_records": self.remaining_records,
            "additional_suppressed_records": self.suppressed_records,
            "suppression_rate (%)": round(self.suppressed_records / self.total_records * 100, 2),
            "equivalence_class_count": self.equivalence_class_count,
            "avg_equivalence_class_size": round(self.remaining_records / self.equivalence_class_count, 2)
            if self.equivalence_class_count > 0 else 0,
            "avg_reidentification_risk": round(self.reidentification_risk, 6),
            "max_reidentification_probability": round(1 / self.l, 6)
        }
