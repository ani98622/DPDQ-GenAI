import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0):
        """
        Initialize Differential Privacy with a privacy budget epsilon.
        """
        self.epsilon = epsilon
        self.anonymized_data = None
        self.original_data = None
        self.sensitivity = {}
        self.noise_added = {}
        self.info_loss = {}

    def fit_transform(self, df, quasi_identifiers, sensitive_attributes, numerical_cols=None):
        """
        Apply differential privacy to the sensitive attributes.
        """
        self.original_data = df.copy()
        df_anonymized = df.copy()
        self.total_records = len(df)

        for col in sensitive_attributes:
            if df[col].dtype in [np.int64, np.float64, float, int]:
                df_anonymized, noise_info = self._add_laplace_noise(df_anonymized, col)
                self.noise_added[col] = noise_info
            else:
                df_anonymized = self._apply_exponential_mechanism(df_anonymized, col)

        self.anonymized_data = df_anonymized
        self.info_loss = self._calculate_info_loss(numerical_cols)

        return df_anonymized

    def _add_laplace_noise(self, df, column):
        """
        Add Laplace noise to a numerical column.
        """
        data = pd.to_numeric(df[column], errors='coerce').values
        sensitivity = (np.nanmax(data) - np.nanmin(data)) / self.total_records
        self.sensitivity[column] = sensitivity
        scale = max(sensitivity / self.epsilon, 1e-10)
        noise = np.random.laplace(0, scale, size=len(data))
        df[column] = data + noise

        noise_info = {
            'mean': float(np.mean(noise)),
            'std': float(np.std(noise)),
            'min': float(np.min(noise)),
            'max': float(np.max(noise)),
            'scale': float(scale)
        }

        return df, noise_info

    def _apply_exponential_mechanism(self, df, column):
        """
        Apply exponential mechanism to a categorical column.
        """
        if column not in df.columns:
            print(f"Warning: Column {column} not found. Skipping.")
            return df

        value_counts = df[column].value_counts()
        if len(value_counts) == 0:
            print(f"Warning: Column {column} has no values. Skipping.")
            return df

        values = list(value_counts.index)
        freq = list(value_counts.values)
        utility_scores = np.array(freq) / np.sum(freq)
        probs = np.exp(self.epsilon * utility_scores / 2)
        probs = probs / np.sum(probs)

        new_values = np.random.choice(values, size=len(df), p=probs)
        df[column] = new_values

        return df

    def _calculate_info_loss(self, numerical_cols):
        """
        Calculate normalized mean absolute error for numerical columns.
        """
        info_loss = {}

        if numerical_cols is None or self.anonymized_data is None or self.original_data is None:
            return {'overall': 0}

        loss_values = []

        for col in numerical_cols:
            if col in self.anonymized_data.columns and col in self.original_data.columns:
                mae = np.mean(np.abs(self.original_data[col] - self.anonymized_data[col]))
                col_range = self.original_data[col].max() - self.original_data[col].min()
                normalized_mae = mae / col_range if col_range > 0 else 0
                info_loss[col] = normalized_mae
                loss_values.append(normalized_mae)

        info_loss['overall'] = sum(loss_values) / len(loss_values) if loss_values else 0
        return info_loss

    def _calculate_reidentification_risk(self):
        """
        Estimate re-identification risk based on epsilon.
        """
        privacy_guarantee = np.exp(self.epsilon)
        risk = min(1.0, privacy_guarantee / (1 + privacy_guarantee))
        return risk

    def get_stats(self):
        """
        Return differential privacy stats.
        """
        if self.anonymized_data is None:
            return "No anonymization performed yet."

        stats = {
            "epsilon": self.epsilon,
            "original_records": self.total_records,
            "remaining_records": len(self.anonymized_data),
            "suppressed_records": 0,
            "suppression_rate": 0,
            "avg_info_loss": self.info_loss.get('overall', 0),
            "noise_added": self.noise_added,
            "reidentification_risk": self._calculate_reidentification_risk()
        }

        return stats
