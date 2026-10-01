import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import re

class AutoDataCleaner:
    def __init__(self):
        self.audit_log = []
        self.stats = {}

    def log_event(self, stage: str, column: str, message: str, impact_count: int = 0):
        self.audit_log.append({
            "Stage": stage,
            "Column": column,
            "Action Taken": message,
            "Records Impacted": impact_count
        })

    def run_pipeline(self, df_raw: pd.DataFrame, 
                     handle_missing=True, 
                     missing_strategy="median", 
                     handle_outliers=True, 
                     outlier_threshold=1.5,
                     clean_text=True, 
                     drop_duplicates=True):
        
        df = df_raw.copy()
        self.audit_log = []
        
        # 1. Deduplication
        if drop_duplicates:
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                self.log_event("Deduplication", "All", f"Removed {dup_count} exact duplicate rows", dup_count)

        # 2. Inconsistent String & Column Standardizing
        if clean_text:
            # Clean column names: lowercase, replace spaces/symbols with underscores
            new_cols = [re.sub(r'[^a-zA-Z0-9_]', '', col.strip().lower().replace(" ", "_")) for col in df.columns]
            col_rename_map = dict(zip(df.columns, new_cols))
            df.rename(columns=col_rename_map, inplace=True)
            self.log_event("Schema Normalization", "Header", "Standardized column headers to snake_case", len(df.columns))

            for col in df.select_dtypes(include=['object', 'string']).columns:
                # Strip leading/trailing whitespaces and lower casing inconsistencies
                initial_series = df[col].astype(str)
                cleaned_series = df[col].astype(str).str.strip()
                cleaned_series = cleaned_series.replace({
                    'nan': np.nan, 'NaN': np.nan, 'null': np.nan, 'NULL': np.nan, 'None': np.nan, '?': np.nan, '-': np.nan
                })
                impacted = (initial_series != cleaned_series).sum()
                if impacted > 0:
                    df[col] = cleaned_series
                    self.log_event("Text Formatting", col, "Trimmed whitespace and normalized null tokens", impacted)

        # 3. Dynamic Type Inference (Detect potential dates and numerics in object columns)
        for col in df.select_dtypes(include=['object']).columns:
            # Try numeric conversion
            numeric_conv = pd.to_numeric(df[col], errors='coerce')
            if numeric_conv.notna().sum() > (0.85 * len(df[col])):
                df[col] = numeric_conv
                self.log_event("Type Inference", col, "Converted high-density numeric string to Float64", len(df))
                continue

            # Try datetime conversion
            try:
                datetime_conv = pd.to_datetime(df[col], errors='coerce', format='mixed')
                if datetime_conv.notna().sum() > (0.85 * len(df[col])):
                    df[col] = datetime_conv
                    self.log_event("Type Inference", col, "Unified dates to ISO standard datetime", len(df))
            except Exception:
                pass

        # 4. Outlier Handling (Winsorization via IQR)
        if handle_outliers:
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - (outlier_threshold * iqr)
                    upper_bound = q75 + (outlier_threshold * iqr)
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        df[col] = np.clip(df[col], lower_bound, upper_bound)
                        self.log_event("Outlier Capping", col, f"Capped {outliers} values beyond {outlier_threshold}x IQR", outliers)

        # 5. Missing Value Imputation
        if handle_missing:
            # Numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                nans = df[col].isna().sum()
                if nans > 0:
                    fill_val = df[col].median() if missing_strategy == "median" else df[col].mean()
                    df[col] = df[col].fillna(fill_val)
                    self.log_event("Imputation", col, f"Replaced {nans} missing entries with {missing_strategy} ({fill_val:.3f})", nans)

            # Categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
            for col in cat_cols:
                nans = df[col].isna().sum()
                if nans > 0:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    self.log_event("Imputation", col, f"Replaced {nans} missing strings with mode ('{mode_val}')", nans)

        audit_df = pd.DataFrame(self.audit_log)
        return df, audit_df