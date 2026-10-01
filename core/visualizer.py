import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class VisualProfiler:
    @staticmethod
    def get_summary_metrics(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
        return {
            "initial_shape": df_raw.shape,
            "final_shape": df_clean.shape,
            "initial_missing": int(df_raw.isna().sum().sum()),
            "final_missing": int(df_clean.isna().sum().sum()),
            "initial_duplicates": int(df_raw.duplicated().sum()),
            "final_duplicates": int(df_clean.duplicated().sum()),
        }

    @staticmethod
    def plot_missingness_comparison(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
        raw_missing = df_raw.isna().sum().reset_index()
        raw_missing.columns = ['Feature', 'Missing Count']
        raw_missing['Dataset'] = 'Before'

        clean_missing = df_clean.isna().sum().reset_index()
        clean_missing.columns = ['Feature', 'Missing Count']
        clean_missing['Dataset'] = 'After'

        combined = pd.concat([raw_missing, clean_missing])
        fig = px.bar(
            combined, 
            x='Feature', 
            y='Missing Count', 
            color='Dataset', 
            barmode='group',
            title="Dual-State Missingness Audit (Before vs. After)",
            color_discrete_map={"Before": "#EF4444", "After": "#10B981"}
        )
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def plot_distribution_shift(df_raw: pd.DataFrame, df_clean: pd.DataFrame, column: str):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Pre-Cleaning: {column}", f"Post-Cleaning: {column}"))

        fig.add_trace(
            go.Histogram(x=df_raw[column].dropna(), name="Pre-Clean", marker_color="#F59E0B", opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=df_clean[column].dropna(), name="Post-Clean", marker_color="#3B82F6", opacity=0.8),
            row=1, col=2
        )

        fig.update_layout(
            template="plotly_dark", 
            title_text=f"Feature Distribution Shift Analysis: {column}",
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, title="Correlation Matrix"):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        corr = numeric_df.corr().round(2)
        fig = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto", 
            title=title,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
        return fig