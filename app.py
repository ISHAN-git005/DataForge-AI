import streamlit as st
import pandas as pd
import io
from core.cleaner import AutoDataCleaner
from core.visualizer import VisualProfiler

st.set_page_config(
    page_title="DataForge AI | Data Cleaner & Profiler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .download-box {
            background-color: #1E293B;
            border: 2px solid #10B981;
            padding: 1.2rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DataForge AI: Automated Data Cleaning & EDA Engine")
st.caption("Upload a dirty dataset to instantly clean, standardize, and download the refined data.")

# Sidebar Configuration
st.sidebar.header("🛠️ Cleaning Configuration")
missing_strategy = st.sidebar.selectbox("Numeric Imputation Strategy", ["median", "mean"], index=0)
outlier_threshold = st.sidebar.slider("Outlier IQR Threshold", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
clean_text = st.sidebar.checkbox("Standardize Strings & Column Names", value=True)
handle_outliers = st.sidebar.checkbox("Cap Outliers (Winsorize)", value=True)
handle_missing = st.sidebar.checkbox("Handle Missing Values", value=True)
drop_duplicates = st.sidebar.checkbox("Remove Duplicate Rows", value=True)

uploaded_file = st.sidebar.file_uploader("Upload Dirty Dataset (CSV, Excel, Parquet)", type=["csv", "xlsx", "parquet"])

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    return None

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    
    cleaner = AutoDataCleaner()
    with st.spinner("⚡ Processing & Auto-Cleaning Dataset..."):
        df_clean, audit_log = cleaner.run_pipeline(
            df_raw=df_raw,
            handle_missing=handle_missing,
            missing_strategy=missing_strategy,
            handle_outliers=handle_outliers,
            outlier_threshold=outlier_threshold,
            clean_text=clean_text,
            drop_duplicates=drop_duplicates
        )
        metrics = VisualProfiler.get_summary_metrics(df_raw, df_clean)

    # ==========================
    # PROMINENT DOWNLOAD SECTION
    # ==========================
    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.subheader("✅ Dataset Successfully Cleaned!")
    st.write(f"Processed **{metrics['initial_shape'][0]} rows** → Resolved **{abs(metrics['final_missing'] - metrics['initial_missing'])} missing values** and removed **{metrics['initial_duplicates'] - metrics['final_duplicates']} duplicate rows**.")

    down_col1, down_col2, down_col3 = st.columns(3)

    # 1. CSV Download
    csv_buffer = io.StringIO()
    df_clean.to_csv(csv_buffer, index=False)
    with down_col1:
        st.download_button(
            label="📥 Download Clean CSV",
            data=csv_buffer.getvalue(),
            file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. Excel Download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_clean.to_excel(writer, index=False, sheet_name='CleanedData')
    with down_col2:
        st.download_button(
            label="📊 Download Clean Excel (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # 3. JSON Download
    json_buffer = df_clean.to_json(orient="records", indent=2)
    with down_col3:
        st.download_button(
            label="🧾 Download Clean JSON",
            data=json_buffer,
            file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================
    # METRICS SUMMARY
    # ==========================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Shape", f"{metrics['initial_shape'][0]} × {metrics['initial_shape'][1]}")
    with col2:
        st.metric("Clean Shape", f"{metrics['final_shape'][0]} × {metrics['final_shape'][1]}")
    with col3:
        delta_missing = metrics['final_missing'] - metrics['initial_missing']
        st.metric("Missing Values", f"{metrics['final_missing']}", delta=f"{delta_missing} resolved", delta_color="inverse")
    with col4:
        delta_dup = metrics['final_duplicates'] - metrics['initial_duplicates']
        st.metric("Duplicates Cleared", f"{metrics['initial_duplicates']}", delta=f"{delta_dup}", delta_color="inverse")

    st.markdown("---")

    # ==========================
    # TABS FOR INSPECTION
    # ==========================
    tab_inspect, tab_eda, tab_audit = st.tabs([
        "👀 Side-by-Side Data Preview", 
        "📊 Dual-State EDA Visuals", 
        "📜 Audit & Changelog"
    ])

    with tab_inspect:
        st.subheader("Raw (Dirty) vs Cleaned Preview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Raw (First 50 Rows)**")
            st.dataframe(df_raw.head(50), use_container_width=True)
        with c2:
            st.markdown("**Cleaned & Standardized (First 50 Rows)**")
            st.dataframe(df_clean.head(50), use_container_width=True)

    with tab_eda:
        st.subheader("1. Missingness Resolution")
        st.plotly_chart(VisualProfiler.plot_missingness_comparison(df_raw, df_clean), use_container_width=True)

        st.subheader("2. Distribution Shift Profiler")
        numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select numeric feature to inspect shift:", numeric_cols)
            st.plotly_chart(VisualProfiler.plot_distribution_shift(df_clean, df_clean, selected_col), use_container_width=True)
        else:
            st.info("No numeric columns found for distribution plots.")

        st.subheader("3. Cleaned Feature Correlation Heatmap")
        corr_fig = VisualProfiler.plot_correlation_heatmap(df_clean)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)

    with tab_audit:
        st.subheader("Transformation Changelog")
        st.caption("Every transformation, imputation, and outlier treatment is recorded below.")
        if not audit_log.empty:
            st.dataframe(audit_log, use_container_width=True, hide_index=True)
        else:
            st.success("No anomalies detected; dataset was already clean.")

else:
    st.info("👈 Upload a dataset in the sidebar to clean and download it immediately.")

    st.write("---")
    st.subheader("🧪 Don't have a dataset ready? Generate a dirty one to test:")
    
    if st.button("Generate & Download Sample Dirty Dataset"):
        import numpy as np
        data = {
            "User ID": [101, 102, 103, 104, 105, 105, 106, 107],
            " Age ": [25, np.nan, 29, 32, 190, 190, 41, 36],
            "Income ($)": [50000, 62000, np.nan, 85000, 92000, 92000, 1000000, 58000],
            "Join Date": ["2023-01-15", "15/02/2023", "2023-03-20", "invalid_date", "2023-05-12", "2023-05-12", "2023-07-01", "2023-08-11"],
            "City": [" New York", "new york", "LONDON", "london", "Paris", "Paris", "Tokio?", np.nan]
        }
        sample_df = pd.DataFrame(data)
        csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Click to Download Sample Dirty CSV", 
            csv_bytes, 
            "dirty_sample.csv", 
            "text/csv"
        )