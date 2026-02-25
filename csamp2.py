import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
RATING_LABELS = {
    1: "1-Poor",
    2: "2-Fair",
    3: "3-Average",
    4: "4-Good",
    5: "5-Excellent",
}

RATING_COLORS = {
    1: "#d73027",
    2: "#fc8d59",
    3: "#fee08b",
    4: "#91cf60",
    5: "#1a9850",
}

# ── Helper functions ───────────────────────────────────────────────────────────
def preprocess(data: pd.DataFrame, selected_features: list):
    features = data[selected_features]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled


def run_segmentation(data: pd.DataFrame, scaled, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(scaled)
    data = data.copy()
    data["Cluster"] = clusters + 1
    return data, clusters


def reduce_to_2d(scaled):
    if scaled.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(scaled)
    return scaled


def plot_before_after(data, scaled, clusters, selected_features):
    X2 = reduce_to_2d(scaled)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(True, linestyle="--", alpha=0.3, color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax1.scatter(X2[:, 0], X2[:, 1], c="#6c757d", s=40, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax1.set_title("Before K-Means")
    ax1.set_xlabel(selected_features[0])
    ax1.set_ylabel(selected_features[1] if len(selected_features) > 1 else "PC2")

    for cl in sorted(data["Cluster"].unique()):
        mask = (data["Cluster"] == cl).values
        ax2.scatter(
            X2[mask, 0], X2[mask, 1],
            c=RATING_COLORS[cl],
            label=RATING_LABELS.get(cl, f"Cluster {cl}"),
            s=50, alpha=0.95, edgecolor="k", linewidth=0.3,
        )
    ax2.set_title("After K-Means")
    ax2.set_xlabel(selected_features[0])
    ax2.set_ylabel(selected_features[1] if len(selected_features) > 1 else "PC2")
    legend = ax2.legend(facecolor="#1a1d23", labelcolor="white", edgecolor="#444")

    fig.suptitle("Customer Segmentation  ·  Before vs After K-Means", color="white", fontsize=14)
    plt.tight_layout()
    return fig


def plot_pie(data):
    counts = data["Cluster"].value_counts().sort_index()
    colors = [RATING_COLORS[cl] for cl in counts.index]
    labels = [RATING_LABELS.get(cl, f"Cluster {cl}") for cl in counts.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": "white"},
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontweight("bold")
    ax.set_title("Cluster Distribution", color="white", pad=12)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Customer Segmentation")
    st.markdown("Upload your CSV, pick features, and run K-Means clustering.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)

# ── Main ───────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown(
        """
        ## Welcome 👋
        Upload a **CSV file** in the sidebar to get started.

        ### What this app does
        1. **EDA** – Preview your data, stats, and missing values
        2. **Feature Selection** – Choose which columns to cluster on
        3. **K-Means Segmentation** – Group customers into clusters
        4. **Visualization** – Scatter plots (PCA) and cluster distribution pie chart
        """
    )
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

with st.sidebar:
    st.markdown("---")
    selected_features = st.multiselect(
        "Select Features for Clustering",
        options=numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
    )
    run_btn = st.button("▶ Run Segmentation", use_container_width=True, type="primary")

# ── EDA Tab ────────────────────────────────────────────────────────────────────
tab_eda, tab_seg, tab_viz = st.tabs(["📊 EDA", "🔢 Segmentation", "📈 Visualizations"])

with tab_eda:
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Missing Values per Column")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found ✅")
    else:
        st.dataframe(missing.rename("Missing Count"), use_container_width=True)

# ── Segmentation ───────────────────────────────────────────────────────────────
if run_btn:
    if len(selected_features) < 1:
        st.sidebar.error("Select at least one feature.")
        st.stop()

    with st.spinner("Running K-Means..."):
        scaled = preprocess(df, selected_features)
        result_df, clusters = run_segmentation(df, scaled, n_clusters)

    st.session_state["result_df"] = result_df
    st.session_state["scaled"] = scaled
    st.session_state["clusters"] = clusters
    st.session_state["selected_features"] = selected_features

if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    scaled = st.session_state["scaled"]
    clusters = st.session_state["clusters"]
    sel_feat = st.session_state["selected_features"]

    with tab_seg:
        st.subheader("Cluster Assignments")

        # Cluster summary
        summary = (
            result_df.groupby("Cluster")[sel_feat]
            .mean()
            .round(2)
            .reset_index()
        )
        summary["Rating Label"] = summary["Cluster"].map(RATING_LABELS)
        summary["Count"] = result_df["Cluster"].value_counts().sort_index().values
        st.dataframe(summary, use_container_width=True)

        st.subheader("Full Segmented Dataset")
        st.dataframe(result_df, use_container_width=True)

        # Download button
        csv_out = result_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Segmented CSV",
            data=csv_out,
            file_name="segmented_customers.csv",
            mime="text/csv",
        )

    with tab_viz:
        st.subheader("Scatter Plot (Before vs After K-Means)")
        fig1 = plot_before_after(result_df, scaled, clusters, sel_feat)
        st.pyplot(fig1)

        st.subheader("Cluster Distribution")
        col_pie, col_info = st.columns([1, 1])
        with col_pie:
            fig2 = plot_pie(result_df)
            st.pyplot(fig2)
        with col_info:
            st.markdown("### Cluster → Rating Mapping")
            for cl, label in RATING_LABELS.items():
                color = RATING_COLORS[cl]
                count = (result_df["Cluster"] == cl).sum()
                pct = count / len(result_df) * 100
                st.markdown(
                    f'<span style="color:{color}; font-size:18px">■</span> '
                    f'**{label}** — {count} customers ({pct:.1f}%)',
                    unsafe_allow_html=True,
                )
else:
    with tab_seg:
        st.info("Select features in the sidebar and click **▶ Run Segmentation**.")
    with tab_viz:
        st.info("Run segmentation first to see visualizations.")