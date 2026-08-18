"""Streamlit UI for customer segmentation."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.clustering import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    ID_COLUMN,
    run_segmentation,
)

st.set_page_config(page_title="Customer Segmentation", layout="wide")


def main() -> None:
    st.title("Customer Segmentation")
    st.caption(
        "K-Means clustering of retail customers by visits, items bought, and spending."
    )

    with st.sidebar:
        st.header("Settings")
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
        uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
        st.caption(
            f"Default local file: `data/{DEFAULT_DATA_PATH.name}` (not stored on GitHub). "
            f"Required columns: {ID_COLUMN}, {', '.join(FEATURE_COLUMNS)}."
        )

    try:
        if uploaded_file is not None:
            df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
            result = run_segmentation(n_clusters=n_clusters, df=df)
            source_label = uploaded_file.name
        else:
            result = run_segmentation(path=DEFAULT_DATA_PATH, n_clusters=n_clusters)
            source_label = DEFAULT_DATA_PATH.name
    except FileNotFoundError:
        st.info(
            "No dataset is bundled with the app. Put `visitItemSpend.csv` in the "
            "`data/` folder, or upload a CSV in the sidebar."
        )
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", result.n_customers)
    col2.metric("Clusters", n_clusters)
    col3.metric("Data source", source_label)

    st.subheader("Cluster sizes")
    st.dataframe(result.cluster_sizes, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Centroids (Min-Max scaled)")
        st.dataframe(result.scaled_centroids, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Centroids (original units)")
        st.dataframe(
            result.original_centroids.round(2),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("3D cluster view")
    fig = px.scatter_3d(
        result.labeled_df,
        x=FEATURE_COLUMNS[0],
        y=FEATURE_COLUMNS[1],
        z=FEATURE_COLUMNS[2],
        color="Segment",
        hover_data=[ID_COLUMN, "Cluster"],
        title=f"K-Means clustering with {n_clusters} clusters",
    )
    fig.update_traces(marker={"size": 4})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=620)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Labeled customers")
    st.dataframe(result.labeled_df, use_container_width=True, hide_index=True, height=320)
    csv_bytes = result.labeled_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download labeled CSV",
        data=csv_bytes,
        file_name="customer_segments.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
