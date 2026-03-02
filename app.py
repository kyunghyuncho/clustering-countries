import streamlit as st
import pandas as pd
import plotly.express as px
from data import load_data, preprocess_data, INDICATORS
from models import run_kmeans, run_gmm

st.set_page_config(page_title="Global Demographics & Economic Clustering", layout="wide")

st.title("Global Demographics & Economic Clustering")
st.markdown("This application clusters the world's countries based on demographic and economic statistics using K-Means and Gaussian Mixture Models (GMM) with a strictly enforced uniform prior.")

# Load raw data
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading indicator data: {e}")
    st.stop()


# -- Sidebar Configuration Panel --
st.sidebar.header("Configuration")

# Feature selection
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering",
    options=list(INDICATORS.values()),
    default=list(INDICATORS.values())
)

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

# Model selection
model_choice = st.sidebar.radio("Clustering Algorithm", ["K-Means", "GMM (Uniform Prior)"])

# Hyperparameters
k_clusters = st.sidebar.slider("Number of Clusters / Components (K)", min_value=2, max_value=15, value=5)

if model_choice == "GMM (Uniform Prior)":
    gmm_covariance = st.sidebar.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"], index=0)

# K-Nearest Neighbors Imputation Parameter
k_neighbors = st.sidebar.slider("Imputation k-Neighbors", min_value=1, max_value=15, value=5)


# -- Data Processing --
# Preprocess the data based on selected features
df_scaled, df_imputed, scaler = preprocess_data(df_raw, selected_features=selected_features, k_neighbors=k_neighbors)


# -- Modeling --
if model_choice == "K-Means":
    model, labels, sil_score = run_kmeans(df_scaled.values, n_clusters=k_clusters)
    aic = bic = None
else:
    model, labels, sil_score, aic, bic = run_gmm(df_scaled.values, n_components=k_clusters, covariance_type=gmm_covariance)

# Add assignments to dataframe for visualization
df_results = df_imputed.copy()
df_results['Cluster'] = [str(lbl) for lbl in labels]
df_results = df_results.reset_index()


# -- Main Dashboard --

# Metrics Row
st.header("Clustering Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    if sil_score is not None:
        st.metric("Silhouette Score (Higher is better)", f"{sil_score:.3f}")
    else:
        st.metric("Silhouette Score", "N/A")
        
with col2:
    if aic is not None:
        st.metric("AIC (Lower is better)", f"{aic:.1f}")
    else:
        st.metric("AIC", "N/A (GMM Only)")
        
with col3:
    if bic is not None:
        st.metric("BIC (Lower is better)", f"{bic:.1f}")
    else:
        st.metric("BIC", "N/A (GMM Only)")

st.divider()

# Geospatial Visualization
st.header("Global Cluster Distribution")

# Add Country Name to hover data
hover_data_columns = ['Country Name'] + selected_features

fig = px.choropleth(
    df_results,
    locations="economy", # ISO-3 Codes
    color="Cluster",
    hover_name="Country Name",
    hover_data=selected_features,
    color_discrete_sequence=px.colors.qualitative.Plotly,
    projection="natural earth",
    title=f"{model_choice} Clustering (K={k_clusters})"
)
fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)


# Data & Centroid Explorer
st.header("Centroid Explorer")
with st.expander("View Cluster Centroids (Unscaled / Mean Values)"):
    # Group by cluster and calculate mean of original (imputed, but not logged/scaled) values
    
    # We log-transformed some features in imputation step, let's look at the raw un-transformed means per cluster
    # To do this accurately, let's merge the labels back into df_raw and group by it
    df_raw_with_labels = df_raw.copy()
    
    # Standardize index so we only map matching ones
    common_idx = df_raw_with_labels.index.intersection(df_scaled.index)
    df_raw_with_labels.loc[common_idx, 'Cluster'] = labels
    
    # Calculate means
    centroids = df_raw_with_labels.groupby('Cluster')[selected_features].mean()
    
    st.dataframe(centroids.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))
    st.markdown("*(Green indicates the highest value across clusters, Red indicates the lowest)*")
