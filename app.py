import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data import load_data, preprocess_data, INDICATORS
from models import run_kmeans, run_gmm, align_clusters

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

# Time slider
min_year, max_year = int(df_raw['year'].min()), int(df_raw['year'].max())
selected_year = st.sidebar.slider("Active Year", min_value=min_year, max_value=max_year, value=max_year)

# Model selection
model_choice = st.sidebar.radio("Clustering Algorithm", ["K-Means", "GMM (Uniform Prior)"])

# Hyperparameters
k_clusters = st.sidebar.slider("Number of Clusters / Components (K)", min_value=2, max_value=15, value=5)

if model_choice == "GMM (Uniform Prior)":
    gmm_covariance = st.sidebar.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"], index=0)
else:
    gmm_covariance = None

# K-Nearest Neighbors Imputation Parameter
k_neighbors = st.sidebar.slider("Imputation k-Neighbors", min_value=1, max_value=15, value=5)


# -- Data Processing --
# Preprocess the data based on selected features
df_scaled_all, df_imputed_all = preprocess_data(df_raw, selected_features=selected_features, k_neighbors=k_neighbors)


# -- Modeling (Longitudinal Alignment) --
# To ensure cluster label stability (e.g. Cluster 1 remains conceptually similar over time),
# we process the clustering chronologically from min_year to selected_year and align centroids.

previous_centroids = None
final_labels = None
final_sil_score = None
final_aic = final_bic = None

for yr in range(min_year, selected_year + 1):
    df_yr_scaled = df_scaled_all[df_scaled_all['year'] == yr].sort_values('Country Name')
    X = df_yr_scaled[selected_features].values
    
    if len(X) == 0:
        continue
        
    if model_choice == "K-Means":
        model, labels, sil_score = run_kmeans(X, n_clusters=k_clusters)
        centroids = model.cluster_centers_
        aic = bic = None
    else:
        model, labels, sil_score, aic, bic = run_gmm(X, n_components=k_clusters, covariance_type=gmm_covariance)
        centroids = model.means_
        
    if previous_centroids is not None:
        # Align new centroids to the previous year's centroids using the Hungarian Algorithm
        mapping = align_clusters(previous_centroids, centroids)
        
        # Apply mapping to current year labels
        labels = np.array([mapping[l] for l in labels])
        
        # Reorder current centroids to match the old indexing structure for the next iteration
        ordered_centroids = np.zeros_like(centroids)
        for new_lbl, old_lbl in mapping.items():
            ordered_centroids[old_lbl] = centroids[new_lbl]
        centroids = ordered_centroids
        
    previous_centroids = centroids
    
    # Store the results for the year closest to the user's selection (terminal year of this loop)
    if yr == selected_year:
        final_labels = labels
        final_sil_score = sil_score
        final_aic = aic
        final_bic = bic

# Fetch the raw/imputed dataframe slice purely for the selected year and append finalized labels
df_results = df_imputed_all[df_imputed_all['year'] == selected_year].sort_values('Country Name').copy()
df_results['Cluster'] = [str(lbl) for lbl in final_labels]


# -- Main Dashboard --

st.header(f"Clustering Metrics ({selected_year})")
col1, col2, col3 = st.columns(3)

with col1:
    if final_sil_score is not None:
        st.metric("Silhouette Score (Higher is better)", f"{final_sil_score:.3f}")
    else:
        st.metric("Silhouette Score", "N/A")
        
with col2:
    if final_aic is not None:
        st.metric("AIC (Lower is better)", f"{final_aic:.1f}")
    else:
        st.metric("AIC", "N/A (GMM Only)")
        
with col3:
    if final_bic is not None:
        st.metric("BIC (Lower is better)", f"{final_bic:.1f}")
    else:
        st.metric("BIC", "N/A (GMM Only)")

st.divider()

tab1, tab2 = st.tabs(["🌎 Global Map", "📑 Cluster Inspector"])

with tab1:
    st.header(f"Global Cluster Distribution ({selected_year})")

    fig = px.choropleth(
        df_results,
        locations="economy", 
        color="Cluster",
        hover_name="Country Name",
        hover_data=selected_features,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        projection="natural earth",
        title=f"{model_choice} Clustering (K={k_clusters}) for {selected_year}"
    )
    fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    st.header("Centroid Explorer")
    with st.expander("View Cluster Centroids (Unscaled / Mean Values)"):
        df_raw_slice = df_raw[df_raw['year'] == selected_year].sort_values('Country Name').copy()
        df_raw_slice['Cluster'] = [str(lbl) for lbl in final_labels]
        
        centroids_df = df_raw_slice.groupby('Cluster')[selected_features].mean()
        
        st.dataframe(centroids_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))
        st.markdown("*(Green indicates the highest value across clusters, Red indicates the lowest)*")

with tab2:
    st.header(f"Inspect Countries per Cluster ({selected_year})")
    st.markdown("Filter and inspect the exact nations assigned to each cluster, alongside their raw collected World Bank attributes.")
    
    df_raw_slice_tab2 = df_raw[df_raw['year'] == selected_year].sort_values('Country Name').copy()
    df_raw_slice_tab2['Cluster'] = [str(lbl) for lbl in final_labels]
    df_raw_slice_tab2 = df_raw_slice_tab2.dropna(subset=['Cluster'])
    
    cluster_options = ["All Clusters"] + sorted(list(df_raw_slice_tab2['Cluster'].unique()))
    selected_cluster = st.selectbox("Select Cluster to Filter", cluster_options)
    
    if selected_cluster == "All Clusters":
        display_df = df_raw_slice_tab2.sort_values(by=['Cluster', 'Country Name'])
    else:
        display_df = df_raw_slice_tab2[df_raw_slice_tab2['Cluster'] == selected_cluster].sort_values(by=['Country Name'])
        
    st.dataframe(
        display_df[['Cluster', 'Country Name'] + selected_features].set_index('Country Name'),
        use_container_width=True
    )
