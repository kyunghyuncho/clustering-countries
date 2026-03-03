# Global Demographics & Economic Clustering

An interactive Streamlit web application that clusters the world's countries based on live demographic and economic statistics. This tool utilizes unsupervised machine learning—specifically K-Means and a custom Gaussian Mixture Model (GMM) with a strictly enforced uniform prior—to group nations and plot them onto an interactive global map.

## Key Features

*   **Live Data Ingestion:** Fetches real-time, up-to-date indicators from the World Bank API (`wbgapi`).
*   **Intelligent Preprocessing:** Handles naturally sparse global datasets using K-Nearest Neighbors (KNN) Imputation, preserving data variance far better than simple mean imputation. Applies log-transformations to skewed features (like GDP) and scales data for distance-based clustering.
*   **Unsupervised Learning Models:**
    *   **K-Means:** Standard centroid-based clustering.
    *   **GMM (Uniform Prior):** A custom subclass of `scikit-learn`'s `GaussianMixture` that strictly enforces a uniform prior (equal component weights, $\pi_k = 1/K$) across all Expectation-Maximization (EM) iterations.
*   **Interactive Visualizations:** Renders robust, interactive Plotly choropleth maps, displaying country assignments and underlying metrics on hover.
*   **Real-time Metrics:** Calculates and displays evaluation metrics dynamically, including Silhouette Score, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

## Evaluated Indicators

The application clusters economies based on the following World Bank API metrics:
*   GDP per capita (Current US$)
*   GDP growth (Annual %)
*   Life expectancy at birth, total (Years)
*   Population density (People per sq. km of land area)
*   Population growth (Annual %)
*   Unemployment, total (% of total labor force)
*   Inflation, consumer prices (Annual %)
*   Individuals using the Internet (% of population)
*   Gini index (Inequality)
*   Military expenditure (% of GDP)
*   Intentional homicides (per 100,000 people)
*   Proportion of seats held by women in national parliaments (%)
*   Energy use (kg of oil equivalent per capita)

*Note: The script queries the past 10 years of data and cleverly backfills to handle null values, ensuring maximum global coverage without dropping sparse nations.*

## Architecture

*   **`app.py`**: The Streamlit frontend. It constructs the user interface, sidebar configuration panel, and renders the geospatial dashboard.
*   **`data.py`**: The data pipeline. Contains functions to download metrics via `wbgapi`, merge full economy names, and handle imputation and standardization using `scikit-learn` pipelines.
*   **`models.py`**: The core machine learning module. Hosts the standard K-Means function and the specialized `UniformGMM` expectation-maximization engine.

## Setup & Execution

### Prerequisites

This project utilizes `uv` for lightning-fast package management and strictly operates within a local virtual environment. Make sure you have `uv` installed.

### 1. Initialize the Environment

Create a virtual environment and activate it:

```bash
uv venv
source .venv/bin/activate
```

*(Note: The environment is created in `./.venv/`)*

### 2. Install Dependencies

Install the required packages outlined in `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

### 3. Run the Application

Execute the Streamlit dashboard using the activated `uv` environment:

```bash
uv run streamlit run app.py
```

Streamlit will launch a local web server (usually on `http://localhost:8501`) where you can interact with the clustering engine.

## Usage Guide

1.  **Select Features**: In the sidebar, choose which combinations of economic and demographic indicators the models should evaluate.
2.  **Select Algorithm**: Choose between the traditional K-Means standard or the probabilistic GMM with Uniform Prior.
3.  **Tune Hyperparameters**: 
    *   Adjust $K$ (number of clusters) to see how the mathematical groupings evolve and affect validation scores (Silhouette, AIC, BIC).
    *   If using GMM, tune the Covariance Type (Full, Tied, Diag, Spherical) to alter cluster boundaries.
    *   Adjust the "Imputation k-Neighbors" slider to determine how broadly the pipeline guesses missing features for sparse economies.
4.  **Explore**: Hover over the world map to see a country's assigned cluster and its metrics. Open the "Centroid Explorer" expander to view the mathematical center (average) of each resulting cluster in its unscaled numerical state.