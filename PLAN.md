# Project Plan: Global Demographics & Economic Clustering Web App

## 1. Project Overview & Architecture
This project involves developing an interactive Streamlit web application that clusters the world's countries based on demographic and economic statistics. The app will utilize K-Means and Gaussian Mixture Models (GMM) with a strictly enforced uniform prior. The architecture is designed to be completely data-driven, fetching live or recent data, processing it through a robust `scikit-learn` pipeline, and visualizing the output on a global Plotly choropleth map.

* **Frontend/UI:** Streamlit (`streamlit`)
* **Machine Learning:** `scikit-learn`
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `plotly.express`
* **Data Ingestion:** World Bank API (`wbgapi`)

## 2. Data Pipeline & Preprocessing


### 2.1. Data Ingestion & Caching
* **Source:** Fetch live indicator data using the `wbgapi` library to avoid bundling static CSVs. Query metrics such as GDP per capita (`NY.GDP.PCAP.CD`), Life expectancy (`SP.DYN.LE00.IN`), Population density (`EN.POP.DNST`), and CO2 emissions (`EN.ATM.CO2E.PC`).
* **Format requirement:** The dataset must be pivoted so that rows represent countries (indexed by ISO-3 standard country codes) and columns represent the selected features.
* **Caching:** Use the `@st.cache_data` decorator on the ingestion function to prevent redundant API calls when the user interacts with the UI.

### 2.2. Handling Missing Data
* **Sparsity Reality:** Global datasets are inherently sparse; dropping rows with `NaN` values (listwise deletion) will remove too many countries from the visualization.
* **Imputation Strategy:** Implement `sklearn.impute.KNNImputer`.
* **Mechanism:** Estimate missing features by averaging the values of the $k$ nearest countries (default $k=5$) based on the features that are present, preserving the variance better than global mean imputation.

### 2.3. Feature Transformation & Scaling
* **Log-Transformation:** Identify highly skewed, power-law distributed features (e.g., GDP, total population) and apply a log-transformation $x' = \log(x + 1)$ to normalize the distribution and reduce the disproportionate pull of extreme outliers.
* **Standardization:** Apply `sklearn.preprocessing.StandardScaler`. Both K-Means (Euclidean distance) and GMM (covariance structures) are scale-sensitive. Standardize the data to $\mu = 0$ and $\sigma = 1$ using:
    $$z = \frac{x - \mu}{\sigma}$$

## 3. Clustering Algorithms & Implementation


### 3.1. Algorithm 1: K-Means
* **Implementation:** `sklearn.cluster.KMeans`
* **Objective:** Minimizes the within-cluster sum of squares:
    $$\arg\min_{\mathbf{S}} \sum_{i=1}^{K} \sum_{\mathbf{x} \in S_i} \left\| \mathbf{x} - \boldsymbol{\mu}_i \right\|^2$$
* **Configurable Hyperparameters:** Number of clusters ($K$), `init` (default to 'k-means++' for smarter centroid initialization), `n_init`, and `random_state`.

### 3.2. Algorithm 2: Gaussian Mixture Model (GMM) with Uniform Prior
* **Implementation Challenge:** The standard `sklearn.mixture.GaussianMixture` uses the Expectation-Maximization (EM) algorithm, which updates the component mixing weights (priors) $\pi_k$ during the M-step based on the empirical distribution of the responsibilities.
* **Enforcing Uniform Prior:** To strictly enforce a uniform prior where $\pi_k = \frac{1}{K}$ across all EM iterations, standard instantiation is insufficient.
* **Solution Strategy:** Instruct the agent to subclass `GaussianMixture` and override the `_m_step` method. The custom class must update the means and covariances as usual, but manually reset or freeze the weights to $\frac{1}{K}$ at the end of each M-step.
* **Configurable Hyperparameters:** Number of components ($K$), `covariance_type` (full, tied, diag, spherical).

## 4. Evaluation & Generalization


Because this is an unsupervised learning task on a finite dataset (~195 countries), a standard train/validation/test split will result in heavily fragmented clusters that lack meaningful global context. Instead of a holdout set, the app will evaluate cluster quality and stability.

* **Cluster Quality (Both Models):** Calculate and display the **Silhouette Score**. This metric ranges from $-1$ to $1$ and evaluates how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
    $$s = \frac{b - a}{\max(a, b)}$$
    *(where $a$ is the mean intra-cluster distance and $b$ is the mean nearest-cluster distance).*
* **Probabilistic Fit (GMM Only):** Calculate and display the **Bayesian Information Criterion (BIC)** and **Akaike Information Criterion (AIC)**. These penalize model complexity (number of components $K$ and covariance parameters) to help the user find the optimal $K$ without overfitting.
* **Generalization/Stability Test:** Implement a "Robustness Check" button that fits the chosen model on a random 80% subsample of the data and compares the resulting cluster centroids to the full-dataset centroids (measuring variance).

## 5. Application UI/UX Design (Streamlit)
* **Sidebar Configuration Panel:**
    * **Data Selection:** `st.multiselect` to choose the demographic/economic features.
    * **Model Selection:** `st.radio` to toggle between "K-Means" and "GMM (Uniform Prior)".
    * **Hyperparameters:** `st.slider` for the number of clusters/components ($K$), and `st.selectbox` for GMM covariance types.
* **Main Dashboard:**
    * **Metrics Row:** Use `st.columns` and `st.metric` to display the Silhouette Score, AIC, and BIC dynamically as the user adjusts the sidebar.
    * **Geospatial Visualization:** Implement `plotly.express.choropleth`.
        
        * Map the model's output `labels_` to the country ISO-3 codes.
        * Use a discrete categorical color scale (e.g., `px.colors.qualitative.Plotly`) so clusters are easily distinguishable.
        * Include hover data showing the country name, assigned cluster, and key stats.
    * **Data & Centroid Explorer:** Include an `st.expander` containing a Pandas dataframe of the final cluster centroids (in the original, unscaled feature space via `inverse_transform`) to allow the user to interpret what each cluster actually represents (e.g., "High GDP, Low Birth Rate").

## 6. Execution Steps for the AI Agent
1.  **Dependency Management:** Generate `requirements.txt` with exact library versions.
2.  **Data Module (`data.py`):** Write the `wbgapi` fetching logic, the `KNNImputer` pipeline, and the log/standard scaling functions. Ensure strict ISO-3 indexing.
3.  **Modeling Module (`models.py`):** Implement the K-Means logic and specifically author the custom `GaussianMixture` subclass that overrides `_m_step` to force $\pi_k = \frac{1}{K}$.
4.  **UI Module (`app.py`):** Build the Streamlit interface, bind the hyperparameters to the modeling functions, and render the Plotly choropleth figure based on the returned labels.