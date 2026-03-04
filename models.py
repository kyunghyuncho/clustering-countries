import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def run_kmeans(X, n_clusters, random_state=42):
    """
    Fits a K-Means model on the data and calculates the silhouette score.
    Returns the fitted model, cluster assignments, and silhouette score.
    """
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    labels = model.fit_predict(X)
    
    # Calculate silhouette score (only valid for 2 <= n_clusters <= n_samples-1)
    if 2 <= n_clusters <= len(X) - 1:
        sil_score = silhouette_score(X, labels)
    else:
        sil_score = None
        
    return model, labels, sil_score

class UniformGMM(GaussianMixture):
    """
    A subclass of GaussianMixture that strictly enforces a uniform prior 
    (equal component weights) across all EM iterations.
    """
    def _m_step(self, X, log_resp):
        """
        M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        # Call the parent's M-step to update means, covariances, and precisions.
        # This standard M-step ALSO updates the component weights based on responsibilities.
        super()._m_step(X, log_resp)
        
        # Override the updated weights to enforce the uniform prior
        n_components = self.n_components
        self.weights_ = np.ones(n_components) / n_components

def run_gmm(X, n_components, covariance_type='full', random_state=42):
    """
    Fits our custom UniformGMM on the data and calculates metrics.
    Returns the fitted model, cluster assignments, silhouette score, AIC, and BIC.
    """
    model = UniformGMM(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = model.fit_predict(X)
    
    # Probabilistic fit metrics
    aic = model.aic(X)
    bic = model.bic(X)
    
    # Silhouette score
    if 2 <= n_components <= len(X) - 1:
        sil_score = silhouette_score(X, labels)
    else:
        sil_score = None
        
    return model, labels, sil_score, aic, bic


def align_clusters(old_centroids, new_centroids):
    """
    Aligns new cluster labels to old cluster labels using the Hungarian algorithm
    based on the Euclidean distance between their centroids.
    
    Returns a dictionary mapping {new_label_index: old_label_index}.
    """
    # Compute cross-distance matrix (cost matrix)
    cost_matrix = cdist(new_centroids, old_centroids, metric='euclidean')
    
    # Minimize total distance to assign pairs
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # row_ind => new cluster index
    # col_ind => corresponding old cluster index
    return {new_idx: old_idx for new_idx, old_idx in zip(row_ind, col_ind)}
