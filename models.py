import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

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
