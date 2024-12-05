import numpy as np

class RandomizedPCA:
    def __init__(self, n_components, n_oversamples=10, n_iter=2):
        """
        Randomized PCA implementation.

        Parameters:
        - n_components: Number of principal components to compute.
        - n_oversamples: Additional dimensions for oversampling.
        - n_iter: Number of power iterations for accuracy improvement.
        """
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.components_ = None  # Principal axes in the feature space, shape (n_components, n_features)
        self.mean_ = None  # Per-feature empirical mean, shape (n_features,)
        self.explained_variance_ = None  # Amount of variance explained per principal component, shape (n_components,)

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).
        """
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Random projection
        n_samples, n_features = X_centered.shape
        n_random = self.n_components + self.n_oversamples
        random_matrix = np.random.randn(n_features, n_random)
        Y = X_centered @ random_matrix

        # 3. Power iterations (optional for improved accuracy)
        for _ in range(self.n_iter):
            Y = X_centered @ (X_centered.T @ Y)

        # 4. Orthonormal basis (QR decomposition)
        Q, _ = np.linalg.qr(Y, mode='reduced')

        # 5. Small matrix SVD
        B = Q.T @ X_centered
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

        # 6. Recover principal components
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (n_samples - 1)

    def transform(self, X):
        """
        Project the data onto the principal components.

        Parameters:
        - X: Input data, shape (n_samples, n_features).

        Returns:
        - X_transformed: Projected data, shape (n_samples, n_components).
        """
        # Center the data
        X_centered = X - self.mean_

        # Transform the data
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """
        Fit the model to the data and transform it.

        Parameters:
        - X: Input data, shape (n_samples, n_features).

        Returns:
        - X_transformed: Projected data, shape (n_samples, n_components).
        """
        # Fit the model to the data
        self.fit(X)

        # Transform the data
        return self.transform(X)
