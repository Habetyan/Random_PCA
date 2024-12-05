import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import svd

class RandomizedPCA:
    def __init__(self, n_components, n_oversamples=10, n_iter=2, gpu=False):
        """
        Randomized PCA implementation.

        Parameters:
        - n_components: Number of principal components to compute.
        - n_oversamples: Additional dimensions for oversampling.
        - n_iter: Number of power iterations for accuracy improvement.
        - gpu: If True, uses GPU for computations (requires CuPy).
        """
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.gpu = gpu
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None

    def fit(self, X):
        # Use CuPy for GPU operations, NumPy otherwise
        if self.gpu:
            import cupy as cp
            xp = cp
        else:
            xp = np

        # 1. Center the data
        X = xp.array(X)  # Ensure input is a CuPy array if GPU
        self.mean_ = xp.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Random projection
        n_samples, n_features = X_centered.shape
        n_random = self.n_components + self.n_oversamples
        random_matrix = sparse_random(n_features, n_random, density=0.1, format='csr').toarray()
        random_matrix = xp.array(random_matrix)  # Convert random_matrix to CuPy array if GPU
        Y = X_centered @ random_matrix

        # 3. Power iterations
        for _ in range(self.n_iter):
            Y = X_centered @ (X_centered.T @ Y)
            Y = Y / xp.linalg.norm(Y, axis=0, keepdims=True)  # Normalize columns

        # 4. Orthonormal basis
        Q, _ = xp.linalg.qr(Y, mode='reduced')

        # 5. Small matrix SVD
        B = Q.T @ X_centered
        if self.gpu:
            B_cpu = B.get()  # Transfer to CPU for SVD
            U_hat, S, Vt = svd(B_cpu, full_matrices=False)
        else:
            U_hat, S, Vt = svd(B, full_matrices=False)

        # 6. Recover principal components
        self.components_ = xp.array(Vt[:self.n_components]) if self.gpu else Vt[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (n_samples - 1)

    def transform(self, X):
        if self.gpu:
            import cupy as cp
            xp = cp
        else:
            xp = np

        X = xp.array(X)  # Ensure input is consistent
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
