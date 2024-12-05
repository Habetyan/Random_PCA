#import numpy as np
# from randomized_pca import RandomizedPCA
# from sklearn.decomposition import PCA
# import time
#
# # Generate synthetic data
# np.random.seed(42)
# X = np.random.randn(1000, 500)  # Large dataset
#
# # Scikit-learn PCA
# start_time = time.time()
# pca_sklearn = PCA(n_components=10)
# X_reduced_sklearn = pca_sklearn.fit_transform(X)
# end_time = time.time()
# print(f"Scikit-learn PCA time: {end_time - start_time:.4f} seconds")
#
# # Randomized PCA
# start_time = time.time()
# pca_randomized = RandomizedPCA(n_components=10, n_iter=5)
# X_reduced_randomized = pca_randomized.fit_transform(X)
# end_time = time.time()
# print(f"Randomized PCA time: {end_time - start_time:.4f} seconds")
#
# # Compare explained variance
# print("Explained Variance (Scikit-learn):", pca_sklearn.explained_variance_[:5])
# print("Explained Variance (Randomized):", pca_randomized.explained_variance_[:5])
import numpy as np
from r_pca_gpu import RandomizedPCA
np.random.seed(42)
X = np.random.randn(10000, 500)
pca = RandomizedPCA(n_components=10, gpu=True)
X_reduced = pca.fit_transform(X)
print("Reduced Shape:", X_reduced.shape)
print("Explained Variance:", pca.explained_variance_)
