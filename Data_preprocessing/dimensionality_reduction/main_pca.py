import matplotlib.pyplot as plt

from sklearn import datasets 

from pca import ACP 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearn_pca_model

iris_dataset = datasets.load_iris()
original_data = iris_dataset.data
target = iris_dataset.target

#print(original_data[:20])
#print(target[:20])
#print(original_data.shape)
#print(target.shape)

print("PCA FROM SCRATCH:")
pca_from_scratch = ACP(n_component=2)
pca_from_scratch.fit(original_data)
transformed_data = pca_from_scratch.transform(original_data)
#transformed_data = pca_from-scratch.fit_transform(original_data)

pca_from_scratch.plot_cov_matrix()
pca_from_scratch.plot_cumulative_explained_variance_ratio()

print(pca_from_scratch.components)
print(pca_from_scratch.explained_variances())
print(pca_from_scratch.explained_variances_ratio())

print()
print("PCA SCIKIT-LEARN:")

scaler = StandardScaler()
scaler.fit(original_data)
normalized_data = scaler.transform(original_data)

sklearn_pca = sklearn_pca_model(n_components=2)
sklearn_pca.fit(normalized_data)
transformed_data_sklearn = sklearn_pca.transform(normalized_data)

print(sklearn_pca.components_)
print(sklearn_pca.explained_variance_)
print(sklearn_pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 3, figsize=(15, 7))
axes[0].scatter(original_data[:, 0], original_data[:, 1], c=target) 
axes[0].set_xlabel('VAR1') 
axes[0].set_ylabel('VAR2') 
axes[0].set_title('Avant PCA') 
axes[1].scatter(transformed_data[:, 0], transformed_data[:, 1], c=target) 
axes[1].set_xlabel('PC1') 
axes[1].set_xlabel('PC2')  
axes[1].set_xlabel('PCA From Scratch')
axes[2].scatter(original_data[:, 0], original_data[:, 1], c=target) 
axes[2].set_xlabel('PC1')
axes[2].set_xlabel('PC2')
axes[2].set_xlabel('Scikit-learn PCA')

plt.show()
