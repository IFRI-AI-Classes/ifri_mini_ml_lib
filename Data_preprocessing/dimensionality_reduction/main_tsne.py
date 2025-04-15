from t_sne import TSNE
import matplotlib.pyplot as plt
import time

def main():
    # Génération des données
    X, y = TSNE.generate_test_data(300, case='blobs', random_state=42)
    
    # Visualisation des données originales
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131)
    TSNE.plot_results(X[:, :2], y, "Projection 2D originale", ax1)
    
    ax2 = fig.add_subplot(132, projection='3d')
    TSNE.plot_results(X, y, "Données originales 3D", ax2)
    
    # Application de t-SNE
    tsne = TSNE(n_components=2, perplexity=30, 
                learning_rate=200, n_iter=1000,
                random_state=42, verbose=1)
    
    start_time = time.time()
    X_tsne = tsne.fit_transform(X)
    duration = time.time() - start_time
    
    # Visualisation des résultats
    ax3 = fig.add_subplot(133)
    TSNE.plot_results(X_tsne, y, f"t-SNE 2D (temps: {duration:.2f}s)", ax3)
    plt.tight_layout()
    plt.show()
    
    # Affichage des informations
    print("\nRésultats t-SNE:")
    print(f"Divergence KL finale: {tsne.kl_divergence_:.4f}")
    print(f"Itérations effectuées: {tsne.n_iter_}")

if __name__ == "__main__":
    main()