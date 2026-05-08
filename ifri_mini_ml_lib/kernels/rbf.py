import numpy as np

def rbf_kernel(x, y, gamma = 1.0) :
    
    """
    Computes the RBF (Radial Basis Function) kernel between
    two vectors x and y
    
    """
    
    # Compute the difference between the two vectors
    difference = x - y
    
    
    # Compute the squared distance
    squared_distance = np.sum(difference ** 2)
    
    
    # Apply the exponential RBF formula
    rbf_value = np.exp( -gamma * squared_distance )
    
    
    return rbf_value




def rbf_gram_matrix ( X1, X2, gamma = 1.0 ) :
    
    """
    Computes the RBF gram matrix between two datasets X1 and X2
    
    """
    
    
    # Number of samples (rows) in X1 and X2
    n_samples_1 = X1.shape[0]
    n_samples_2 = X2.shape[0]
    
    
    # Create an empty matrix filled with zeros
    gram_matrix = np.zeros((n_samples_1, n_samples_2))
    
    
    for i in range (n_samples_1) :
        for j in range (n_samples_2) :
            gram_matrix[i, j] = rbf_kernel (X1[i], X2[j], gamma)
            
            
    
    return gram_matrix
    

    