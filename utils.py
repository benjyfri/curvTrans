import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import eigh

def createLap(point_cloud, normalized):
    # Calculate pairwise distances between points
    distances = distance_matrix(point_cloud, point_cloud)

    # Compute weights as e^(distance)
    weights = np.exp(distances)

    column_sums = np.sum(weights, axis=0)

    # Create diagonal matrix from column sums
    diag_matrix = np.diag(column_sums)

    Laplacian = diag_matrix - weights
    # Maybe better to use?
    if normalized:
        inv_D_sqrt = np.sqrt(np.linalg.inv(diag_matrix))
        identity = np.ones(weights.shape[0])
        Laplacian = identity - (inv_D_sqrt @ weights @ inv_D_sqrt)
    return Laplacian

def top_k_smallest_eigenvectors(graph, k):
    """
    Calculate the top k smallest eigenvectors corresponding to the smallest eigenvalues of a graph Laplacian.

    Parameters:
        graph (ndarray): The graph Laplacian matrix.
        k (int): The number of eigenvectors to return.

    Returns:
        eigvecs (ndarray): The top k smallest eigenvectors - not including the first trivial one!
    """
    n = graph.shape[0]  # Number of nodes
    eigenvalues, eigenvectors = eigh(graph, eigvals=(1, k))
    sorted_indices = np.argsort(eigenvalues)
    eigvecs = eigenvectors[:, sorted_indices]
    return eigvecs

def sort_by_first_eigenvector(eigenvectors):
    """
    Sort the rows of a matrix by the absolute values of the entries in the first eigenvector.

    Parameters:
        matrix (ndarray): The matrix whose rows are to be sorted.
        eigenvector (ndarray): The first eigenvector.

    Returns:
        sorted_matrix (ndarray): The matrix with rows sorted based on the absolute values of the entries in the first eigenvector.
    """
    # Calculate absolute values of the entries in the first eigenvector
    abs_eigenvector = np.abs(eigenvectors[:, 0])

    # Sort the rows of the matrix based on the absolute values of the entries in the first eigenvector
    sorted_indices = np.argsort(abs_eigenvector)

    # make 0 (a.k.a centroid) be the first canonical entry
    sorted_indices = [0] + [x for x in sorted_indices if x != 0]
    sorted_matrix = eigenvectors[sorted_indices]

    return sorted_indices , sorted_matrix

def createLPEembedding(point_cloud, emb_size=5, normalize=False):
    l = createLap(point_cloud, normalize)
    eigvecs = top_k_smallest_eigenvectors(l, emb_size)
    indices, fixed_eigs = sort_by_first_eigenvector(eigvecs)
    pcl_fixed = np.array(point_cloud)[indices]
    return pcl_fixed, fixed_eigs