from torch.utils.data import Dataset
import numpy as np
# from scipy.sparse import csr_matrix
import torch
import h5py
# import dgl
from utils import createLPEembedding, positional_encoding_nerf
class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, args):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.point_clouds_group = self.hdf5_file['point_clouds']
        self.num_point_clouds = len(self.point_clouds_group)
        self.indices = list(range(self.num_point_clouds))
        self.lpe_dim = args.lpe_dim
        self.PE_dim = args.PE_dim
        self.normalize = args.lpe_normalize
    def __len__(self):
        return self.num_point_clouds

    def __getitem__(self, idx):
        point_cloud_name = f"point_cloud_{self.indices[idx]}"

        # Load point cloud data
        point_cloud = self.point_clouds_group[point_cloud_name]
        #get canonical point cloud order
        pcl, lpe = createLPEembedding(point_cloud, self.lpe_dim, normalize=self.normalize)
        point_cloud = torch.tensor(pcl, dtype=torch.float32)
        if self.lpe_dim!=0:
            lpe = torch.tensor(lpe, dtype=torch.float32)
        else:
            lpe = torch.tensor([])

        if self.PE_dim!=0:
            # lpe, pcl = createLPE(point_cloud, self.lpe_dim)
            pe = positional_encoding_nerf(point_cloud, self.PE_dim)
            pe = torch.tensor(pe, dtype=torch.float32)
        else:
            pe = torch.tensor([])
        # Load metadata from attributes
        info = {key: self.point_clouds_group[point_cloud_name].attrs[key] for key in
                    self.point_clouds_group[point_cloud_name].attrs}

        return {"point_cloud": point_cloud, "lpe": lpe, "info": info, "pe": pe}

# def createLPE(data, lpe_dim):
#     umbrella = create_triangles_ring(data[1:, :], data[0, :])
#     centroids = umbrella[:, 2, :]
#     p1 = umbrella[:, 0, :]
#     p2 = umbrella[:, 1, :]
#     centroid_to_p1_distances = np.linalg.norm(centroids - p1, axis=1)
#     p1_to_p2_distances = np.linalg.norm(p1 - p2, axis=1)
#
#     # Combine the distances into a single list
#     distances_list = list(centroid_to_p1_distances) + list(p1_to_p2_distances)
#
#     a = [0 for x in range(1, 21)]
#     b = [x for x in range(1, 21)]
#     shifted = b[1:] + b[:1]
#     a.extend(b)
#     b.extend(shifted)
#     old_a = a.copy()
#     a.extend(b)
#     b.extend(old_a)
#     distances_list.extend(distances_list)
#     row = np.array(a)
#     col = np.array(b)
#     weights = np.array(distances_list)
#     # mat = csr_matrix((weights, (row, col)), shape=(21, 21)).toarray()
#     g = dgl.from_scipy(csr_matrix((weights, (row, col)), shape=(21, 21)))
#     # lap = csgraph.laplacian(mat)
#     # lpe = laplacian_pe(lap, 15)
#     lpe = dgl.lap_pe(g, lpe_dim)
#     pcl = np.concatenate([(data[0, :][np.newaxis]), p1])
#     return lpe, pcl
def laplacian_pe(lap, k):

    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(lap)
    kpartition_indices = np.argpartition(EigVal, k + 1)[:k + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = np.real(EigVec[:, topk_indices])

    return topk_EigVec
def create_triangles_ring(point_cloud, centroid):
    num_points = point_cloud.shape[0]
    triangles = np.zeros((num_points, 3, 3))

    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    # Project points onto the plane perpendicular to the normal
    projected_points = point_cloud - np.outer(np.dot(point_cloud, normal_at_centroid), normal_at_centroid)
    centroid_projected = centroid - np.outer(np.dot(centroid, normal_at_centroid), normal_at_centroid)
    # Calculate angles of each point with respect to the x-axis on the XY plane
    angles = np.arctan2(projected_points[:, 1] - centroid_projected[0,1], projected_points[:, 0] - centroid_projected[0,0])

    # Wrap angles to [0, 2*pi)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    # Sort the points based on angles
    sorted_indices = np.argsort(angles)
    sorted_points = point_cloud[sorted_indices]

    for i in range(0, num_points):
        triangles[i] = np.array([sorted_points[i], sorted_points[(i + 1) % num_points], centroid])

    return triangles




