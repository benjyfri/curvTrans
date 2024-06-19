from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from utils import createLPEembedding, positional_encoding_nerf
import plotly.express as px
import random

class BasicPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, args):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.point_clouds_group = self.hdf5_file['point_clouds']
        self.num_point_clouds = len(self.point_clouds_group)
        self.pcls_per_class = self.num_point_clouds // 4
        self.indices = list(range(self.num_point_clouds))
        self.std_dev = args.std_dev
        self.rotate_data = args.rotate_data
        self.contr_loss_weight = args.contr_loss_weight
        self.sampled_points = args.sampled_points
        self.smoothness_loss = args.smoothness_loss
        self.smooth_num_of_neighbors = args.smooth_num_of_neighbors
        self.pcl_scaling = args.pcl_scaling
    def __len__(self):
        return self.num_point_clouds

    def __getitem__(self, idx):
        point_cloud_name = f"point_cloud_{self.indices[idx]}"
        # Load metadata from attributes
        info = {key: self.point_clouds_group[point_cloud_name].attrs[key] for key in
                    self.point_clouds_group[point_cloud_name].attrs}
        point_cloud = self.point_clouds_group[point_cloud_name]
        point_cloud = np.array(point_cloud, dtype=np.float32)
        increase_scale = np.random.uniform(low=1, high=self.pcl_scaling)
        decrease_scale = np.random.uniform(low=(1/self.pcl_scaling), high=1)
        scale = random.choice([increase_scale, decrease_scale])
        point_cloud = scale * point_cloud
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        # permute points
        shuffled_indices = torch.randperm(self.sampled_points) + 1
        permuted_indices = torch.cat((torch.tensor([0]), shuffled_indices), dim=0)
        point_cloud = point_cloud[permuted_indices]
        if self.rotate_data:
            point_cloud1 = random_rotation(point_cloud)
        else:
            point_cloud1 = point_cloud
        #Add noise to point cloud
        if self.std_dev != 0:
            noise = torch.normal(0, self.std_dev, size=point_cloud1.shape, dtype=torch.float32)
            point_cloud1 = point_cloud1 + noise
            point_cloud1 = point_cloud1 - point_cloud1[0,:]
        if self.smoothness_loss != 0:
            _, indices = torch.sort(torch.norm(point_cloud, dim=1)[1:], dim=0)
            positive_random_neighbor = random.randrange(self.smooth_num_of_neighbors)
            positive_chosen_neighbor_index = indices[positive_random_neighbor]
            positive_point = point_cloud[1 + positive_chosen_neighbor_index,:].clone()
            positive_smooth_point_cloud = samplePoints(info['a'], info['b'], info['c'], info['d'], info['e'],
                                                       count=self.sampled_points, center_point=positive_point.numpy())
            positive_smooth_point_cloud = torch.tensor(positive_smooth_point_cloud, dtype=torch.float32)
            positive_smooth_point_cloud = random_rotation(positive_smooth_point_cloud)

            negative_random_neighbor = random.randrange((self.sampled_points - self.smooth_num_of_neighbors), self.sampled_points)
            negative_chosen_neighbor_index = indices[negative_random_neighbor]
            negative_point = point_cloud[1 + negative_chosen_neighbor_index,:].clone()
            negative_smooth_point_cloud = samplePoints(info['a'], info['b'], info['c'], info['d'], info['e'],
                                                count=self.sampled_points, center_point=negative_point.numpy())
            negative_smooth_point_cloud = torch.tensor(negative_smooth_point_cloud, dtype=torch.float32)
            negative_smooth_point_cloud = random_rotation(negative_smooth_point_cloud)

            if self.std_dev != 0:
                positive_smooth_noise = torch.normal(0, self.std_dev, size=positive_smooth_point_cloud.shape, dtype=torch.float32)
                positive_smooth_point_cloud = positive_smooth_point_cloud + positive_smooth_noise
                positive_smooth_point_cloud = positive_smooth_point_cloud - positive_smooth_point_cloud[0, :]
                negative_smooth_noise = torch.normal(0, self.std_dev, size=negative_smooth_point_cloud.shape, dtype=torch.float32)
                negative_smooth_point_cloud = negative_smooth_point_cloud + negative_smooth_noise
                negative_smooth_point_cloud = negative_smooth_point_cloud - negative_smooth_point_cloud[0, :]

        else:
            positive_smooth_point_cloud = torch.tensor((0))
            negative_smooth_point_cloud = torch.tensor((0))
        if self.contr_loss_weight  != 0:
            a = info['a'] + np.random.normal(0, 2)
            b = info['b'] + np.random.normal(0, 2)
            c = info['c'] + np.random.normal(0, 2)
            d = info['d'] + np.random.normal(0, 2)
            e = info['e'] + np.random.normal(0, 2)
            contrastive_point_cloud = samplePoints(a, b, c, d, e, count=self.sampled_points)
            contrastive_point_cloud = torch.tensor(contrastive_point_cloud, dtype=torch.float32)
            contrastive_point_cloud = random_rotation(contrastive_point_cloud)

            positive_point_cloud = samplePoints(info['a'], info['b'], info['c'], info['d'], info['e'], count=self.sampled_points)
            point_cloud2 = torch.tensor(positive_point_cloud, dtype=torch.float32)
            point_cloud2 = random_rotation(point_cloud2)
            if self.std_dev != 0:
                noise = torch.normal(0, self.std_dev, size=point_cloud2.shape, dtype=torch.float32)
                point_cloud2 = point_cloud2 + noise
                point_cloud2 = point_cloud2 - point_cloud2[0, :]
                contrastive_noise = torch.normal(0, self.std_dev, size=contrastive_point_cloud.shape, dtype=torch.float32)
                contrastive_point_cloud = contrastive_point_cloud + contrastive_noise
                contrastive_point_cloud = contrastive_point_cloud - contrastive_point_cloud[0, :]
        else:
            point_cloud2 = torch.tensor((0))
            contrastive_point_cloud = torch.tensor((0))

        return {"point_cloud": point_cloud1, "point_cloud2": point_cloud2, "contrastive_point_cloud":contrastive_point_cloud, "positive_smooth_point_cloud":positive_smooth_point_cloud, "negative_smooth_point_cloud":negative_smooth_point_cloud, "info": info}
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
        old_pcl = torch.tensor(point_cloud, dtype=torch.float32)
        # createLPE(point_cloud)
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

        return {"point_cloud": point_cloud, "lpe": lpe, "info": info, "pe": pe, "old_pcl": old_pcl}

def createLPE(data):
    umbrella = estimate_HK_from_one_ring(data[1:, :], data[0, :], k=3)

def laplacian_pe(lap, k):

    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(lap)
    kpartition_indices = np.argpartition(EigVal, k + 1)[:k + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = np.real(EigVec[:, topk_indices])

    return topk_EigVec
def estimate_KH_from_one_ring(point_cloud, centroid, k):
    num_points = point_cloud.shape[0]

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
    sampled_indices = np.random.choice(num_points, size=k, replace=False)
    sampled_points = sorted_points[sampled_indices]

    full_area = 0.0
    angles_sum = 0.0
    mean_curvature_normal = np.zeros((3,))
    for i in range(k):
        a = centroid
        b = sampled_points[i]
        c = sampled_points[(i + 1) % k]
        d = sampled_points[(i - 1) % k]
        angle_at_a, angle_at_c, angle_at_d, area = calculate_angle_and_area(a, b, c, d)
        full_area += area
        angles_sum += angle_at_a
        mean_curvature_normal += ((1 / np.tan(angle_at_d)) + (1 / np.tan(angle_at_c))) * (a - b)
    H_est = ( np.linalg.norm((mean_curvature_normal) / (full_area / 3)) ) / 2
    K_est = ( 2 * (np.pi) - angles_sum ) / (full_area / 3)

    return K_est , H_est

def calculate_angle_and_area(a, b, c , d):
    # Calculate vectors AB and AC
    ab = b - a
    ac = c - a

    cb = b - c
    ca = -ac

    da = a - d
    db = b - d

    # Calculate dot product of AB and AC
    dot_product_a = np.dot(ab, ac)
    dot_product_c = np.dot(ca, cb)
    dot_product_d = np.dot(da, db)

    ab_magnitude = np.linalg.norm(ab)
    ac_magnitude = np.linalg.norm(ac)

    cb_magnitude = np.linalg.norm(cb)
    ca_magnitude = ac_magnitude

    da_magnitude = np.linalg.norm(da)
    db_magnitude = np.linalg.norm(db)

    # Calculate cosine of the angle at vertex A
    cos_angle_a = dot_product_a / (ab_magnitude * ac_magnitude)
    cos_angle_c = dot_product_c / (ca_magnitude * cb_magnitude)
    cos_angle_d = dot_product_d / (da_magnitude * db_magnitude)

    # Calculate angle at vertex A (in radians)
    angle_at_a = np.arccos(np.clip(cos_angle_a, -1, 1))
    angle_at_c = np.arccos(np.clip(cos_angle_c, -1, 1))
    angle_at_d = np.arccos(np.clip(cos_angle_d, -1, 1))

    # Calculate area of the triangle using cross product
    area = 0.5 * np.linalg.norm(np.cross(ab, ac))

    return angle_at_a, angle_at_c, angle_at_d, area


def random_rotation(point_cloud):
    device = point_cloud.device
    is_rotation = False
    while is_rotation==False:

        # Generate random rotation angles around x, y, and z axes
        theta_x = torch.tensor(np.random.uniform(0, 2 * np.pi), device=device)
        theta_y = torch.tensor(np.random.uniform(0, 2 * np.pi), device=device)
        theta_z = torch.tensor(np.random.uniform(0, 2 * np.pi), device=device)

        # Rotation matrices around x, y, and z axes
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(theta_x), -torch.sin(theta_x)],
                           [0, torch.sin(theta_x), torch.cos(theta_x)]], device=device)

        Ry = torch.tensor([[torch.cos(theta_y), 0, torch.sin(theta_y)],
                           [0, 1, 0],
                           [-torch.sin(theta_y), 0, torch.cos(theta_y)]], device=device)

        Rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                           [torch.sin(theta_z), torch.cos(theta_z), 0],
                           [0, 0, 1]], device=device)

        # Combine rotation matrices
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        is_rotation = torch.allclose(torch.eye(3, device=device), torch.matmul(R, R.T),atol=1e-06)
    # Apply rotation to point cloud
    rotated_point_cloud = torch.matmul(point_cloud, R.T)
    return rotated_point_cloud
import plotly.graph_objects as go

def plot_point_clouds(point_cloud1, point_cloud2, title):
    """
    Plot two point clouds in an interactive 3D plot with Plotly.

    Args:
        point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
        point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
        title (str): Title of the plot
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
        mode='markers', marker=dict(color='red'),opacity=0.8, name='Point Cloud 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
        mode='markers', marker=dict(color='blue'),opacity=0.8, name='Point Cloud 2'
    ))
    # Add a separate trace for the point (0, 0, 0) in bright pink
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(color='rgb(255, 105, 180)'), name='Origin (0, 0, 0)'
    ))

    fig.update_layout(
        title=title,  # Set the title
        title_y=0.9,  # Adjust the y position of the title
        scene=dict(
            xaxis=dict(title='X', range=[min(point_cloud1[:,0].min(), point_cloud2[:,0].min()),
                                         max(point_cloud1[:,0].max(), point_cloud2[:,0].max())]),
            yaxis=dict(title='Y', range=[min(point_cloud1[:,1].min(), point_cloud2[:,1].min()),
                                         max(point_cloud1[:,1].max(), point_cloud2[:,1].max())]),
            zaxis=dict(title='Z', range=[min(point_cloud1[:,2].min(), point_cloud2[:,2].min()),
                                         max(point_cloud1[:,2].max(), point_cloud2[:,2].max())]),
            aspectmode='cube'  # Enforce same scale for all axes
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig.show()

def samplePoints(a, b, c, d, e, count, center_point=np.array([0,0,0])):
    def surface_function(x, y):
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y

    # Generate random points within the range [-1, 1] for both x and y
    x_samples = np.random.uniform(-1, 1, count) + center_point[0]
    y_samples = np.random.uniform(-1, 1, count) + center_point[1]

    # Evaluate the surface function at the random points
    z_samples = surface_function(x_samples, y_samples)

    # Create an array with the sampled points
    sampled_points = np.column_stack((x_samples, y_samples, z_samples))

    # Concatenate the centroid [0, 0, 0] to the beginning of the array
    centroid = np.expand_dims(center_point, axis=0)
    sampled_points_with_centroid = np.concatenate((centroid, sampled_points), axis=0)

    return sampled_points_with_centroid - centroid

def plotFunc(a, b, c, d, e,sampled_points):
    # Create a grid of points for the surface
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)

    # Compute the surface using the generated coefficients
    z = a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y

    # Compute the distance from each point to the origin
    distance_to_origin = np.sqrt(x**2 + y**2)

    # Create a mask for points within a radius of 0.25 from the origin
    mask = distance_to_origin <= 0.25

    # Create 3D surface plot using Plotly Express
    fig = px.scatter_3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), color=mask.flatten(),
                        color_continuous_scale=['blue', 'red'], title="Generated Surface",
                        labels={'x': 'X', 'y': 'Y', 'z': 'Z'}, range_color=[0, 1])

    for i, point in enumerate(sampled_points):
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                   mode='markers+text', text=[f'{i + 1}'],
                                   marker=dict(size=25, color='yellow'), name='Point Cloud'),)

    # Show the plot
    fig.show()
