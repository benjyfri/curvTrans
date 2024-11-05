from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from utils import createLPEembedding, positional_encoding_nerf
import plotly.express as px
import random
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
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
        self.normalization_factor = args.normalization_factor
    def __len__(self):
        return self.num_point_clouds

    def __getitem__(self, idx):
        point_cloud_name = f"point_cloud_{self.indices[idx]}"
        # Load metadata from attributes
        info = {key: self.point_clouds_group[point_cloud_name].attrs[key] for key in
                    self.point_clouds_group[point_cloud_name].attrs}
        # point_cloud = self.point_clouds_group[point_cloud_name]
        # point_cloud_orig = np.array(point_cloud, dtype=np.float32)
        if info['class'] <= 3:
            point_cloud = samplePoints(info['a'], info['b'], info['c'], info['d'], info['e'],
                                                count=self.sampled_points)
        else:
            point_cloud = sampleHalfSpacePoints(info['a'], info['b'], info['c'], info['d'], info['e'],
                                                count=self.sampled_points)
        # elif info['class'] == 7:
        #     # point_cloud = sample_points_on_pyramid(num_samples=self.sampled_points)
        #     point_cloud = generate_room_corner_with_points(self.sampled_points)
        # else:
        #     if info['class']==4:
        #         angle = 10
        #     if info['class']==5:
        #         angle = 45
        #     if info['class']==6:
        #         angle = 90
        #     rand_angle = np.random.uniform(angle - 10, angle + 10)
        #     point_cloud = generate_surfaces_angles_and_sample(N=self.sampled_points, angle=rand_angle)


        point_cloud = point_cloud / self.normalization_factor
        if self.pcl_scaling > 1.0:
            increase_scale = np.random.uniform(low=1, high=self.pcl_scaling)
            decrease_scale = np.random.uniform(low=(1/self.pcl_scaling), high=1)
            scale = random.choice([increase_scale, decrease_scale])
            point_cloud = scale * point_cloud
        if self.rotate_data:
            # point_cloud1 = random_rotation(point_cloud)
            rot = R.random().as_matrix()
            point_cloud1 = np.matmul(point_cloud, rot.T)
        else:
            point_cloud1 = point_cloud
        point_cloud1 = torch.tensor(point_cloud1, dtype=torch.float32)
        # permute points
        shuffled_indices = torch.randperm(self.sampled_points) + 1
        permuted_indices = torch.cat((torch.tensor([0]), shuffled_indices), dim=0)
        point_cloud1 = point_cloud1[permuted_indices]

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

def plot_point_clouds(point_cloud1, point_cloud2, title=""):
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

    # Calculate global min and max values for all axes
    all_points = np.vstack((point_cloud1, point_cloud2))
    min_val = all_points.min()
    max_val = all_points.max()

    fig.update_layout(
        title=title,  # Set the title
        title_y=0.9,  # Adjust the y position of the title
        scene=dict(
            xaxis=dict(title='X', range=[min_val, max_val]),
            yaxis=dict(title='Y', range=[min_val, max_val]),
            zaxis=dict(title='Z', range=[min_val, max_val]),
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

    return sampled_points_with_centroid
def sampleHalfSpacePoints(a, b, c, d, e, count):
    def surface_function(x, y):
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y

    # Generate random points within the range [-1, 1] for both x and y
    x_samples = np.random.uniform(-1, 1, count)
    y_samples = np.random.uniform(-1, 1, count)

    # Evaluate the surface function at the random points
    z_samples = surface_function(x_samples, y_samples)

    # Create an array with the sampled points
    sampled_points = np.column_stack((x_samples, y_samples, z_samples))

    # Concatenate the centroid [0, 0, 0] to the beginning of the array
    centroid = np.array([[0, 0, 0]])
    sampled_points_with_centroid = np.concatenate((centroid, sampled_points), axis=0)
    center_point_idx = np.argsort(np.linalg.norm(sampled_points_with_centroid, axis=1))[np.random.choice(np.arange(-15,0))]
    # center_point_idx = np.argsort(np.linalg.norm(sampled_points_with_centroid, axis=1))[-15]
    sampled_points_with_centroid = sampled_points_with_centroid - sampled_points_with_centroid[center_point_idx, :]
    return sampled_points_with_centroid

def sample_points_on_pyramid(num_samples=40):
    # Define an equilateral triangle as the base
    # base_vertices = np.array([
    #     [0, 0, 0],  # Vertex A
    #     [1, 0, 0],  # Vertex B
    #     [0.5, np.sqrt(3) / 2, 0]  # Vertex C
    # ])
    tri_edge_len = 4 * (np.sqrt(3))
    tri_height = (np.sqrt(3) / 2) * tri_edge_len
    centroid_vertex_len = (2/3) * tri_height
    max_dist_point = np.random.normal(loc=13.21, scale=4.522)
    max_dist_point = np.clip(max_dist_point,4.5,40)
    height = np.sqrt((max_dist_point**2) - (centroid_vertex_len**2)  )
    square_min = -1
    square_max = 1

    # Vertices of an equilateral triangle touching the edges of the square
    # One vertex on the bottom edge, two on the left and right edges
    A = np.array([0, -1,0])  # Middle of bottom edge
    B = np.array([2*(np.sqrt(3)), 1,0])  # Right edge
    C = np.array([-2*(np.sqrt(3)), 1,0])  # Left edge

    # Apex is directly above the center of the triangle
    centroid = (A + B + C) / 3
    apex = np.array([centroid[0], centroid[1], height])

    # Apex is directly above the center of the base
    # apex = np.array([0.5, np.sqrt(3) / 6, height])
    #
    # A, B, C = base_vertices

    # Ensure at least one point is sampled from each face
    N1, N2, N3 = np.random.multinomial(num_samples-3, [1/3, 1/3, 1/3]) + np.array([1,1,1])

    # 1. Sample points on the three faces
    def sample_on_face(P, Q, num_samples_face):
        u = np.random.rand(num_samples_face, 1)
        v = np.random.rand(num_samples_face, 1)
        mask = (u + v) > 1
        u[mask], v[mask] = 1 - u[mask], 1 - v[mask]  # Reflect points that are outside the triangle
        return (1 - u - v) * P + u * Q + v * apex

    face_points_AB = sample_on_face(A, B, N1) - apex
    face_points_BC = sample_on_face(B, C, N2) - apex
    face_points_CA = sample_on_face(C, A, N3) - apex
    center = np.array([0, 0, 0])
    # Combine all sampled points
    sampled_points = np.vstack((center,face_points_AB, face_points_BC, face_points_CA))

    return sampled_points
def generate_room_corner_with_points(N):
    # value = np.random.normal(loc=3.2715, scale=0.8955)
    # value = np.random.normal(loc=4.85, scale=0.8955)
    # value = np.clip(value, 1, 8)
    # value = np.random.normal(loc=1.924, scale=0.41)

    upper_bound1, upper_bound2, upper_bound3 = [
        np.clip(np.random.normal(loc=2.04, scale=0.4), 1, 6) * np.cos(np.radians(45)) for _ in range(3)]

    # upper_bound = 1

    N1, N2, N3 = np.random.multinomial(N-3, [1/3, 1/3, 1/3]) + np.array([1,1,1])
    center = np.array([0, 0, 0])
    points1 = np.stack((np.random.uniform(0, upper_bound1, N1), np.random.uniform(0, upper_bound1, N1), np.zeros(N1)), axis=-1)
    points2 = np.stack((np.zeros(N2), np.random.uniform(0, upper_bound2, N2), -np.random.uniform(0, upper_bound2, N2)), axis=-1)
    points3 = np.stack((np.random.uniform(0, upper_bound3, N3), np.zeros(N3), -np.random.uniform(0, upper_bound3, N3)), axis=-1)
    points = np.vstack((center, points1,points2,points3))
    center_point_idx = np.argsort(np.linalg.norm(points, axis=1))[np.random.choice([0, 1, 2])]
    points = points - points[center_point_idx, :]
    return points
def generate_surfaces_angles_and_sample(N, angle):
    angle_rad = np.radians((180 - angle) / 2)
    # value = np.random.normal(loc=3.2715, scale=0.8955)
    # value = np.random.normal(loc=3.42, scale=0.8955)
    # value = np.random.normal(loc=1.924, scale=0.41)
    value = np.random.normal(loc=2, scale=0.4)
    # value = np.clip(value, 1, 8)
    value = np.clip(value, 1, 6)
    upper_bound_y = np.clip(np.random.normal(loc=1, scale=0.3), min(0.2, value), value - 0.1)
    upper_bound_x = np.sqrt( ( value**2 ) - ( upper_bound_y**2 ) ) * np.cos(angle_rad)
    # upper_bound = 1
    # 1. Generate a random angle between 0 and 30 degrees


    # 2. Compute the slopes (m1 and m2) for the surfaces
    m1 = np.tan(angle_rad)  # slope for the left surface (x < 0)
    m2 = -m1  # slope for the right surface (x >= 0)

    alpha = np.clip(np.random.normal(loc=0.5, scale=0.2), 0.1, 0.9)
    N1, N2 = np.random.multinomial(N - 2, [alpha, 1-alpha]) + np.array([1, 1])
    # 3. Generate N random points in the square [-1, 1] x [-1, 1]
    x_coords_neg = np.random.uniform(-upper_bound_x, 0, N1)
    x_coords_pos = np.random.uniform(0, upper_bound_x, N2)
    x_coords = np.concatenate((x_coords_neg,x_coords_pos))
    y_coords = np.random.uniform(-upper_bound_y, upper_bound_y, N)

    # 4. Calculate the corresponding z values based on the surfaces
    z_coords = np.where(x_coords < 0, m1 * x_coords, m2 * x_coords)
    # z_coords = np.abs(x_coords)

    # 5. Stack the points into a single array
    points = np.stack((x_coords, y_coords, z_coords), axis=-1) - np.array([0, (np.random.uniform(-(upper_bound_y / 2 ), (upper_bound_y / 2 ))), 0])
    center = np.array([0,0,0])
    points = np.vstack((center,points))
    center_point_idx = np.argsort(np.linalg.norm(points, axis=1))[np.random.choice([0, 1, 2])]
    points = points - points[center_point_idx, :]
    return points
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
def random_rotation(point_cloud):
    rot = R.random().as_matrix()
    rot_mat = torch.tensor(rot, dtype=torch.float32)
    rotated_point_cloud = torch.matmul(point_cloud, rot_mat.T)
    return rotated_point_cloud