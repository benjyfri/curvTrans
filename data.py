from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from utils import createLPEembedding, positional_encoding_nerf
import plotly.express as px
import random
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

from sklearn.neighbors import NearestNeighbors
class BasicPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, args):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.point_clouds_group = self.hdf5_file['point_clouds']
        self.num_point_clouds = len(self.point_clouds_group)
        self.pcls_per_class = self.num_point_clouds // 4
        self.indices = list(range(self.num_point_clouds))
        self.std_dev = args.std_dev
        self.clip = args.clip
        self.rotate_data = args.rotate_data
        self.contr_loss_weight = args.contr_loss_weight
        self.sampled_points = args.sampled_points
    def __len__(self):
        return self.num_point_clouds

    def __getitem__(self, idx):
        point_cloud_name = f"point_cloud_{self.indices[idx]}"
        # Load metadata from attributes
        info = {key: self.point_clouds_group[point_cloud_name].attrs[key] for key in
                    self.point_clouds_group[point_cloud_name].attrs}

        class_label = info['class']
        angle = info['angle']
        radius = info['radius']
        bias = 0.25
        length = 1
        point_cloud = samplePcl(angle,radius,class_label,self.sampled_points, length, bias, info)

        point_cloud1 = torch.tensor(point_cloud, dtype=torch.float32)
        if self.rotate_data:
            rot_orig, point_cloud1 = random_rotation(point_cloud1)

        # permute points
        shuffled_indices = torch.randperm(self.sampled_points) + 1
        permuted_indices = torch.cat((torch.tensor([0]), shuffled_indices), dim=0)
        point_cloud1 = point_cloud1[permuted_indices]

        #Add noise to point cloud
        if self.std_dev != 0:
            noise = torch.normal(0, self.std_dev, size=point_cloud1.shape, dtype=torch.float32, device=point_cloud1.device)
            noise = torch.clamp(noise, min=-self.clip, max=self.clip)
            point_cloud1 = point_cloud1 + noise
            point_cloud1 = point_cloud1 - point_cloud1[0,:]

        if self.contr_loss_weight  != 0:

            if class_label==0:
                min_curve_diff = 0.05
                max_curve_diff = 0.15
            else:
                min_curve_diff = 0.1
                max_curve_diff = 0.2

            # if radius == 0 and angle == 0:
            #
            #     a,b,c,d,e = info['a'], info['b'], info['c'], info['d'],info['e']
            #     K_orig = (4 * ((a) * (b)) - (((c) ** 2))) / ((1 + (d) ** 2 + (e) ** 2) ** 2)
            #     H_orig = ((a) * (1 + (e) ** 2) - (d) * (e) * (c) + (b) * (1 + (d) ** 2)) / (
            #                 (((d) ** 2) + ((e) ** 2) + 1) ** 1.5)
            #
            #     discriminant_orig = H_orig ** 2 - K_orig
            #     k1_orig = H_orig + np.sqrt(discriminant_orig)
            #     k2_orig = H_orig - np.sqrt(discriminant_orig)
            #     count=0
            #     while True:
            #         noise_to_add = np.random.normal(0, 0.1, 5)
            #         K_cont = (4 * ((a + noise_to_add[0]) * (b + noise_to_add[1])) - (
            #         ((c + noise_to_add[2]) ** 2))) / (
            #                              (1 + (d + noise_to_add[3]) ** 2 + (e + noise_to_add[4]) ** 2) ** 2)
            #         H_cont = ((a + noise_to_add[0]) * (1 + (e + noise_to_add[4]) ** 2) - (
            #                     d + noise_to_add[3]) * (e + noise_to_add[4]) * (
            #                               c + noise_to_add[2]) + (b + noise_to_add[1]) * (
            #                               1 + (d + noise_to_add[3]) ** 2)) / ((((d + noise_to_add[
            #             3]) ** 2) + ((e + noise_to_add[4]) ** 2) + 1) ** 1.5)
            #         discriminant_cont = H_cont ** 2 - K_cont
            #         k1_cont = H_cont + np.sqrt(discriminant_cont)
            #         k2_cont = H_cont - np.sqrt(discriminant_cont)
            #
            #         temp_max_diff = abs(k1_cont-k1_orig)
            #         temp_min_diff = abs(k2_cont-k2_orig)
            #
            #         if (( (temp_max_diff > min_curve_diff) or (temp_min_diff > min_curve_diff)) and
            #                 ((temp_max_diff < max_curve_diff) and (temp_min_diff < max_curve_diff))):
            #             a = info['a'] + noise_to_add[0]
            #             b = info['b'] + noise_to_add[1]
            #             c = info['c'] + noise_to_add[2]
            #             d = info['d'] + noise_to_add[3]
            #             e = info['e'] + noise_to_add[4]
            #             break
            #         count += 1
            #     contrastive_point_cloud = samplePoints(a, b, c, d, e, count=self.sampled_points, label=class_label)
            # else:

            positive_point_cloud = point_cloud
            if class_label != 4:
                positive_point_cloud = samplePcl(angle, radius, class_label, self.sampled_points, length, bias, info)

            contrastive_point_cloud = torch.tensor(contrastive_point_cloud, dtype=torch.float32)
            neg_rot,contrastive_point_cloud = random_rotation(contrastive_point_cloud)

            point_cloud2 = torch.tensor(positive_point_cloud, dtype=torch.float32)
            pos_rot, point_cloud2 = random_rotation(point_cloud2)


            if self.std_dev != 0:
                noise = torch.normal(0, self.std_dev, size=point_cloud2.shape, dtype=torch.float32,
                                     device=point_cloud2.device)
                noise = torch.clamp(noise, min=-self.clip, max=self.clip)
                point_cloud2 = point_cloud2 + noise
                point_cloud2 = point_cloud2 - point_cloud2[0, :]


                contrastive_noise = torch.normal(0, self.std_dev, size=contrastive_point_cloud.shape, dtype=torch.float32,
                                     device=contrastive_point_cloud.device)
                contrastive_noise = torch.clamp(contrastive_noise, min=-self.clip, max=self.clip)
                contrastive_point_cloud = contrastive_point_cloud + contrastive_noise
                contrastive_point_cloud = contrastive_point_cloud - contrastive_point_cloud[0, :]
        else:
            point_cloud2 = torch.tensor((0))
            contrastive_point_cloud = torch.tensor((0))


    #     if class_label in [2]:
    #         plot_point_clouds(point_cloud1@rot_orig, np.load("10_pcl_noisy.npy")*2, f'class: {class_label}, {k1_orig}, {k2_orig}')
    # #         plot_point_clouds(point_cloud1,point_cloud1, f'class: {class_label}, k1: {k1_orig}, k2: {k2_orig}')
    # #         # plotFunc(info['a'], info['b'], info['c'], info['d'], info['e'],point_cloud1@rot_orig)
    #         a=1
    #         plot_point_clouds(point_cloud1@rot_orig, point_cloud2@pos_rot, f'pos; class: {class_label}')
    #         plot_point_clouds(point_cloud1@rot_orig, contrastive_point_cloud@neg_rot,
    # f'neg; class: {class_label}, orig_k1:{k1_orig:.2f}, orig_k2:{k2_orig:.2f}||\n'
    #         f'cont_k1:{k1_cont:.2f}, cont_k2:{k2_cont:.2f}')
    #         a=1

        # return {"point_cloud": point_cloud1, "point_cloud2": point_cloud2, "contrastive_point_cloud":contrastive_point_cloud, "info": info, "count": count}
        return {"point_cloud": point_cloud1, "point_cloud2": point_cloud2, "contrastive_point_cloud":contrastive_point_cloud, "info": info}

def samplePcl(angle,radius,class_label,sampled_points, length, bias, info):
    if angle != 0:
        if class_label == 1:
            point_cloud = sample_pyramid(length, sampled_points, np.pi, bias=bias)
        if class_label == 2:
            point_cloud = generate_surfaces_angles_and_sample(sampled_points, angle, bias=bias)

    elif radius != 0:

        if class_label == 1:
            point_cloud = sample_sphere_point_cloud(radius=radius, num_of_points=sampled_points)
        if class_label == 2:
            point_cloud = sample_cylinder_point_cloud(radius=radius, length=length, num_of_points=sampled_points,
                                                      bias=bias)
    else:
        point_cloud = samplePoints(info['a'], info['b'], info['c'], info['d'], info['e'], count=sampled_points,bias=bias)
    if class_label == 4:
        point_cloud = sampleHalfSpacePoints(point_cloud)
    return point_cloud

def sampleContrastivePcl(angle,radius,class_label,sampled_points, length, bias, info,min_curve_diff, max_curve_diff):
    curve_diff = np.random.uniform(min_curve_diff, max_curve_diff)
    if angle != 0:
        curve = (np.sqrt(2) * np.sin(np.pi - np.radians(angle))) / np.sqrt( 1 - np.cos(np.pi - np.radians(angle)) )
        if class_label == 1:
            contrastive_point_cloud = sample_pyramid(length, sampled_points, np.pi, bias=bias)
        if class_label == 2:
            contrastive_point_cloud = generate_surfaces_angles_and_sample(sampled_points, angle, bias=bias)

    elif radius != 0:

        if class_label == 1:
            contrastive_point_cloud = sample_sphere_point_cloud(radius=radius, num_of_points=sampled_points)
        if class_label == 2:
            contrastive_point_cloud = sample_cylinder_point_cloud(radius=radius, length=length, num_of_points=sampled_points,
                                                      bias=bias)
    else:
        a, b, c, d, e = info['a'], info['b'], info['c'], info['d'], info['e']
        K_orig = (4 * ((a) * (b)) - (((c) ** 2))) / ((1 + (d) ** 2 + (e) ** 2) ** 2)
        H_orig = ((a) * (1 + (e) ** 2) - (d) * (e) * (c) + (b) * (1 + (d) ** 2)) / (
                (((d) ** 2) + ((e) ** 2) + 1) ** 1.5)

        discriminant_orig = H_orig ** 2 - K_orig
        k1_orig = H_orig + np.sqrt(discriminant_orig)
        k2_orig = H_orig - np.sqrt(discriminant_orig)
        count = 0
        while True:
            noise_to_add = np.random.normal(0, 0.1, 5)
            K_cont = (4 * ((a + noise_to_add[0]) * (b + noise_to_add[1])) - (
                ((c + noise_to_add[2]) ** 2))) / (
                             (1 + (d + noise_to_add[3]) ** 2 + (e + noise_to_add[4]) ** 2) ** 2)
            H_cont = ((a + noise_to_add[0]) * (1 + (e + noise_to_add[4]) ** 2) - (
                    d + noise_to_add[3]) * (e + noise_to_add[4]) * (
                              c + noise_to_add[2]) + (b + noise_to_add[1]) * (
                              1 + (d + noise_to_add[3]) ** 2)) / ((((d + noise_to_add[
                3]) ** 2) + ((e + noise_to_add[4]) ** 2) + 1) ** 1.5)
            discriminant_cont = H_cont ** 2 - K_cont
            k1_cont = H_cont + np.sqrt(discriminant_cont)
            k2_cont = H_cont - np.sqrt(discriminant_cont)

            temp_max_diff = abs(k1_cont - k1_orig)
            temp_min_diff = abs(k2_cont - k2_orig)

            if (((temp_max_diff > min_curve_diff) or (temp_min_diff > min_curve_diff)) and
                    ((temp_max_diff < max_curve_diff) and (temp_min_diff < max_curve_diff))):
                a = info['a'] + noise_to_add[0]
                b = info['b'] + noise_to_add[1]
                c = info['c'] + noise_to_add[2]
                d = info['d'] + noise_to_add[3]
                e = info['e'] + noise_to_add[4]
                break
            count += 1
        contrastive_point_cloud = samplePoints(a, b, c, d, e, count=sampled_points, label=class_label)
    if class_label == 4:
        contrastive_point_cloud = sampleHalfSpacePoints(contrastive_point_cloud)
    return contrastive_point_cloud


def rotatePCLToCanonical(point_cloud, centroid, k):
    num_points = point_cloud.shape[0]

    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    # Calculate the rotation matrix to align the normal with the z-axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal_at_centroid, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal_at_centroid, z_axis)

    if s == 0:  # Already aligned with z-axis
        rotation_matrix = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

    # Rotate the point cloud and centroid
    rotated_point_cloud = point_cloud @ rotation_matrix.T
    rotated_centroid = centroid @ rotation_matrix.T

    return rotated_point_cloud, rotated_centroid, rotation_matrix

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

def plot_point_clouds(point_cloud1, point_cloud2=None, title=""):
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
    all_points = point_cloud1
    if point_cloud2 is not None:
        fig.add_trace(go.Scatter3d(
            x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
            mode='markers', marker=dict(color='blue'),opacity=0.8, name='Point Cloud 2'
        ))
        all_points = np.vstack((point_cloud1, point_cloud2))
    # Add a separate trace for the point (0, 0, 0) in bright pink
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(color='rgb(255, 105, 180)'), name='Origin (0, 0, 0)'
    ))

    # # Calculate global min and max values for all axes
    # all_points = np.vstack((point_cloud1, point_cloud2))
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

def samplePoints(a, b, c, d, e, count, center_point=np.array([0,0,0]), bias=0.0):
    def surface_function(x, y):
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y

    add = np.random.uniform(-bias, bias)
    x_size = 1 + add
    y_size = 1 - add

    # Generate random points within the range [-1, 1] for both x and y
    x_samples = np.random.uniform(-x_size, x_size, count) + center_point[0]
    y_samples = np.random.uniform(-y_size, y_size, count) + center_point[1]

    # Evaluate the surface function at the random points
    z_samples = surface_function(x_samples, y_samples)

    # Create an array with the sampled points
    sampled_points = np.column_stack((x_samples, y_samples, z_samples))

    # Concatenate the centroid [0, 0, 0] to the beginning of the array
    centroid = np.expand_dims(center_point, axis=0)
    sampled_points_with_centroid = np.concatenate((centroid, sampled_points), axis=0)

    return sampled_points_with_centroid
def sampleHalfSpacePoints(sampled_points_with_centroid):
    center_point_idx = np.argsort(np.linalg.norm(sampled_points_with_centroid, axis=1))[np.random.choice(np.arange(-10,0))]
    sampled_points_with_centroid = sampled_points_with_centroid - sampled_points_with_centroid[center_point_idx, :]
    sampled_points_with_centroid[center_point_idx, :] = (sampled_points_with_centroid[0, :]).copy()
    sampled_points_with_centroid[0, :] = np.array([[0, 0, 0]])
    return sampled_points_with_centroid


def generate_room_corner_with_points(n_points, bias=0.0):
    upper_bound1, upper_bound2, upper_bound3 = np.random.uniform(1 - bias, 1 + bias, 3)

    N1, N2, N3 = np.random.multinomial(n_points-3, [1/3, 1/3, 1/3]) + np.array([1,1,1])
    center = np.array([0, 0, 0])
    points1 = np.stack((np.random.uniform(0, upper_bound1, N1), np.random.uniform(0, upper_bound1, N1), np.zeros(N1)), axis=-1)
    points2 = np.stack((np.zeros(N2), np.random.uniform(0, upper_bound2, N2), -np.random.uniform(0, upper_bound2, N2)), axis=-1)
    points3 = np.stack((np.random.uniform(0, upper_bound3, N3), np.zeros(N3), -np.random.uniform(0, upper_bound3, N3)), axis=-1)
    points = np.vstack((center, points1,points2,points3))

    center_point_idx = np.argsort(np.linalg.norm(points, axis=1))[np.random.choice([0, 1, 2])]
    points = points - points[center_point_idx, :]

    # must fix - is wrong! first point must be centered!
    points[center_point_idx, :] = (points[0, :]).copy()
    points[0, :] = np.array([[0, 0, 0]])
    return points
def generate_surfaces_angles_and_sample(N, angle, bias=0.0):
    angle_rad = np.radians((180 - angle) / 2)
    value = np.random.uniform(0.5-bias, 0.5+bias)
    upper_bound_y = np.random.uniform(0.5, 0.5+bias)
    upper_bound_x = np.random.uniform(0.5, 0.5+bias)
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

    # must fix - is wrong! first point must be centered!
    points[center_point_idx, :] = (points[0, :]).copy()
    points[0, :] = np.array([[0, 0, 0]])
    return points
def plotFunc(a, b, c, d, e,sampled_points):
    # Create a grid of points for the surface
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
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

def sample_cylinder_point_cloud(radius, length, num_of_points,top_half=True, bias=0.0):
    length = np.random.uniform(length-bias, length+bias)
    if top_half:
        theta = np.random.uniform(-0.5*np.pi, 0.5*np.pi, num_of_points)
    else:
        theta = np.random.uniform(0.5*np.pi, 1.5*np.pi, num_of_points)

    # Sample random heights (z) along the length of the cylinder
    x = np.random.uniform(0, length, num_of_points)

    # Compute the (x, y) coordinates on the circular cross-section
    z = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Stack the coordinates into a (num_of_points, 3) array
    point_cloud = np.stack((x, y, z), axis=-1)

    centering = radius if top_half else -radius
    points = np.vstack([(np.array([0, 0, 0])).reshape(1, 3), point_cloud - (np.array([length/2,0,centering]))])
    # center_point_idx = np.argsort(np.linalg.norm(points, axis=1))[np.random.choice([0, 1, 2])]
    # points = points - points[center_point_idx, :]
    # points[center_point_idx, :] = (points[0, :]).copy()
    # points[0, :] = np.array([[0, 0, 0]])
    return points


def sample_sphere_point_cloud(radius, num_of_points, top_half=True):
    theta = np.random.uniform(0, 2 * np.pi, num_of_points)

    if top_half:
        phi = np.random.uniform(0, np.pi / 2, num_of_points)  # Top hemisphere
    else:
        phi = np.random.uniform(np.pi / 2, np.pi, num_of_points)  # Bottom hemisphere

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Stack the coordinates into a (num_of_points, 3) array
    point_cloud = np.stack((x, y, z), axis=-1)

    centering = radius if top_half else -radius
    points = np.vstack([(np.array([0, 0, 0])).reshape(1, 3), point_cloud - (np.array([0,0,centering]))])

    center_point_idx = np.argsort(np.linalg.norm(points, axis=1))[np.random.choice([0, 1, 2])]
    points = points - points[center_point_idx, :]
    points[center_point_idx, :] = (points[0, :]).copy()
    points[0, :] = np.array([[0, 0, 0]])
    return points

def equilateral_triangle_coordinates(h, a):
    beta = np.tan(a / 2)
    edge_len = ( np.sqrt( ( 12 * (beta**2) ) / ( 3 - beta**2) ) ) * h
    r = edge_len / np.sqrt(3)

    # Calculate the 2D coordinates of the vertices of an equilateral triangle
    # Centered at (0, 0) in the x-y plane
    vertices = [np.array([0,0,0])]
    for i in range(3):
        angle = 2 * np.pi * i / 3  # 120-degree steps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append((x, y, h))

    return np.array(vertices)


# def sample_pyramid(base_vertices, n_points, gauss_curv):
def sample_pyramid(h, n_points, gauss_curv, bias=0.0):
    sum_of_tip_angles = gauss_curv - 2 * np.pi

    base_vertices = equilateral_triangle_coordinates(h, sum_of_tip_angles / 3 )[1:,:]

    # Define the pyramid tip
    tip = np.array([0, 0, 0])
    base_vertices[0] *= (np.random.uniform(1-bias, 1+bias))
    base_vertices[1] *= (np.random.uniform(1-bias, 1+bias))
    base_vertices[2] *= (np.random.uniform(1-bias, 1+bias))
    # Define the three triangular side faces
    triangles = np.array([
        [tip, base_vertices[0], base_vertices[1]],  # Side 1
        [tip, base_vertices[1], base_vertices[2]],  # Side 2
        [tip, base_vertices[2], base_vertices[0]]  # Side 3
    ])

    # Calculate areas of the triangular sides
    def triangle_area(v1, v2, v3):
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    areas = np.array([
        triangle_area(*triangles[i]) for i in range(3)
    ])
    total_area = areas.sum()

    # Allocate points proportionally to triangle areas
    points_per_triangle = np.random.multinomial(n_points, areas / total_area)

    # Generate barycentric coordinates for surface sampling
    u = np.random.rand(n_points)
    v = np.random.rand(n_points)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)

    # Repeat triangles based on the number of points per triangle
    triangle_indices = np.repeat(np.arange(3), points_per_triangle)
    selected_triangles = triangles[triangle_indices]

    # Extract vertices for selected triangles
    v1 = selected_triangles[:, 0]
    v2 = selected_triangles[:, 1]
    v3 = selected_triangles[:, 2]

    # Compute sampled points using barycentric coordinates
    sampled_points = u[:, None] * v1 + v[:, None] * v2 + w[:, None] * v3

    points = np.vstack([tip.reshape(1,3), sampled_points])

    center_point_idx = np.argsort(np.linalg.norm(points, axis=1))[np.random.choice([0, 1, 2])]
    points = points - points[center_point_idx, :]
    points[center_point_idx, :] = (points[0, :]).copy()
    points[0, :] = np.array([[0, 0, 0]])
    return points


def random_rotation(point_cloud):
    rot = R.random().as_matrix()
    rot_mat = torch.tensor(rot, dtype=torch.float32)
    rotated_point_cloud = torch.matmul(point_cloud, rot_mat.T)
    return rot, rotated_point_cloud

if __name__ == '__main__':
    sampleContrastivePcl(angle=45,radius=0,class_label=0,sampled_points=0, length=0, bias=0, info=0,min_curve_diff=0, max_curve_diff=0)
    plot_point_clouds(sample_cylinder_point_cloud(2, 1, 250, False, 0.5))
    a =1