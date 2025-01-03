from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from models import shapeClassifier
from data import BasicPointCloudDataset
from scipy.spatial.transform import Rotation
import glob
import h5py
import os
import plotly.graph_objects as go
from train import configArgsPCT
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
from sklearn.neighbors import NearestNeighbors
import cProfile
import pstats
from scipy.spatial import cKDTree
from plotting_functions import *
from experiments_utils import *


def farthest_point_sampling(point_cloud, k):
    N, _ = point_cloud.shape

    # Array to hold indices of sampled points
    sampled_indices = np.zeros(k, dtype=int)

    # Initialize distances to a large value
    distances = np.full(N, np.inf)

    # Randomly select the first point
    current_index = np.random.randint(N)
    sampled_indices[0] = current_index

    for i in range(1, k):
        # Update distances to the farthest point selected so far
        current_point = point_cloud[current_index]
        new_distances = np.linalg.norm(point_cloud - current_point, axis=1)
        distances = np.minimum(distances, new_distances)

        # Select the point that has the maximum distance to the sampled points
        current_index = np.argmax(distances)
        sampled_indices[i] = current_index

    return sampled_indices
def get_k_nearest_neighbors(point_cloud, k):
    """
    Returns the k nearest neighbors for each point in the point cloud.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (pcl_size, 3)
        k (int): Number of nearest neighbors to return

    Returns:
        np.ndarray: Array of shape (1, 3, pcl_size, k) containing the k nearest neighbors for each point
    """
    pcl_size = point_cloud.shape[0]
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(point_cloud)
    distances, indices = neigh.kneighbors(point_cloud)

    neighbors_centered = np.empty((1, 3, pcl_size, k), dtype=point_cloud.dtype)
    neighbors_non_centered = np.empty((1, 3, pcl_size, k), dtype=point_cloud.dtype)
    # Each point cloud should be centered around first point which is at the origin
    for i in range(pcl_size):
        orig = point_cloud[indices[i, :]] - point_cloud[i,:]
        neighbors_centered[0, :, i, :] = orig.T
        neighbors_non_centered[0, :, i, :] = (point_cloud[indices[i, :]]).T

    return neighbors_centered, neighbors_non_centered
def get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k):
    """
    Returns the k nearest neighbors for each point in the point cloud.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (pcl_size, 3)
        k (int): Number of nearest neighbors to return

    Returns:
        np.ndarray: Array of shape (1, 3, pcl_size, k) containing the k nearest neighbors for each point
    """
    pcl_size = pcl_interest.shape[0]
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(pcl_src)
    distances, indices = neigh.kneighbors(pcl_interest)

    neighbors_centered = np.empty((1, 3, pcl_size, k), dtype=pcl_src.dtype)
    # Each point cloud should be centered around first point which is at the origin
    for i in range(pcl_size):
        orig = pcl_src[indices[i, :]] - pcl_interest[i,:]
        if not (np.array_equal(orig[0,], np.array([0,0,0]))):
            orig = np.vstack([np.array([0,0,0]), orig])[:-1]
        neighbors_centered[0, :, i, :] = orig.T

    return neighbors_centered
def find_mean_diameter_for_specific_coordinate(specific_coordinates):
    pairwise_distances = torch.cdist(specific_coordinates.unsqueeze(2), specific_coordinates.unsqueeze(2))
    largest_dist = pairwise_distances.view(specific_coordinates.shape[0], -1).max(dim=1).values
    mean_distance = torch.mean(largest_dist)
    return mean_distance

def checkOnShapes(model_name=None, input_data=None, args_shape=None, scaling_factor=None):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'models_weights/{model_name}.pt'))
    model.eval()
    neighbors_centered, neighbors_non_centered = get_k_nearest_neighbors(input_data, 41)
    aaa=get_k_nearest_neighbors_diff_pcls(input_data, input_data, k=41)
    src_knn_pcl = torch.tensor(neighbors_centered)
    x_scale_src = find_mean_diameter_for_specific_coordinate(src_knn_pcl[0,0,:,:])
    y_scale_src = find_mean_diameter_for_specific_coordinate(src_knn_pcl[0,1,:,:])
    z_scale_src = find_mean_diameter_for_specific_coordinate(src_knn_pcl[0,2,:,:])
    scale = torch.mean(torch.stack((x_scale_src, y_scale_src, z_scale_src), dim=0))
    #scale KNN point clouds to be of size 1
    src_knn_pcl = src_knn_pcl / scale
    #use 23 as it is around the size of the synthetic point clouds
    src_knn_pcl = scaling_factor * src_knn_pcl
    output = model(src_knn_pcl.permute(2,1,0,3))
    return output
def load_data(partition='test', divide_data=1):
    BASE_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes'
    DATA_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes\\data'
    # DATA_DIR = r'/content/curvTrans/bbsWithShapes/data'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        print(f'++++++++{h5_name}++++++++')
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    # return all_data, all_label
    size = len(all_label)
    return all_data[:(int)(size/divide_data),:,:], all_label[:(int)(size/divide_data),:]

def checkData():
    args = configArgsPCT()
    args.std_dev = 0.05
    args.rotate_data = 1
    # args.batch_size = 40000
    max_list_x = []
    min_list_x = []
    max_list_y = []
    min_list_y = []
    max_list_z = []
    min_list_z = []
    train_dataset = BasicPointCloudDataset(file_path="train_surfaces_40_stronger_boundaries.h5", args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    with tqdm(train_dataloader) as tqdm_bar:
        for batch in tqdm_bar:
            pcl = batch['point_cloud']
            pcl = pcl.reshape(pcl.shape[0]*pcl.shape[1], -1)
            max_list_x.append(torch.max(pcl[:,0]).item())
            min_list_x.append(torch.min(pcl[:,0]).item())
            max_list_y.append(torch.max(pcl[:,1]).item())
            min_list_y.append(torch.min(pcl[:,1]).item())
            max_list_z.append(torch.max(pcl[:,2]).item())
            min_list_z.append(torch.min(pcl[:,2]).item())
            print('yay')
    print(f'-----------X--------------')
    print(f'MIN mean: {np.mean(min_list_x)}')
    print(f'MIN median: {np.median(min_list_x)}')
    print(f'MIN MIN: {np.min(min_list_x)}')
    print(f'MAX mean: {np.mean(max_list_x)}')
    print(f'MAX median: {np.median(max_list_x)}')
    print(f'MAX MAX: {np.max(max_list_x)}')
    print(f'-----------Y--------------')
    print(f'MIN mean: {np.mean(min_list_y)}')
    print(f'MIN median: {np.median(min_list_y)}')
    print(f'MIN MIN: {np.min(min_list_y)}')
    print(f'MAX mean: {np.mean(max_list_y)}')
    print(f'MAX median: {np.median(max_list_y)}')
    print(f'MAX MAX: {np.max(max_list_y)}')
    print(f'-----------Z--------------')
    print(f'MIN mean: {np.mean(min_list_z)}')
    print(f'MIN median: {np.median(min_list_z)}')
    print(f'MIN MIN: {np.min(min_list_z)}')
    print(f'MAX mean: {np.mean(max_list_z)}')
    print(f'MAX median: {np.median(max_list_z)}')
    print(f'MAX MAX: {np.max(max_list_z)}')
def visualizeShapesWithEmbeddings(model_name=None, args_shape=None, scaling_factor=None):
    pcls, label = load_data()
    shapes = [86, 174, 179]
    # shapes = [86]
    for k in shapes:
        pointcloud = pcls[k][:]
        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors = checkOnShapes(model_name=model_name,
                                    input_data=pointcloud, args_shape=args_shape, scaling_factor=scaling_factor)
        colors = colors.detach().cpu().numpy()
        colors = colors[:,:4]

        colors_normalized = colors.copy()
        colors_normalized[:, 0] = ((colors[:, 0] - colors[:, 0].min()) / (
                    colors[:, 0].max() - colors[:, 0].min())) * 255
        colors_normalized[:, 1] = ((colors[:, 1] - colors[:, 1].min()) / (
                    colors[:, 1].max() - colors[:, 1].min())) * 255
        colors_normalized[:, 2] = ((colors[:, 2] - colors[:, 2].min()) / (
                    colors[:, 2].max() - colors[:, 2].min())) * 255
        colors_normalized[:, 3] = ((colors[:, 3] - colors[:, 3].min()) / (
                    colors[:, 3].max() - colors[:, 3].min())) * 255
        colors_normalized = np.clip(colors_normalized, 0, 255).astype(np.uint8)
        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        data_rgb = [
            go.Scatter3d(
                x=pointcloud[:, 0],
                y=pointcloud[:, 1],
                z=pointcloud[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    opacity=0.8,
                    color=['rgb(' + ', '.join(map(str, rgb)) + ')' for rgb in colors_normalized],  # Set RGB values
                ),
                name='RGB Embeddings'
            )
        ]

        # Your existing code

        # Plotting the RGB embeddings separately
        fig_rgb = go.Figure(data=data_rgb, layout=layout)
        fig_rgb.show()

        # Plot the maximum value embedding with specified colors
        max_embedding_index = np.argmax(colors, axis=1)
        max_embedding_colors = np.array(['red', 'blue', 'green', 'pink'])[max_embedding_index]

        data_max_embedding = []
        colors_shape = ['red', 'blue', 'green', 'pink']
        names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
        for color, name in zip(colors_shape, names):
            indices = np.where(max_embedding_colors == color)[0]
            data_max_embedding.append(
                go.Scatter3d(
                    x=pointcloud[indices, 0],
                    y=pointcloud[indices, 1],
                    z=pointcloud[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        opacity=0.8,
                        color=color
                    ),
                    name=f'Max Value Embedding - {name}'
                )
            )
        fig_max_embedding = go.Figure(data=data_max_embedding, layout=layout)
        fig_max_embedding.show()

        # # Plot the maximum value embedding with specified colors
        # max_embedding_index = surface_labels
        # max_embedding_colors = np.array(['red', 'blue', 'green', 'pink'])[max_embedding_index]
        #
        # data_max_embedding = []
        # colors_shape = ['red', 'blue', 'green', 'pink']
        # names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
        # for color, name in zip(colors_shape, names):
        #     indices = np.where(max_embedding_colors == color)[0]
        #     data_max_embedding.append(
        #         go.Scatter3d(
        #             x=pointcloud[indices, 0],
        #             y=pointcloud[indices, 1],
        #             z=pointcloud[indices, 2],
        #             mode='markers',
        #             marker=dict(
        #                 size=2,
        #                 opacity=0.8,
        #                 color=color
        #             ),
        #             name=f'Max Value Embedding - {name}'
        #         )
        #     )
        # fig_max_embedding = go.Figure(data=data_max_embedding, layout=layout)
        # fig_max_embedding.show()


def find_max_difference_indices(array, k=200):
    # Find the maximum values along axis 1
    max_values = np.max(array, axis=1)
    max_indices = np.argmax(array, axis=1)
    # Find the maximum values ignoring the maximum value itself along axis 1
    max_values_without_max = np.max(np.where(array == max_values[:, np.newaxis], -np.inf, array), axis=1)
    diff_from_max = (max_values - max_values_without_max)
    good_class_indices = np.argpartition(diff_from_max, -k)[-k:]
    if len(good_class_indices) != k:
        raise Exception(f'Wrong size! k = {k}, actual size = {len(good_class_indices)}')
    # sorted_indices = np.argsort(max_values)
    # k_largest_indices = sorted_indices[-k:]
    # good_class_indices = k_largest_indices
    return max_values, max_indices, diff_from_max, good_class_indices

def findRotTrans(model_name=None, args_shape=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None):
    pcls, label = load_data()
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    shapes = np.arange(pcls.shape[0])
    shapes = np.arange(1000) # or any other range or selection of indices
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    iter_2_ransac_convergence = []
    num_of_inliers = []
    shape_size_list = []
    shortest_dist_list = []
    avg_dist_list = []
    dist_from_orig = []
    # shapes = [86, 162, 174, 176, 179]
    shapes = np.arange(1000)
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_2,args_shape=args_shape, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue
        #
        # max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
        #                                                                                                  k=200)
        # max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
        #                                                                                                  k=200)

        good_class_indices_1 = farthest_point_sampling(emb_1[:,:4],k=200)
        good_class_indices_2 = farthest_point_sampling(emb_2[:,:4],k=200)

        best_point_desc_pcl1 = emb_1[good_class_indices_1, :]
        best_point_desc_pcl2 = emb_2[good_class_indices_2, :]

        source_indices, target_indices = find_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
                                                             num_neighbors=40, max_non_unique_correspondences=max_non_unique_correspondences)


        chosen_indices_pcl1 = good_class_indices_1[source_indices]
        chosen_indices_pcl2 = good_class_indices_2[target_indices]
        chosen_points_1 = noisy_pointcloud_1[chosen_indices_pcl1, :]
        chosen_points_2 = noisy_pointcloud_2[chosen_indices_pcl2, :]

        amount_same_index = np.sum(chosen_indices_pcl1 == chosen_indices_pcl2)
        original_points_chosen = pointcloud[chosen_indices_pcl1, :]
        original_corresponding_points = pointcloud[chosen_indices_pcl2, :]
        pairwise_distances = cdist(pointcloud, pointcloud)
        shape_size = np.max(pairwise_distances)
        shape_size_list.append(shape_size)
        correspondences_dist = (pairwise_distances[chosen_indices_pcl1,chosen_indices_pcl2])
        dist_from_orig.append(np.mean(correspondences_dist))
        mask = ~np.eye(pairwise_distances.shape[0], dtype=bool)
        flattened_arr = pairwise_distances[mask]
        avg_dist_list.append(np.mean(flattened_arr))
        shortest_dist_list.append(np.min(flattened_arr))
        centered_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :] - np.mean(noisy_pointcloud_2)
        best_rotation, inliers, best_iter = ransac(centered_points_1, centered_points_2, max_iterations=1000, threshold=0.1,
                                  min_inliers=10)
        num_of_inliers.append(len(inliers))
        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        losses.append(loss)
        iter_2_ransac_convergence.append(best_iter)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if loss > smallest_loss:
            worst_losses[index_of_smallest_loss] = (loss, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': chosen_points_1,
                'chosen_points_2': chosen_points_2,
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })

    return worst_losses, losses, num_of_inliers, iter_2_ransac_convergence, shape_size_list, dist_from_orig, shortest_dist_list, avg_dist_list

def find_closest_points(embeddings1, embeddings2, num_neighbors=40, max_non_unique_correspondences=3):
    classification_1 = np.argmax(embeddings1[:,:4], axis=1)
    classification_2 = np.argmax(embeddings2[:,:4], axis=1)

    # Initialize NearestNeighbors instance
    # nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2[:,4:])
    # nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2[:,:4])

    # Find the indices and distances of the closest points in embeddings2 for each point in embeddings1
    # distances, indices = nbrs.kneighbors(embeddings1[:,4:])
    # distances, indices = nbrs.kneighbors(embeddings1)
    distances, indices = nbrs.kneighbors(embeddings1[:,:4])
    duplicates = np.zeros(len(embeddings1))
    for i,index in enumerate(indices):
        if duplicates[index] >= max_non_unique_correspondences:
            distances[i]= np.inf
        duplicates[index] += 1
    same_class = (classification_1==(classification_2[indices].squeeze()))
    distances[~same_class] = np.inf

    smallest_distances_indices = np.argsort(distances.flatten())[:num_neighbors]
    emb1_indices = smallest_distances_indices.squeeze()
    emb2_indices = indices[smallest_distances_indices].squeeze()
    return emb1_indices, emb2_indices

def random_rotation_translation(pointcloud):
  """
  Performs a random 3D rotation on a point cloud after centering it.

  Args:
      pointcloud: A NumPy array of shape (N, 3) representing the point cloud.

  Returns:
      A new NumPy array of shape (N, 3) representing the rotated point cloud.
  """
  # Center the point cloud by subtracting the mean of its coordinates
  center = np.mean(pointcloud, axis=0)
  centered_cloud = pointcloud - center

  # Generate random rotation angles for each axis
  theta_x = np.random.rand() * 2 * np.pi
  theta_y = np.random.rand() * 2 * np.pi
  theta_z = np.random.rand() * 2 * np.pi
  rotation_matrix = (Rotation.from_euler("xyz", [theta_x, theta_y, theta_z], degrees=False)).as_matrix()
  # Apply rotation to centered pointcloud
  rotated_cloud = (centered_cloud @ rotation_matrix)

  return ( rotated_cloud + center) , rotation_matrix



def ransac(data1, data2, max_iterations=1000, threshold=0.1, min_inliers=2):
    """
    Performs RANSAC to find the best rotation and translation between two sets of 3D points.

    Args:
        data1 (np.ndarray): Array of shape (N, 3) containing the first set of 3D points.
        data2 (np.ndarray): Array of shape (N, 3) containing the second set of 3D points.
        max_iterations (int): Maximum number of RANSAC iterations.
        threshold (float): Maximum distance for a point to be considered an inlier.
        min_inliers (int): Minimum number of inliers required to consider a model valid.

    Returns:
        rotation (np.ndarray): Array of shape (3, 3) representing the rotation matrix.
        translation (np.ndarray): Array of shape (3,) representing the translation vector.
        inliers1 (np.ndarray): Array containing the indices of the inliers in data1.
        inliers2 (np.ndarray): Array containing the indices of the inliers in data2.
    """
    N = data1.shape[0]
    best_inliers = None
    best_rotation = None
    best_translation = None

    # src_mean = np.mean(data1, axis=0)
    # dst_mean = np.mean(data2, axis=0)

    src_centered = data1 #- src_mean
    dst_centered = data2 #- dst_mean
    best_iter = 0
    for iteration in range(max_iterations):
        # Randomly sample 3 corresponding points
        indices = np.random.choice(N, size=4, replace=False)
        src_points = src_centered[indices]
        dst_points = dst_centered[indices]

        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        inliers1, inliers2 = find_inliers(src_centered, dst_centered, rotation, threshold)

        # Update best model if we have enough inliers
        if len(inliers1) >= min_inliers and (best_inliers is None or len(inliers1) > len(best_inliers)):
            best_inliers = inliers1
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_inliers == None:
        return ransac(data1, data2, max_iterations=max_iterations, threshold=threshold + 0.1, min_inliers=min_inliers)
    return best_rotation, best_inliers, best_iter

def classification_only_ransac(cls1_1,cls1_2,cls1_3,cls1_4, cls2_1,cls2_2,cls2_3,cls2_4,
                               max_iterations=1000, threshold=0.1, min_inliers=2):
    # N = data1.shape[0]
    best_num_of_inliers = 0
    corres = None
    best_rotation = None
    best_translation = None

    # Choose one array based on the probabilities
    pcl_1 = []
    pcl_2 = []
    for pcl1,pcl2 in zip([cls1_1, cls1_2, cls1_3, cls1_4], [cls2_1, cls2_2, cls2_3, cls2_4]):
        if len(pcl1)>0 and len(pcl2)>0:
            pcl_1.append(pcl1)
            pcl_2.append(pcl2)

    sizes = [len(cls) for cls in pcl_1]
    total_size = sum(sizes)
    probabilities = [(1 - (size / total_size)) for size in sizes if size > 0]
    if len(probabilities) > 1:
        probabilities = [(prob / (len(probabilities) - 1)) for prob in probabilities]
    else:
        probabilities = [1]
    best_iter = 0
    for iteration in range(max_iterations):
        # if iteration%10==0:
        #     print(f'Iteration {iteration}')
        # Randomly sample 3 corresponding points
        chosen_classes = random.choices(range(len(probabilities)), weights=probabilities, k=3)
        src_points = np.array([random.choice(pcl_1[cls]) for cls in chosen_classes])
        dst_points = np.array([random.choice(pcl_2[cls]) for cls in chosen_classes])
        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        inliers_1, inliers_2 = find_inliers_classification(cls1_1, cls1_2, cls1_3, cls1_4, cls2_1, cls2_2, cls2_3, cls2_4, rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return classification_only_ransac(cls1_1,cls1_2,cls1_3,cls1_4, cls2_1,cls2_2,cls2_3,cls2_4, max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_num_of_inliers, best_iter, corres, threshold

def multiclass_classification_only_ransac(cls_1, cls_2, max_iterations=1000, threshold=0.1, min_inliers=2):
    # N = data1.shape[0]
    best_num_of_inliers = 0
    corres = None
    best_rotation = None
    best_translation = None

    # Choose one array based on the probabilities
    pcl_1 = []
    pcl_2 = []
    for pcl1,pcl2 in zip(cls_1, cls_2):
        if len(pcl1)>0 and len(pcl2)>0:
            pcl_1.append(pcl1)
            pcl_2.append(pcl2)

    sizes = [len(cls) for cls in pcl_1]
    total_size = sum(sizes)
    probabilities = [(1 - (size / total_size)) for size in sizes if size > 0]
    if len(probabilities) > 1:
        probabilities = [(prob / (len(probabilities) - 1)) for prob in probabilities]
    else:
        probabilities = [1]
    best_iter = 0
    for iteration in range(max_iterations):
        # if iteration%10==0:
        #     print(f'Iteration {iteration}')
        # Randomly sample 3 corresponding points
        chosen_classes = random.choices(range(len(probabilities)), weights=probabilities, k=3)
        src_points = np.array([random.choice(pcl_1[cls]) for cls in chosen_classes])
        dst_points = np.array([random.choice(pcl_2[cls]) for cls in chosen_classes])
        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        inliers_1, inliers_2 = find_inliers_classification_multiclass(cls_1, cls_2, rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return multiclass_classification_only_ransac(cls_1, cls_2, max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_num_of_inliers, best_iter, corres, threshold

def random_only_ransac(cls1_1,cls1_2,cls1_3,cls1_4, cls2_1,cls2_2,cls2_3,cls2_4,
                               max_iterations=1000, threshold=0.1, min_inliers=2):
    # N = data1.shape[0]
    best_num_of_inliers = 0
    corres = None
    best_rotation = None
    best_translation = None

    # Choose one array based on the probabilities
    yay1 = [pcl for pcl in [cls1_1, cls1_2, cls1_3, cls1_4] if len(pcl) > 0]
    yay2 = [pcl for pcl in [cls2_1, cls2_2, cls2_3, cls2_4] if len(pcl) > 0]
    pcl_1 = np.vstack(yay1)
    # pcl_1 = np.random.permutation(pcl_1)
    pcl_2 = np.vstack(yay2)
    # pcl_2 = np.random.permutation(pcl_2)

    best_iter = 0
    for iteration in range(max_iterations):
        src_points = np.array([random.choice(pcl_1) for i in range(3)])
        dst_points = np.array([random.choice(pcl_2) for j in range(3)])
        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        dummy_1 = pcl_1.tolist()
        dummy_2 = pcl_2.tolist()
        inliers_1, inliers_2 = find_inliers_classification(dummy_1, [], [], [], dummy_2, [], [], [], rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return random_only_ransac(dummy_1, [], [], [], dummy_2, [], [], [], max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_num_of_inliers, best_iter, corres, threshold


def estimate_rigid_transform(src_points, dst_points):
    H = np.matmul(src_points.T, dst_points)
    U, _, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt, U.T)
    return R


def find_inliers(data1, data2, rotation, threshold):
    """
    Finds the inliers in two sets of 3D points given a rotation and translation.
    """
    inliers1 = []
    inliers2 = []
    for i in range(data1.shape[0]):
        point1 = data1[i]
        point2 = data2[i]
        # transformed = np.matmul(point1, rotation)
        transformed = np.matmul(rotation, point1)
        distance = np.linalg.norm(transformed - point2)
        if distance < threshold:
            inliers1.append(i)
            inliers2.append(i)

    return inliers1, inliers2
def find_inliers_classification(cls1_1, cls1_2, cls1_3, cls1_4, cls2_1, cls2_2, cls2_3, cls2_4, rotation, threshold=0.1):
    """
    Finds the inliers in two sets of labeled 3D points for each class using 1-nearest neighbor.
    """
    inliers_1 = []
    inliers_2 = []

    # Combine the inputs into lists for easier iteration
    orig_cls1 = []
    cls2 = []
    for pcl1,pcl2 in zip([cls1_1, cls1_2, cls1_3, cls1_4], [cls2_1, cls2_2, cls2_3, cls2_4]):
        if len(pcl1)>0 and len(pcl2)>0:
            orig_cls1.append(pcl1)
            cls2.append(pcl2)
    cls1 = [np.dot(pcl, rotation.T) for pcl in orig_cls1]

    # Iterate over each class
    for i in range(len(orig_cls1)):
        points_cls1 = np.array(cls1[i])
        points_cls2 = np.array(cls2[i])
        original_cls1 = np.array(orig_cls1[i])
        # Use NearestNeighbors to find the nearest neighbor in cls2 for each point in cls1
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points_cls2)
        distances, indices = nbrs.kneighbors(points_cls1)
        counter_arr = np.zeros((len(points_cls2)))
        for i in range(len(points_cls1)):
            if counter_arr[indices[i]]>1:
                distances[i] = np.inf
            counter_arr[indices[i]] += 1
        mask = distances<threshold
        if (np.count_nonzero(mask) > 0):
            in_1 = original_cls1[mask.squeeze()]
            in_2 = (points_cls2[indices])[mask]
            if len(in_1.shape)==3:
                in_1 = in_1.squeeze(0)
            if len(in_2.shape)==3:
                in_2 = in_2.squeeze(0)
            inliers_1.append(in_1)
            inliers_2.append(in_2)
    if len(inliers_1)==0 or len(inliers_2)==0:
        return inliers_1, inliers_2
    # print(f'inliers_1:')
    # print([x.shape for x in inliers_1])
    # print(f'inliers_2:')
    # print([x.shape for x in inliers_2])
    return np.vstack(inliers_1), np.vstack(inliers_2)
def find_inliers_classification_multiclass(cls_1, cls_2, rotation, threshold=0.1):
    """
    Finds the inliers in two sets of labeled 3D points for each class using 1-nearest neighbor.
    """
    inliers_1 = []
    inliers_2 = []

    # Combine the inputs into lists for easier iteration
    orig_cls1 = []
    cls2 = []
    for pcl1,pcl2 in zip(cls_1, cls_2):
        if len(pcl1)>0 and len(pcl2)>0:
            orig_cls1.append(pcl1)
            cls2.append(pcl2)
    cls1 = [np.dot(pcl, rotation.T) for pcl in orig_cls1]

    # Iterate over each class
    for i in range(len(orig_cls1)):
        points_cls1 = np.array(cls1[i])
        points_cls2 = np.array(cls2[i])
        original_cls1 = np.array(orig_cls1[i])
        # Use NearestNeighbors to find the nearest neighbor in cls2 for each point in cls1
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points_cls2)
        distances, indices = nbrs.kneighbors(points_cls1)
        counter_arr = np.zeros((len(points_cls2)))
        for i in range(len(points_cls1)):
            if counter_arr[indices[i]]>1:
                distances[i] = np.inf
            counter_arr[indices[i]] += 1
        mask = distances<threshold
        if (np.count_nonzero(mask) > 0):
            in_1 = original_cls1[mask.squeeze()]
            in_2 = (points_cls2[indices])[mask]
            if len(in_1.shape)==3:
                in_1 = in_1.squeeze(0)
            if len(in_2.shape)==3:
                in_2 = in_2.squeeze(0)
            inliers_1.append(in_1)
            inliers_2.append(in_2)
    if len(inliers_1)==0 or len(inliers_2)==0:
        return inliers_1, inliers_2
    # print(f'inliers_1:')
    # print([x.shape for x in inliers_1])
    # print(f'inliers_2:')
    # print([x.shape for x in inliers_2])
    return np.vstack(inliers_1), np.vstack(inliers_2)
def plotWorst(worst_losses, model_name=""):
    count = 0
    for (loss,worst_loss_variables) in worst_losses:
        noisy_pointcloud_1 = worst_loss_variables['noisy_pointcloud_1']
        noisy_pointcloud_2 = worst_loss_variables['noisy_pointcloud_2']
        chosen_points_1 = worst_loss_variables['chosen_points_1']
        chosen_points_2 = worst_loss_variables['chosen_points_2']
        rotation_matrix = worst_loss_variables['rotation_matrix']
        best_rotation = worst_loss_variables['best_rotation']
        save_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, title="", filename=model_name+f"_{loss:.3f}_orig_{count}_loss.html")
        save_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, filename=model_name+f"_{loss:.3f}_correspondence_{count}_loss.html", rotation=rotation_matrix)

        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        save_point_clouds(transformed_points1, noisy_pointcloud_2 - center2, title="", filename=model_name+f"_{loss:.3f}_{count}_loss.html")
        count = count + 1

def plot_losses(losses, inliers, filename="loss_plot.png"):
    """
    Plots the given list of losses and corresponding number of inliers,
    and saves the plot as an image
    """
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)

    plt.figure(figsize=(8, 6))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot((np.array(inliers)), marker='o', linestyle='-', color='blue', label='Inliers')
    ax2.plot(losses, marker='s', linestyle='-', color='red', label='Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('#Inliers', color='blue')
    ax2.set_ylabel('Loss', color='red')

    ax1.grid(True)
    plt.title(f'mean loss = {mean_loss:.5f}, median loss = {median_loss:.5f}')

    # Save the plot as an image
    plt.savefig(filename)

    # You can optionally remove the following line if you don't want to display the plot as well
    plt.show()



def test_coress_dis(model_name=None, args_shape=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None):
    pcls, label = load_data()
    shape_size_list = []
    shortest_dist_list = []
    avg_dist_list = []
    dist_from_orig = []
    # diameter_20_nn_list = []
    shapes = [86, 162, 174, 176, 179]
    # shapes = np.arange(100)
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=10)
        emb_2 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_2,args_shape=args_shape, scaling_factor=10)
        # diameter_20_nn_list.append(diameter_20_nn)
        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue

        # max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
        #                                                                                                  k=200)
        # max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
        #                                                                                                  k=200)

        good_class_indices_1 = farthest_point_sampling(emb_1[:, :4], k=200)
        good_class_indices_2 = farthest_point_sampling(emb_2[:, :4], k=200)

        best_point_desc_pcl1 = emb_1[good_class_indices_1, :]
        best_point_desc_pcl2 = emb_2[good_class_indices_2, :]

        source_indices, target_indices = find_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
                                                             num_neighbors=40, max_non_unique_correspondences=1)


        chosen_indices_pcl1 = good_class_indices_1[source_indices]
        chosen_indices_pcl2 = good_class_indices_2[target_indices]
        chosen_points_1 = noisy_pointcloud_1[chosen_indices_pcl1, :]
        chosen_points_2 = noisy_pointcloud_2[chosen_indices_pcl2, :]

        amount_same_index = np.sum(chosen_indices_pcl1 == chosen_indices_pcl2)
        original_points_chosen = pointcloud[chosen_indices_pcl1, :]
        original_corresponding_points = pointcloud[chosen_indices_pcl2, :]
        pairwise_distances = cdist(pointcloud, pointcloud)
        shape_size = np.max(pairwise_distances)
        shape_size_list.append(shape_size)
        correspondences_dist = (pairwise_distances[chosen_indices_pcl1,chosen_indices_pcl2])
        flat_array = correspondences_dist.flatten()
        sorted_values = np.sort(flat_array)
        best_30 = sorted_values[:(len(correspondences_dist)//10) *3]
        best_30 = np.append(best_30, shape_size)
        # best_30 = np.append(best_30, diameter_20_nn)
        # Step 5: Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(best_30)), best_30)
        plt.grid(True)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Smallest 30% Values in the Array (Ordered)')
        plt.show()
        dist_from_orig.append(np.mean(best_30))
        mask = ~np.eye(pairwise_distances.shape[0], dtype=bool)
        flattened_arr = pairwise_distances[mask]
        avg_dist_list.append(np.mean(flattened_arr))
        shortest_dist_list.append(np.min(flattened_arr))
        centered_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :] - np.mean(noisy_pointcloud_2)
        best_rotation, inliers, best_iter = ransac(centered_points_1, centered_points_2, max_iterations=1000,
                                                   threshold=0.05,
                                                   min_inliers=5)
        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, rotation=rotation_matrix,
                            title=f'Loss is {loss:.3f}; <br>Shape size: {np.max(pairwise_distances)}; <br>best_30: {np.mean(best_30)};')

    return 1, 1, 1, 1, shape_size_list, dist_from_orig, shortest_dist_list, avg_dist_list

def test_classification(cls_args=None,contr_args=None,smooth_args=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None, point_choice=0, num_of_ransac_iter=100, random_pairing=0,subsampled_points=100):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    worst_point_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    # shapes = [86, 162, 174, 176, 179]
    shapes = np.arange(100)
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_2,args_shape=cls_args, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue
        if point_choice == 0:
            max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
                                                                                                             k=subsampled_points)
            max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
                                                                                                             k=subsampled_points)
        if point_choice == 1:
            max_values_1 = np.max(emb_1[:, :4], axis=1)
            good_class_indices_1 = np.argsort(max_values_1)[-subsampled_points:][::-1]
            max_values_2 = np.max(emb_2[:, :4], axis=1)
            good_class_indices_2 = np.argsort(max_values_2)[-subsampled_points:][::-1]
        if point_choice == 2:
            good_class_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=subsampled_points)
            good_class_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=subsampled_points)

        classification_pcl1 = np.argmax((emb_1[good_class_indices_1, :]), axis=1)
        classification_pcl2 = np.argmax((emb_2[good_class_indices_2, :]), axis=1)


        centered_points_1 = noisy_pointcloud_1[good_class_indices_1, :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2, :] - np.mean(noisy_pointcloud_2)

        # Concatenate along the second axis (axis=1)
        result_1 = np.concatenate((classification_pcl1.reshape(len(classification_pcl1), 1), centered_points_1), axis=1)
        result_2 = np.concatenate((classification_pcl2.reshape(len(classification_pcl2), 1), centered_points_2), axis=1)
        cls1_0 = [pcl[1:] for pcl in result_1 if pcl[0]==0]
        cls1_1 = [pcl[1:] for pcl in result_1 if pcl[0]==1]
        cls1_2 = [pcl[1:] for pcl in result_1 if pcl[0]==2]
        cls1_3 = [pcl[1:] for pcl in result_1 if pcl[0]==3]
        cls2_0 = [pcl[1:] for pcl in result_2 if pcl[0]==0]
        cls2_1 = [pcl[1:] for pcl in result_2 if pcl[0]==1]
        cls2_2 = [pcl[1:] for pcl in result_2 if pcl[0]==2]
        cls2_3 = [pcl[1:] for pcl in result_2 if pcl[0]==3]
        # plot_8_point_clouds( cls1_0, cls1_1, cls1_2, cls1_3, cls2_0, cls2_1, cls2_2, cls2_3,  rotation=rotation_matrix)
        if random_pairing == 1:
            best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = random_only_ransac(
                cls1_0, cls1_1, cls1_2, cls1_3, cls2_0, cls2_1, cls2_2, cls2_3, max_iterations=num_of_ransac_iter,
                                                       threshold=0.1,
                                                       min_inliers=subsampled_points/10)
        else:
            best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = classification_only_ransac(
                cls1_0, cls1_1, cls1_2, cls1_3, cls2_0, cls2_1, cls2_2, cls2_3, max_iterations=num_of_ransac_iter,
                threshold=0.1,
                min_inliers=subsampled_points / 10)
        final_thresh_list.append(final_threshold)
        final_inliers_list.append(best_num_of_inliers)
        iter_2_ransac_convergence.append(best_iter)

        center = np.mean(noisy_pointcloud_1, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        losses.append(loss)

        kdtree = cKDTree(transformed_points1)

        # Query the KDTree with points from pcl2
        distances, indices = kdtree.query(noisy_pointcloud_2)

        point_distance = np.mean(distances)
        point_distance_list.append(point_distance)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if loss > smallest_loss:
            worst_losses[index_of_smallest_loss] = (loss, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_point_losses[i][0] for i in range(len(worst_point_losses))])
        smallest_loss = worst_point_losses[index_of_smallest_loss][0]
        if point_distance > smallest_loss:
            worst_point_losses[index_of_smallest_loss] = (point_distance, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
    return worst_losses, losses, final_thresh_list, final_inliers_list, point_distance_list, worst_point_losses, iter_2_ransac_convergence

def test_multi_scale_classification(cls_args=None,num_worst_losses = 3, scaling_factor=None, point_choice=0, num_of_ransac_iter=100, subsampled_points=100):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    worst_point_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    shapes = [86, 162, 174, 176, 179]
    # shapes = [86, 162]
    # shapes = np.arange(100)
    num_of_points_to_sample = subsampled_points
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_2,args_shape=cls_args, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue
        if point_choice == 0:
            max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
                                                                                                             k=num_of_points_to_sample)
            max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
                                                                                                             k=num_of_points_to_sample)
        if point_choice == 1:
            max_values_1 = np.max(emb_1[:, :4], axis=1)
            good_class_indices_1 = np.argsort(max_values_1)[-num_of_points_to_sample:][::-1]
            max_values_2 = np.max(emb_2[:, :4], axis=1)
            good_class_indices_2 = np.argsort(max_values_2)[-num_of_points_to_sample:][::-1]
        if point_choice == 2:
            good_class_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=num_of_points_to_sample)
            good_class_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=num_of_points_to_sample)

        classification_pcl1 = np.argmax((emb_1[good_class_indices_1, :]), axis=1)
        classification_pcl2 = np.argmax((emb_2[good_class_indices_2, :]), axis=1)

        centered_points_1 = noisy_pointcloud_1[good_class_indices_1, :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2, :] - np.mean(noisy_pointcloud_2)


        # global_emb_1 = checkOnShapes(model_name=cls_args.exp,
        #                             input_data=centered_points_1, args_shape=cls_args, scaling_factor=scaling_factor)
        # global_emb_2 = checkOnShapes(model_name=cls_args.exp,
        #                             input_data=centered_points_2,args_shape=cls_args, scaling_factor=scaling_factor)

        fps_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=50)
        fps_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=50)

        global_emb_1 , scaling_fac = classifyPoints(model_name=cls_args.exp,
                                    pcl_src=noisy_pointcloud_1[fps_indices_1,:], pcl_interest=centered_points_1, args_shape=cls_args, scaling_factor=scaling_factor)

        global_emb_2 , scaling_fac = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_2[fps_indices_2,:], pcl_interest=centered_points_2, args_shape=cls_args, scaling_factor=scaling_factor)


        global_emb_1 = global_emb_1.detach().cpu().numpy()
        global_emb_2 = global_emb_2.detach().cpu().numpy()

        global_classification_pcl1 = np.argmax((global_emb_1), axis=1)
        global_classification_pcl2 = np.argmax((global_emb_2), axis=1)

        pcl1_classes = [[] for _ in range(16)]

        # Iterate over the point cloud and classification arrays
        for i in range(len(centered_points_1)):
            index = classification_pcl1[i] * 4 + global_classification_pcl1[i]
            pcl1_classes[index].append(centered_points_1[i,:])

        pcl2_classes = [[] for _ in range(16)]

        # Iterate over the point cloud and classification arrays
        for i in range(len(centered_points_1)):
            index = classification_pcl2[i] * 4 + global_classification_pcl2[i]
            pcl2_classes[index].append(centered_points_2[i,:])

        plot_multiclass_point_clouds(pcl1_classes, pcl2_classes, rotation=rotation_matrix, title="")

        best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = multiclass_classification_only_ransac(
            pcl1_classes, pcl2_classes, max_iterations=num_of_ransac_iter,
            threshold=0.1,
            min_inliers=num_of_points_to_sample / 10)
        final_thresh_list.append(final_threshold)
        final_inliers_list.append(best_num_of_inliers)
        iter_2_ransac_convergence.append(best_iter)

        center = np.mean(noisy_pointcloud_1, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        losses.append(loss)

        kdtree = cKDTree(transformed_points1)

        # Query the KDTree with points from pcl2
        distances, indices = kdtree.query(noisy_pointcloud_2)

        point_distance = np.mean(distances)
        point_distance_list.append(point_distance)

        plot_point_clouds(transformed_points1, noisy_pointcloud_2, f'loss is: {loss}; inliers: {best_num_of_inliers}; threshold: {final_threshold}')
        plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, corres[0], corres[1], rotation=rotation_matrix.T)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if loss > smallest_loss:
            worst_losses[index_of_smallest_loss] = (loss, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_point_losses[i][0] for i in range(len(worst_point_losses))])
        smallest_loss = worst_point_losses[index_of_smallest_loss][0]
        if point_distance > smallest_loss:
            worst_point_losses[index_of_smallest_loss] = (point_distance, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
    return worst_losses, losses, final_thresh_list, final_inliers_list, point_distance_list, worst_point_losses, iter_2_ransac_convergence


def view_stabiity(model_name=None, args_shape=None, scaling_factor=None):
    pcls, label = load_data()
    shapes = [82, 83, 86, 174]
    for k in shapes:
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 , scaling_fac = classifyPoints(model_name=model_name, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1,
                       args_shape=args_shape, scaling_factor=scaling_factor)
        emb_2 , scaling_fac = classifyPoints(model_name=model_name, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2,
                       args_shape=args_shape, scaling_factor=scaling_factor)
        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()
        # embeddings_1 = []
        # embeddings_2 = []
        # for i in range(2,5):
        #     scaling_factor = i * 10
        #
        #     emb_1 = checkOnShapes(model_name=model_name,
        #                                 input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=scaling_factor)
        #
        #     emb_2 = checkOnShapes(model_name=model_name,
        #                                 input_data=noisy_pointcloud_2, args_shape=args_shape, scaling_factor=scaling_factor)
        #
        #     emb_1 = emb_1.detach().cpu().numpy()
        #     emb_2 = emb_2.detach().cpu().numpy()
        #     embeddings_1.append(emb_1[:,4:])
        #     embeddings_2.append(emb_2[:,4:])
        # emb_1 = np.hstack((embeddings_1))
        # emb_2 = np.hstack((embeddings_2))
        if np.isnan(np.sum(emb_1)):
            print(f'oish')
            continue
        # plot_point_cloud_with_colors_by_dist(noisy_pointcloud_1, emb_1)
        # plot_point_cloud_with_colors_by_dist(noisy_pointcloud_1, emb_1[:,:4])
        # plot_point_cloud_with_colors_by_dist(noisy_pointcloud_1, emb_1[:,4:])

        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2)
        # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1[:,:4], emb_2[:,:4])
        # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1[:,4:], emb_2[:,4:])

def fit_surface_quadratic_constrained(points):
    """
    Fits a quadratic surface constrained to f = 0 to a centered point cloud.

    Args:
      points: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      numpy array of shape (5,) representing the surface coefficients
        [a, b, c, d, e], where:
          z = a * x**2 + b * y**2 + c * x * y + d * x + e * y
    """

    # Center the points around the mean
    centroid = points[0,:]
    centered_points = points - centroid

    # Design matrix without f term
    X = np.c_[centered_points[:, 0] ** 2, centered_points[:, 1] ** 2,
              centered_points[:, 0] * centered_points[:, 1],
    centered_points[:, 0], centered_points[:, 1]]

    # Extract z-coordinates as target vector
    z = centered_points[:, 2]

    # Solve the linear system with f coefficient constrained to 0
    coeffs = np.linalg.lstsq(X, z, rcond=None)[0]

    a, b, c, d, e = coeffs

    K = (4 * (a * b) - ((c ** 2))) / ((1 + d ** 2 + e ** 2) ** 2)
    H = (a * (1 + e ** 2) - d * e * c + b * (1 + d ** 2)) / (((d ** 2) + (e ** 2) + 1) ** 1.5)


    gaussian = 0
    if K > 0.05:
        gaussian = 1
    if K < -0.05:
        gaussian = -1

    mean = 0
    if H > 0.05:
        gaussian = 1
    if H < -0.05:
        gaussian = -1

    if gaussian == 0 and mean == 0:
        return 0
    if gaussian == 1:
        return 1
    if gaussian == 0:
        return 2
    return 3
def fix_orientation(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    rotation_axis = np.cross(np.array([0, 0, 1]), normal_at_centroid)

    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), normal_at_centroid))
    cosine_angle = np.arccos(np.dot(np.array([0, 0, 1]), normal_at_centroid))

    rotation_matrix = np.array([
        [1 + (1 - cosine_angle) * (rotation_axis[0] ** 2),
         (1 - cosine_angle) * rotation_axis[0] * rotation_axis[1] - rotation_angle * rotation_axis[2],
         (1 - cosine_angle) * rotation_axis[0] * rotation_axis[2] + rotation_angle * rotation_axis[1]],
        [(1 - cosine_angle) * rotation_axis[1] * rotation_axis[0] + rotation_angle * rotation_axis[2],
         1 + (1 - cosine_angle) * (rotation_axis[1] ** 2),
         (1 - cosine_angle) * rotation_axis[1] * rotation_axis[2] - rotation_angle * rotation_axis[0]],
        [(1 - cosine_angle) * rotation_axis[2] * rotation_axis[0] - rotation_angle * rotation_axis[1],
         (1 - cosine_angle) * rotation_axis[2] * rotation_axis[1] + rotation_angle * rotation_axis[0],
         1 + (1 - cosine_angle) * (rotation_axis[2] ** 2)]
    ])

    # Apply the rotation to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)

    return rotated_point_cloud + centroid
def read_bin_file(bin_file):
    """
    Read a .bin file and return a numpy array of shape (N, 3) where N is the number of points.

    Args:
        bin_file (str): Path to the .bin file.

    Returns:
        np.ndarray: Numpy array containing the point cloud data.
    """
    # Load the binary file
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

    # We only need the first three columns (x, y, z)
    return points[:, :3]
def create_args(cls_model_name, contrastive_model_name, smoothing_model_name):
    cls_args_shape = configArgsPCT()
    cls_args_shape.batch_size = 1024
    cls_args_shape.num_mlp_layers = 3
    cls_args_shape.num_neurons_per_layer = 32
    cls_args_shape.sampled_points = 40
    cls_args_shape.use_second_deg = 1
    cls_args_shape.lpe_normalize = 1
    cls_args_shape.exp = cls_model_name
    cls_args_shape.lpe_dim = 6
    cls_args_shape.output_dim = 4
    contrastive_args_shape = configArgsPCT()
    contrastive_args_shape.batch_size = 1024
    contrastive_args_shape.num_mlp_layers = 3
    contrastive_args_shape.num_neurons_per_layer = 32
    contrastive_args_shape.sampled_points = 40
    contrastive_args_shape.use_second_deg = 1
    contrastive_args_shape.lpe_normalize = 1
    contrastive_args_shape.exp = contrastive_model_name
    contrastive_args_shape.lpe_dim = 6
    contrastive_args_shape.output_dim = 8
    smoothing_args_shape = configArgsPCT()
    smoothing_args_shape.batch_size = 1024
    smoothing_args_shape.num_mlp_layers = 3
    smoothing_args_shape.num_neurons_per_layer = 32
    smoothing_args_shape.sampled_points = 40
    smoothing_args_shape.use_second_deg = 1
    smoothing_args_shape.lpe_normalize = 1
    smoothing_args_shape.exp = smoothing_model_name
    smoothing_args_shape.lpe_dim = 6
    smoothing_args_shape.output_dim = 8
    return cls_args_shape, contrastive_args_shape, smoothing_args_shape

def create_3MLP32N2deg_lpe0eig36_args():
    cls_args_shape = configArgsPCT()
    cls_args_shape.batch_size = 1024
    cls_args_shape.num_mlp_layers = 3
    cls_args_shape.num_neurons_per_layer = 32
    cls_args_shape.sampled_points = 40
    cls_args_shape.use_second_deg = 1
    cls_args_shape.lpe_normalize = 0
    cls_args_shape.exp = '3MLP32N2deg_lpe0eig36'
    cls_args_shape.lpe_dim = 0
    cls_args_shape.output_dim = 4
    cls_args_shape.use_lap_reorder = 1
    cls_args_shape.lap_eigenvalues_dim = 36
    return cls_args_shape, 1, 1

def select_top_10(data_dict):
    mean_dict = {key: np.mean(value) for key, value in data_dict.items()}
    sorted_keys = sorted(mean_dict, key=mean_dict.get)[:10]
    print(f'----------------------')
    print(f'mean: {np.mean([mean_dict[key] for key in sorted_keys])}')
    print(f'best: {sorted_keys[0]} ,{mean_dict[sorted_keys[0]]}')
    # print({key: mean_dict[key] for key in sorted_keys})
    return {key: data_dict[key] for key in sorted_keys}
def create_plot(data_dict, title, ylabel, amount_of_points_to_subsample):
    plt.figure(figsize=(10, 6))
    for key, data in data_dict.items():
        point_choice, ransac_iter, subsample = key
        if subsample != amount_of_points_to_subsample:
            continue
        label = f'Point choice: {point_choice}, Ransac iter: {ransac_iter}, #Subsample: {subsample}'
        plt.plot(data, label=label)
    plt.title(f"{title} (Subsample: {amount_of_points_to_subsample})")
    plt.xlabel('Index')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
def check_samplings(subsample_list=[200,300,400,500]):
    for amount_2_sample in subsample_list:
        cls_args, contr_args, smooth_args = create_args(cls_model_name='3MLP32Ncls', contrastive_model_name='3MLP32Ncontr',
                                                        smoothing_model_name='3MLP32Nsmooth')
        scaling_factor = 10
        random_fps_loss_list = []
        graph_list_loss = []
        MS_graph_list_loss = []
        num_of_iterations = [50, 100, 200, 400]

        for num_of_ransac_iter in num_of_iterations:
            worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
                = test_classification(cls_args=cls_args, max_non_unique_correspondences=5,
                                      scaling_factor=scaling_factor, point_choice=0,
                                      num_of_ransac_iter=num_of_ransac_iter, random_pairing=0,
                                      subsampled_points=amount_2_sample)
            graph_list_loss.append(np.mean(losses))
        np.save(f'{amount_2_sample}_graph_list_loss.npy', np.array(graph_list_loss))

        for num_of_ransac_iter in num_of_iterations:
            worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
                = test_multi_scale_classification(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor,
                                                  point_choice=0,
                                                  num_of_ransac_iter=num_of_ransac_iter,
                                                  subsampled_points=amount_2_sample)
            MS_graph_list_loss.append(np.mean(losses))
        np.save(f'{amount_2_sample}_MS_graph_list_loss.npy', np.array(MS_graph_list_loss))

        for num_of_ransac_iter in num_of_iterations:
            worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
                = test_classification(cls_args=cls_args, max_non_unique_correspondences=5,
                                      scaling_factor=scaling_factor, point_choice=2,
                                      num_of_ransac_iter=num_of_ransac_iter, random_pairing=1,
                                      subsampled_points=amount_2_sample)
            random_fps_loss_list.append(np.mean(losses))
        np.save(f'{amount_2_sample}_random_fps_loss_list.npy', np.array(random_fps_loss_list))

        plt.plot(num_of_iterations, graph_list_loss, label="max_diff")
        plt.plot(num_of_iterations, MS_graph_list_loss, label="MS_max_diff")
        plt.plot(num_of_iterations, random_fps_loss_list, label="random_fps")

        # Add labels and title
        plt.xlabel("ransac_iters")
        plt.ylabel("rot_loss")
        plt.title(f"Loss for ransac iters; Subsampled points: {amount_2_sample}")
        # Add legend
        plt.legend()
        # Show the plot
        plt.show()
if __name__ == '__main__':
    # check_samplings(subsample_list=[100,200,300,400,500])
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args()
    scaling_factor = 17
    scaling_factor = 15
    # visualizeShapesWithEmbeddings(model_name='3MLP32N2deg_lpe0eig36', args_shape=cls_args, scaling_factor=scaling_factor)
    # for scaling_factor in [10, 15, 20, 25]:
    for scaling_factor in [15]:
        worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
            = test_multi_scale_classification(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor,
                                              point_choice=2,
                                              num_of_ransac_iter=100,
                                              subsampled_points=300)
        # plot_losses(losses=losses, inliers=num_of_inliers, filename=f'{scaling_factor}_sf_multiscale.png')
        # plotWorst(worst_losses=worst_losses, model_name=f'{scaling_factor}_sf_multiscale')