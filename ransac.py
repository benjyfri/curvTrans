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
import platform


def find_mean_diameter_for_specific_coordinate(specific_coordinates):
    pairwise_distances = torch.cdist(specific_coordinates.unsqueeze(2), specific_coordinates.unsqueeze(2))
    largest_dist = pairwise_distances.view(specific_coordinates.shape[0], -1).max(dim=1).values
    mean_distance = torch.mean(largest_dist)
    return mean_distance
def load_data(partition='test', divide_data=1):
    BASE_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes'
    DATA_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes\\data'
    if platform.system() != "Windows":
        DATA_DIR = r'/content/curvTrans/bbsWithShapes/data'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
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
def classifyPoints(model_name=None, pcl_src=None,pcl_interest=None, args_shape=None, scaling_factor=None):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'models_weights/{model_name}.pt'))
    model.eval()
    neighbors_centered = get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k=41)
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
    return max_values, max_indices, diff_from_max, good_class_indices

def random_rotation_translation(pointcloud, translation=np.array([0,0,0])):
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
  new_pointcloud = (rotated_cloud + center) + translation

  return new_pointcloud , rotation_matrix, translation



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
        indices = np.random.choice(N, size=3, replace=False)
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
        inliers_1, inliers_2 = find_inliers_classification_multiclass(cls_1=[dummy_1], cls_2=[dummy_2], rotation=rotation, threshold=threshold)

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
    return np.vstack(inliers_1), np.vstack(inliers_2)

def test_multi_scale_classification(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    interest_point_choice=0, num_of_ransac_iter=100, shapes=[86, 162, 174, 176, 179], plot_graphs=0, create_pcls_func=None):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses
    worst_point_losses = [(0, None)] * num_worst_losses
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pcl1 = pcls[k][:]
        rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl1)

        if create_pcls_func is not None:
            pcl1, pcl2 = create_pcls_func(pcl1)
            rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl2)

        noisy_pointcloud_1 = pcl1 + np.random.normal(0, 0.01, pcl1.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2, args_shape=cls_args, scaling_factor=scaling_factor)


        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue

        # How to choose the interest points
        if interest_point_choice == 0:
            max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
                                                                                                             k=amount_of_interest_points)
            max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
                                                                                                             k=amount_of_interest_points)
        if interest_point_choice == 1:
            max_values_1 = np.max(emb_1[:, :4], axis=1)
            good_class_indices_1 = np.argsort(max_values_1)[-amount_of_interest_points:][::-1]
            max_values_2 = np.max(emb_2[:, :4], axis=1)
            good_class_indices_2 = np.argsort(max_values_2)[-amount_of_interest_points:][::-1]
        if interest_point_choice == 2:
            good_class_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=amount_of_interest_points)
            good_class_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=amount_of_interest_points)

        classification_pcl1 = np.argmax((emb_1[good_class_indices_1, :]), axis=1)
        classification_pcl2 = np.argmax((emb_2[good_class_indices_2, :]), axis=1)

        centered_points_1 = noisy_pointcloud_1[good_class_indices_1, :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2, :] - np.mean(noisy_pointcloud_2)

        classification_by_scale_pcl1 = [classification_pcl1]
        classification_by_scale_pcl2 = [classification_pcl2]
        # multiscale classification
        if scales > 1:
            for scale in receptive_field[1:]:
                fps_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1)//scale))
                fps_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2)//scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_1[fps_indices_1,:], pcl_interest=centered_points_1, args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_2[fps_indices_2,:], pcl_interest=centered_points_2, args_shape=cls_args, scaling_factor=scaling_factor)


                global_emb_1 = global_emb_1.detach().cpu().numpy()
                global_emb_2 = global_emb_2.detach().cpu().numpy()

                global_classification_pcl1 = np.argmax((global_emb_1), axis=1)
                global_classification_pcl2 = np.argmax((global_emb_2), axis=1)

                classification_by_scale_pcl1.append(global_classification_pcl1)
                classification_by_scale_pcl2.append(global_classification_pcl2)

        pcl1_classes = [[] for _ in range(pow(4, scales))]

        # Iterate over the interest point classification
        for i in range(len(centered_points_1)):
            index = 0
            for exp, cls in enumerate(classification_by_scale_pcl1):
                index += (cls[i]) * pow(4, exp)
            pcl1_classes[index].append(centered_points_1[i,:])

        pcl2_classes = [[] for _ in range(pow(4, scales))]
        for i in range(len(centered_points_2)):
            index = 0
            for exp, cls in enumerate(classification_by_scale_pcl2):
                index += (cls[i]) * pow(4, exp)
            pcl2_classes[index].append(centered_points_2[i, :])


        best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = multiclass_classification_only_ransac(
            pcl1_classes, pcl2_classes, max_iterations=num_of_ransac_iter,
            threshold=0.1,
            min_inliers=(int)(amount_of_interest_points // 10))
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

        if plot_graphs:
            plot_multiclass_point_clouds(pcl1_classes, pcl2_classes, rotation=rotation_matrix, title="")
            # plot_point_clouds(transformed_points1, noisy_pointcloud_2, f'loss is: {loss}; inliers: {best_num_of_inliers}; threshold: {final_threshold}')
            # plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, corres[0], corres[1], rotation=rotation_matrix.T)

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


def test_multi_scale_using_embedding(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    interest_point_choice=0, num_of_ransac_iter=100, shapes=[86, 162, 174, 176, 179], plot_graphs=0, create_pcls_func=None):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses
    worst_point_losses = [(0, None)] * num_worst_losses
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pcl1 = pcls[k][:]
        rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl1)

        if create_pcls_func is not None:
            pcl1, pcl2 = create_pcls_func(pcl1)
            rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl2)

        noisy_pointcloud_1 = pcl1 + np.random.normal(0, 0.01, pcl1.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2, args_shape=cls_args, scaling_factor=scaling_factor)


        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue

        # How to choose the interest points
        if interest_point_choice == 0:
            max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
                                                                                                             k=amount_of_interest_points)
            max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
                                                                                                             k=amount_of_interest_points)
        if interest_point_choice == 1:
            max_values_1 = np.max(emb_1[:, :4], axis=1)
            good_class_indices_1 = np.argsort(max_values_1)[-amount_of_interest_points:][::-1]
            max_values_2 = np.max(emb_2[:, :4], axis=1)
            good_class_indices_2 = np.argsort(max_values_2)[-amount_of_interest_points:][::-1]
        if interest_point_choice == 2:
            good_class_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=amount_of_interest_points)
            good_class_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=amount_of_interest_points)

        centered_points_1 = noisy_pointcloud_1[good_class_indices_1, :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2, :] - np.mean(noisy_pointcloud_2)

        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                fps_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1)//scale))
                fps_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2)//scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_1[fps_indices_1,:], pcl_interest=centered_points_1, args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_2[fps_indices_2,:], pcl_interest=centered_points_2, args_shape=cls_args, scaling_factor=scaling_factor)


                global_emb_1 = global_emb_1.detach().cpu().numpy()
                global_emb_2 = global_emb_2.detach().cpu().numpy()

                emb_1 = np.vstack((global_emb_1, emb_1))
                emb_2 = np.vstack((global_emb_2, emb_2))



        best_rotation, inliers, best_iter = ransac(centered_points_1, centered_points_2, max_iterations=num_of_ransac_iter,
                                                   threshold=0.1,
                                                   min_inliers=(int)(amount_of_interest_points // 10))


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

