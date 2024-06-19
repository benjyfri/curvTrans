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
from experiment_shapes import *
import numpy as np
from scipy.spatial.kdtree import KDTree
def load_data(partition='test', divide_data=1):
    BASE_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes'
    DATA_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes\\data'
    # DATA_DIR = r'/content/curvTrans/bbsWithShapes/data'
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

def low_overlap_multi_scale(cls_args=None,num_worst_losses = 3, scaling_factor=None, point_choice=0, num_of_ransac_iter=100, subsampled_points=100):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    worst_point_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    shapes = [86, 174, 179]
    shapes = [179]
    # shapes = [86, 162]
    # shapes = np.arange(100)
    num_of_points_to_sample = subsampled_points
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        pointcloud, pcl2, overlap = get_two_point_clouds_with_neighbors(pointcloud, (int)(len(pointcloud) // 1.53))
        rotated_pcl, rotation_matrix = random_rotation_translation(pcl2)

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

        fps_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=300)
        fps_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=300)

        global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                    pcl_src=noisy_pointcloud_1[fps_indices_1,:], pcl_interest=centered_points_1, args_shape=cls_args, scaling_factor=scaling_factor)

        global_emb_2 = classifyPoints(model_name=cls_args.exp,
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
def low_overlap(cls_args=None,contr_args=None,smooth_args=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None, point_choice=0, num_of_ransac_iter=100, random_pairing=0,subsampled_points=100):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    worst_point_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    # shapes = [86, 162, 174, 176, 179]
    shapes = [86, 179]
    # shapes = np.arange(100)
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        pointcloud, pcl2, overlap = get_two_point_clouds_with_neighbors(pointcloud, (int)(len(pointcloud) // 1.53))
        rotated_pcl, rotation_matrix = random_rotation_translation(pcl2)

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
        plot_multiclass_point_clouds([cls1_0, cls1_1, cls1_2, cls1_3], [cls2_0, cls2_1, cls2_2, cls2_3], rotation=rotation_matrix, title="")
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

        # plot_point_clouds(transformed_points1, noisy_pointcloud_2,
        #                   f'loss is: {loss}; inliers: {best_num_of_inliers}; threshold: {final_threshold}')
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

def check_overlap():
    pcls, label = load_data()
    shapes = [86, 162, 174, 176, 179]
    shapes = np.arange(100)
    overlap_avg = 0
    overlap_max = 0.33
    overlap_min = 0.28
    divider = 1.53
    while (overlap_avg < overlap_min) or (overlap_avg > overlap_max):

        sum = 0
        for k in shapes:
            pointcloud = pcls[k][:]
            pcl1, pcl2, overlap = get_two_point_clouds_with_neighbors(pointcloud, (int)(len(pointcloud)//divider))
            # plot_point_clouds(pcl1, pcl2, f'Overlap = {overlap*100:.1f}%')
            sum += overlap
        overlap_avg = (sum/len(shapes))
        print(f'+++++++++++++++')
        print(f'divider: {divider}')
        print(f'overlap: {overlap_avg}')
        divider += 0.05

def get_two_point_clouds_with_neighbors(point_cloud, k):
    """
    Selects one point randomly and finds the farthest point from it in the point cloud.
    Creates 2 point clouds, one with the first point and its k nearest neighbors
    and the second with the second point and its k nearest neighbors.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (pcl_size, 3)
        k (int): Number of nearest neighbors to include in each new point cloud

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays each of shape (k+1, 3) containing the points
    """

    pcl_size = point_cloud.shape[0]
    # Step 1: Choose a random point from the point cloud
    random_idx = np.random.randint(pcl_size)
    random_point = point_cloud[random_idx]

    distances = np.linalg.norm(point_cloud[:, np.newaxis, :] - point_cloud[np.newaxis, :, :], axis=2)

    # Find the indices of the maximum distance
    max_dist_indices = np.unravel_index(np.argmax(distances, axis=None), distances.shape)

    # Extract the farthest points
    farthest_point_1 = point_cloud[max_dist_indices[0]]
    farthest_point_2 = point_cloud[max_dist_indices[1]]

    # Step 3: Find k nearest neighbors for the random point and the farthest point
    neigh = NearestNeighbors(n_neighbors=k+1)  # k+1 because the point itself is included
    neigh.fit(point_cloud)

    # For random point
    _, indices_farthest_1 = neigh.kneighbors([farthest_point_1])
    random_point_cloud = point_cloud[indices_farthest_1[0]]

    # For farthest point
    _, indices_farthest_2 = neigh.kneighbors([farthest_point_2])
    farthest_point_cloud = point_cloud[indices_farthest_2[0]]

    #calculate overlap
    array1_flat = indices_farthest_1.flatten()
    array2_flat = indices_farthest_2.flatten()
    # Find the intersection of the two arrays
    intersection = np.intersect1d(array1_flat, array2_flat)
    overlap = len(intersection) / pcl_size
    # print(overlap)
    return random_point_cloud, farthest_point_cloud, overlap

if __name__ == '__main__':
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args()
    scaling_factor = 17
    scaling_factor = 15
    # visualizeShapesWithEmbeddings(model_name='3MLP32N2deg_lpe0eig36', args_shape=cls_args, scaling_factor=scaling_factor)
    # for scaling_factor in [10, 15, 20, 25]:
    for scaling_factor in range(15,25):
        # worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
        #     = low_overlap_multi_scale(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor,
        #                                       point_choice=2,
        #                                       num_of_ransac_iter=100,
        #                                       subsampled_points=300)
        worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
            = low_overlap(cls_args=cls_args, max_non_unique_correspondences=5,
                                  scaling_factor=scaling_factor, point_choice=2,
                                  num_of_ransac_iter=100, random_pairing=0,
                                  subsampled_points=300)