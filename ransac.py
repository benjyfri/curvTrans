from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import open3d as o3d
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
from benchmark_modelnet import *
import platform
from modelnet import ModelNetHdf
import transforms
from scipy.spatial import distance_matrix

def load_data(partition='test', divide_data=1):
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

def farthest_point_sampling_torch_old(point_cloud, k):
    N, _ = point_cloud.shape

    # Array to hold indices of sampled points
    sampled_indices = torch.zeros(k, dtype=torch.long)

    # Initialize distances to a large value
    distances = torch.full((N,), float('inf'), device=point_cloud.device)

    # Randomly select the first point
    current_index = torch.randint(N, (1,), device=point_cloud.device).item()
    sampled_indices[0] = current_index

    for i in range(1, k):
        # Update distances to the farthest point selected so far
        current_point = point_cloud[current_index]
        new_distances = torch.norm(point_cloud - current_point, dim=1)
        distances = torch.minimum(distances, new_distances)

        # Select the point that has the maximum distance to the sampled points
        current_index = torch.argmax(distances).item()
        sampled_indices[i] = current_index

    return sampled_indices

def farthest_point_sampling_torch(point_cloud_batch, k):
    batch_size, N, _ = point_cloud_batch.shape

    # Array to hold indices of sampled points for each batch
    sampled_indices = torch.zeros((batch_size, k), dtype=torch.long, device=point_cloud_batch.device)

    # Initialize distances to a large value
    distances = torch.full((batch_size, N), float('inf'), device=point_cloud_batch.device)

    # Randomly select the first point for each batch
    current_indices = torch.randint(N, (batch_size,), device=point_cloud_batch.device)
    sampled_indices[:, 0] = current_indices

    batch_indices = torch.arange(batch_size, device=point_cloud_batch.device)

    for i in range(1, k):
        # Update distances to the farthest point selected so far
        current_points = point_cloud_batch[batch_indices, current_indices]
        new_distances = torch.norm(point_cloud_batch - current_points[:, None, :], dim=2)
        distances = torch.minimum(distances, new_distances)

        # Select the point that has the maximum distance to the sampled points
        current_indices = torch.argmax(distances, dim=1)
        sampled_indices[:, i] = current_indices

    return sampled_indices
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

def find_triangles(data1, data2, first_index, remaining_indices, dist_threshold):
    dist_diff = np.abs(np.linalg.norm(data1[first_index] - data1[remaining_indices], axis=1) -
                       np.linalg.norm(data2[first_index] - data2[remaining_indices], axis=1))

    valid_indices = remaining_indices[dist_diff < dist_threshold]

    if len(valid_indices) < 2:
        return None

    second_index = valid_indices[0]
    last_index_candidates = valid_indices[1:]

    # Calculate distances for the second point
    dist_diff_second = np.abs(np.linalg.norm(data1[second_index] - data1[last_index_candidates], axis=1) -
                              np.linalg.norm(data2[second_index] - data2[last_index_candidates], axis=1))

    valid_last_indices = last_index_candidates[dist_diff_second < dist_threshold]

    if len(valid_last_indices) == 0:
        return None
    last_index = np.random.choice(valid_last_indices)
    return [second_index, last_index]
def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_
def ransac_pose_estimation_correspondences(src_pcd, tgt_pcd, correspondences, mutual=False, distance_threshold=0.05,
                                           ransac_n=3):
    '''
    Run RANSAC estimation based on input correspondences
    :param src_pcd:
    :param tgt_pcd:
    :param correspondences:
    :param mutual:
    :param distance_threshold:
    :param ransac_n:
    :return:
    '''

    # ransac_n = correspondences.shape[0]

    if mutual:
        raise NotImplementedError
    else:
        # src_pcd = src_pcd.cuda()
        # tgt_pcd = tgt_pcd.cuda()
        # correspondences = correspondences.cuda()
        src_pcd = to_o3d_pcd((src_pcd))
        tgt_pcd = to_o3d_pcd((tgt_pcd))
        correspondences = o3d.utility.Vector2iVector((correspondences))

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(src_pcd, tgt_pcd,
                                                                                               correspondences,
                                                                                               distance_threshold,
                                                                                               o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                                   False), ransac_n, [
                                                                                                   o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                                                                                                       0.9),
                                                                                                   o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                                                                                       distance_threshold)],
                                                                                               o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                                   50000, 1000))
    return result_ransac.transformation,[np.array(result_ransac.correspondence_set)[:,0], np.array(result_ransac.correspondence_set)[:,1]]

def ransac_pose_estimation_features(src_pcd, tgt_pcd, source_feature, tgt_feature, mutual=False, distance_threshold=0.05,
                                           ransac_n=3):

    if mutual:
        raise NotImplementedError
    else:
        src_pcd = to_o3d_pcd((src_pcd))
        tgt_pcd = to_o3d_pcd((tgt_pcd))
        reg_module = o3d.pipelines.registration
        pcd1_feat = reg_module.Feature()
        pcd1_feat.data = source_feature
        pcd2_feat = reg_module.Feature()
        pcd2_feat.data = tgt_feature

        result_ransac = reg_module.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, pcd1_feat, pcd2_feat, mutual, distance_threshold,
            reg_module.TransformationEstimationPointToPoint(False), ransac_n, [
                reg_module.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg_module.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            reg_module.RANSACConvergenceCriteria(50000, 1000)
        )

    return result_ransac.transformation,[np.array(result_ransac.correspondence_set)[:,0], np.array(result_ransac.correspondence_set)[:,1]]

def ransac(data1, data2, max_iterations=1000, threshold=0.1, min_inliers=2, nn1_dist=0.05, max_thresh=1, sample=None, tri=False):
    # failing badly....
    if threshold > max_thresh:
        print(f'RANSAC FAIL!')
        print(f'threshold: {threshold}; max_thresh: {max_thresh}')
        return None, None, 0, 1000, [np.array([[1,1,1]]),np.array([[1,1,1]])], threshold
    N = data1.shape[0]
    best_inliers = None
    inliers1 = None
    corres = None
    best_rotation = None
    best_translation = None

    src_centered = data1
    dst_centered = data2
    best_iter = 0
    # print(f'++++++++++++RNASAC with threshold: {threshold} ++++++++++++')
    for iteration in range(max_iterations):
        # Randomly sample 3 corresponding points
        indices = np.arange(N)
        dist_threshold = nn1_dist
        for i in range(100):
            if tri==True:
                first_index = np.random.choice(indices)
                remaining_indices = indices[indices != first_index]
                remaining_indices = np.random.permutation(remaining_indices)
                tri_indices = find_triangles(data1, data2, first_index, remaining_indices, dist_threshold)
                if tri_indices is not None and len(set([first_index, tri_indices[0], tri_indices[1]]))==3:
                    break
            else:
                tri_indices = np.random.choice(N, size=3, replace=False)
                first_index = tri_indices[0]
                tri_indices = tri_indices[1:]
                if tri_indices is not None and len(set(tri_indices))==3:
                    break
        if tri_indices is None:
            continue

        [second_index, last_index] = tri_indices
        indices = np.array([first_index, second_index, last_index])

        # indices = np.random.choice(indices,3 )
        src_points = data1[indices]
        dst_points = data2[indices]

        # indices = np.random.choice(N, size=3, replace=False)
        # src_points = src_centered[indices]
        # dst_points = dst_centered[indices]

        # Estimate rotation and translation
        rotation, translation = estimate_rigid_transform(src_points, dst_points)

        if np.max(np.abs(translation)) > 0.75:
            continue
        r_pred_euler_deg = dcm2euler(np.array([rotation]), seq='xyz')
        # check if magnitude of movement is too big for current setup
        if np.max((r_pred_euler_deg)) > 0 :
            continue
        if np.max(np.abs(r_pred_euler_deg)) > 45:
            continue
        # Find inliers
        inliers1, inliers2 = find_inliers(src_centered, dst_centered, rotation,translation, threshold)

        if sample is not None:
            best_rot_trans = np.hstack((rotation, translation.reshape(3, 1)))
            metrics = compute_metrics({key: torch.tensor(np.expand_dims(val, axis=0)) for key, val in sample.items()},
                                      torch.tensor(np.expand_dims(best_rot_trans, axis=0), dtype=torch.float32))
            rot_loss = metrics['err_r_deg']
            if rot_loss < 5:
                print(f'YAY!! rot loss is: {rot_loss}; inliers={len(inliers1)}')

        # Update best model if we have enough inliers
        if len(inliers1) >= min_inliers and ((best_inliers is None) or (len(inliers1) > len(best_inliers))):
            best_inliers = inliers1
            corres = [src_centered[inliers1], dst_centered[inliers2]]
            best_iter = iteration
            best_rotation = rotation
            best_translation = translation
    if best_inliers is None:
        res = ransac(data1, data2, max_iterations=max_iterations, threshold=threshold * 2, min_inliers=min_inliers, max_thresh=max_thresh)
        if res[0] is None:
            if inliers1 is not None:
                return best_rotation, best_translation, 3, best_iter, [src_centered[inliers1], dst_centered[inliers2]], threshold
            return None, None, 3, 1, [np.array([[1,1,1]]),np.array([[1,1,1]])], threshold
        return res

    # improve registration with LS
    highest_consensus = len(best_inliers)
    while (True):
        rotation, translation = estimate_rigid_transform(corres[0], corres[1])
        #If LS smoothing results in very different registration which is out of range rerun
        r_pred_euler_deg = dcm2euler(np.array([rotation]), seq='xyz')
        # check if magnitude of movement is too big for current setup
        if (np.max(np.abs(translation)) > 0.75) or (np.max((r_pred_euler_deg)) > 0) or (np.max(np.abs(r_pred_euler_deg)) > 45):
            # return best_rotation, best_translation, len(best_inliers), best_iter, corres, threshold
            res =  ransac(data1, data2, max_iterations=max_iterations, threshold=threshold * 2, min_inliers=min_inliers, max_thresh=max_thresh)
            if res[0] is None:
                return best_rotation, best_translation, len(best_inliers), best_iter, corres, threshold
            return res
        # if (np.max(np.abs(translation)) > 0.5) or (np.max(np.abs(dcm2euler(np.array([rotation]), seq='xyz'))) > 45):
        #     return ransac(data1, data2, max_iterations=max_iterations, threshold=threshold * 2, min_inliers=min_inliers)
        best_rotation = rotation
        best_translation = translation
        inliers1, inliers2 = find_inliers(src_centered, dst_centered, best_rotation, best_translation,
                                                threshold)
        best_inliers = inliers1
        corres = [src_centered[inliers1], dst_centered[inliers2]]
        if len(best_inliers) > highest_consensus:
            highest_consensus = len(best_inliers)
        else:
            break
    return best_rotation, best_translation, len(best_inliers), best_iter, corres, threshold

def rand_ransac(data1, data2, max_iterations=1000, threshold=0.1, min_inliers=2, rot=None, trans=None):
    N = data1.shape[0]
    best_inliers = None
    best_rotation = None
    corres = None
    best_translation = None

    src_centered = data1
    dst_centered = data2
    best_iter = 0
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data2)
    if rot is None:
        for iteration in range(max_iterations):
            # Randomly sample 3 corresponding points
            indices_src = np.random.choice(N, size=3, replace=False)
            indices_dst = np.random.choice(N, size=3, replace=False)
            src_points = src_centered[indices_src]
            dst_points = dst_centered[indices_dst]

            # Estimate rotation and translation
            rotation, translation = estimate_rigid_transform(src_points, dst_points)
            # Find inliers
            inliers1, inliers2 = find_inliers_no_correspondence(src_centered, dst_centered, rotation,translation, threshold, nbrs=neighbors)

            # Update best model if we have enough inliers
            if len(inliers1) >= min_inliers and (best_inliers is None or len(inliers1) > len(best_inliers)):
                best_inliers = inliers1
                corres = [src_centered[inliers1], dst_centered[inliers2]]
                best_rotation = rotation
                best_translation = translation
                best_iter = iteration
        if best_inliers is None:
            return rand_ransac(data1, data2, max_iterations=max_iterations, threshold=threshold + 0.1, min_inliers=min_inliers)
    else:
        inliers1, inliers2 = find_inliers_no_correspondence(src_centered, dst_centered, rot, trans.squeeze(), threshold)
        best_inliers = inliers1
        best_rotation = rot
        best_translation = trans
        corres = [src_centered[inliers1], dst_centered[inliers2]]
        best_iter = 1
        return best_rotation, best_translation, len(best_inliers), best_iter, corres, threshold
    # improve registration with LS
    highest_consensus = len(best_inliers)
    while (True):
        best_rotation, best_translation = estimate_rigid_transform(corres[0], corres[1])
        inliers1, inliers2 = find_inliers_no_correspondence(src_centered, dst_centered, best_rotation, best_translation,
                                                            threshold, nbrs=neighbors)

        best_inliers = inliers1
        corres = [src_centered[inliers1], dst_centered[inliers2]]
        if len(best_inliers) > highest_consensus:
            highest_consensus = len(best_inliers)
        else:
            break
    return best_rotation, best_translation, len(best_inliers), best_iter, corres, threshold

def ransac_torch(data1, data2, max_iterations=1000, threshold=0.1, min_inliers=2):
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
    if N<3:
        raise Exception("Not enough corresponding points to perform RANSAC.")
    best_inliers = None
    corres = None
    src_centered = data1
    dst_centered = data2
    dists = torch.norm((src_centered-dst_centered), dim=1)
    best_iter = 0
    for iteration in range(max_iterations):
        # Randomly sample 3 corresponding points
        counter = 0
        smallest_diff = float('inf')
        best_src_points = None
        best_dst_points = None
        while True:
            counter += 1
            indices = np.random.choice(N, size=3, replace=False)
            src_points = src_centered[indices]
            dst_points = dst_centered[indices]
            dist_1, dist_2, dist_3 = dists[indices]
            largest_diff = max(dist_1, dist_2, dist_2) - min(dist_1, dist_2, dist_2)
            if largest_diff < smallest_diff:
                smallest_diff = largest_diff
                best_src_points = src_points
                best_dst_points = dst_points

            if largest_diff < 0.05 or counter > 50:
                src_points = best_src_points
                dst_points = best_dst_points
                break

        # Estimate rotation and translation
        rotation, translation = estimate_rigid_transform_torch(src_points, dst_points)
        # Find inliers
        inliers1, inliers2 = find_inliers_torch(src_centered, dst_centered, rotation, translation, threshold)

        # Update best model if we have enough inliers
        if len(inliers1) >= min_inliers and (best_inliers is None or len(inliers1) > len(best_inliers)):
            best_inliers = inliers1
            corres = [src_centered[inliers1], dst_centered[inliers2]]
            best_iter = iteration
    if best_inliers == None:
        return ransac_torch(data1, data2, max_iterations=max_iterations, threshold=threshold + 0.1, min_inliers=min_inliers)

    #improve registration with LS
    highest_consensus = len(best_inliers)
    while(True):
        best_rotation, best_translation = estimate_rigid_transform_torch(corres[0], corres[1])
        inliers1, inliers2 = find_inliers_torch(src_centered, dst_centered, best_rotation, best_translation, threshold)
        best_inliers = inliers1
        corres = [src_centered[inliers1], dst_centered[inliers2]]
        if len(best_inliers) > highest_consensus:
            highest_consensus = len(best_inliers)
        else:
            break
    return best_rotation, best_translation, highest_consensus, best_iter, corres, threshold
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
        rotation, translation  = estimate_rigid_transform(src_points, dst_points)
        # Find inliers
        inliers_1, inliers_2 = find_inliers_classification_multiclass(cls_1, cls_2, rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return multiclass_classification_only_ransac(cls_1, cls_2, max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_translation, best_num_of_inliers, best_iter, corres, threshold

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
        rotation, translation  = estimate_rigid_transform(src_points, dst_points)
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
    return best_rotation, best_translation, best_num_of_inliers, best_iter, corres, threshold


def estimate_rigid_transform(src_points, dst_points):
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    H = np.matmul(src_centered.T, dst_centered)
    # H = np.matmul(src_points.T, dst_points)
    U, _, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt, U.T)
    translation = dst_mean - ( R @ src_mean )
    return R, translation



def estimate_rigid_transform_torch(src_points, dst_points):
    src_mean = torch.mean(src_points, dim=0)
    dst_mean = torch.mean(dst_points, dim=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    H = torch.matmul(src_centered.t(), dst_centered)

    U, _, Vt = torch.linalg.svd(H)
    R = torch.matmul(Vt.t(), U.t())

    if torch.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = torch.matmul(Vt.t(), U.t())

    translation = dst_mean - torch.matmul(R, src_mean)

    return R, translation

#TODO: Must fix translation!!!
def find_inliers(data1, data2, rotation, translation, threshold):
    """
    Finds the inliers in two sets of 3D points given a rotation and translation.
    """

    transformed = np.matmul(data1, rotation.T) + translation.squeeze()
    dist = np.linalg.norm(transformed - data2, axis=1)
    mask = dist < threshold
    inliers1_new = (np.arange(len(data1)))[mask.squeeze()]
    inliers2_new = inliers1_new

    return inliers1_new, inliers2_new
def find_inliers_no_correspondence(data1, data2, rotation, translation, threshold, nbrs=None):
    """
    Finds the inliers in two sets of 3D points given a rotation and translation.
    """
    transformed_src = np.matmul(data1, rotation.T) + translation.squeeze()
    if nbrs is None:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data2)
    distances, indices = nbrs.kneighbors(transformed_src)
    mask = distances < threshold
    inliers1 = (np.arange(len(data1)))[mask.squeeze()]
    inliers2 = indices.squeeze()[mask.squeeze()]
    return inliers1, inliers2

def find_inliers_torch(data1, data2, rotation, translation, threshold):
    """
    Finds the inliers in two sets of 3D points given a rotation and translation.
    """
    inliers1 = []
    inliers2 = []
    for i in range(data1.shape[0]):
        point1 = data1[i]
        point2 = data2[i]

        transformed = (torch.matmul(point1, rotation.t())) + translation
        distance = torch.norm(transformed - point2)
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
            pcl1, pcl2, pcl1_indices, pcl2_indices, overlapping_indices = create_pcls_func(pcl1)
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
                                    num_of_ransac_iter=100, shapes=[86, 162, 174, 176, 179], create_pcls_func=None, max_non_unique_correspondences=3, pct_of_points_2_take=0.75):
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
            pcl1, pcl2, pcl1_indices, pcl2_indices, overlapping_indices = create_pcls_func(pcl1)
            rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl2)

        noisy_pointcloud_1 = pcl1 + np.random.normal(0, 0.01, pcl1.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        chosen_fps_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=amount_of_interest_points)
        chosen_pcl_1 = noisy_pointcloud_1[chosen_fps_indices_1, :]
        chosen_fps_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=amount_of_interest_points)
        chosen_pcl_2 = noisy_pointcloud_2[chosen_fps_indices_2, :]

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor)


        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()

        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                fps_indices_1 = farthest_point_sampling(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1)//scale))
                fps_indices_2 = farthest_point_sampling(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2)//scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_1[fps_indices_1,:], pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=noisy_pointcloud_2[fps_indices_2,:], pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor)


                global_emb_1 = global_emb_1.detach().cpu().numpy()
                global_emb_2 = global_emb_2.detach().cpu().numpy()

                emb_1 = np.hstack((emb_1, global_emb_1))
                emb_2 = np.hstack((emb_2, global_emb_2))

        # emb1_indices, emb2_indices = find_closest_points(emb_1, emb_2, num_neighbors=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        emb1_indices, emb2_indices = find_closest_points_best_buddy(emb_1, emb_2, num_neighbors=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        centered_points_1 = chosen_pcl_1[emb1_indices, :]
        centered_points_2 = chosen_pcl_2[emb2_indices, :]

        best_rotation, best_translation, best_num_of_inliers, best_iter, corres, final_threshold = ransac(centered_points_1, centered_points_2, max_iterations=num_of_ransac_iter,
                                                   threshold=0.1,
                                                   min_inliers=(int)((amount_of_interest_points*pct_of_points_2_take) // 10))

        final_inliers_list.append(best_num_of_inliers)
        iter_2_ransac_convergence.append(best_iter)

        transformed_points1 = np.matmul(noisy_pointcloud_1, best_rotation.T) + best_translation
        loss = np.mean(((rotation_matrix @ best_rotation.T) - np.eye(3)) ** 2)
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


def test_random_ransac(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, shapes=[86, 162, 174, 176, 179], create_pcls_func=None, max_non_unique_correspondences=3, pct_of_points_2_take=0.75):
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
        # rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl1, np.array([0.1,0.4, 0.2]))
        rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl1)

        noisy_pointcloud_1 = pcl1 + np.random.normal(0, 0.01, pcl1.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        best_rotation, best_translation, best_num_of_inliers, best_iter, corres, final_threshold = rand_ransac(
            noisy_pointcloud_1, noisy_pointcloud_2, max_iterations=num_of_ransac_iter,
            threshold=0.1,
            min_inliers=(int)((amount_of_interest_points * pct_of_points_2_take) // 10))
        iter_2_ransac_convergence.append(best_iter)

        transformed_points1 = np.matmul(noisy_pointcloud_1, best_rotation.T) + best_translation
        loss = np.mean(((rotation_matrix @ best_rotation.T) - np.eye(3)) ** 2)
        losses.append(loss)

        point_distance = np.mean(0)
        point_distance_list.append(0)

    return worst_losses, losses, final_thresh_list, final_inliers_list, point_distance_list, worst_point_losses, iter_2_ransac_convergence


def test_multi_scale_using_embedding_predator_old(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, max_non_unique_correspondences=3, pct_of_points_2_take=0.75, amount_of_samples=100, batch_size=4):
    worst_losses = [(0, None)] * num_worst_losses
    losses_rot_list = []
    losses_trans_list = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    combined_dict = {}
    test_dataset = test_predator_data()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for batch_idx, data in enumerate(test_dataloader):
        # if batch_idx%10 ==0:
        #     print(f'------------{batch_idx}------------')
        print(f'------------{batch_idx * batch_size}------------')

        src_pcd, tgt_pcd, rot, trans, sample = (data['src_pcd'].squeeze().to(device), data['tgt_pcd'].squeeze().to(device), data['rot'].squeeze().to(device),
                                                               data['trans'].squeeze().to(device), data['sample'])
        # src_pcd, tgt_pcd, rot, trans, matching_inds, sample = (data['src_pcd'].squeeze().to(device), data['tgt_pcd'].squeeze().to(device), data['rot'].squeeze().to(device),
        #                                                        data['trans'].squeeze().to(device), data['matching_inds'].squeeze().to(device), data['sample'])
        if batch_size == 1:
            src_pcd = src_pcd.unsqueeze(0)
            tgt_pcd = tgt_pcd.unsqueeze(0)
            rot = rot.unsqueeze(0)
            trans = trans.unsqueeze(0)


        chosen_fps_indices_1 = farthest_point_sampling_torch(src_pcd, k=amount_of_interest_points)
        chosen_pcl_1 = src_pcd[torch.arange(batch_size).unsqueeze(1).expand(batch_size, amount_of_interest_points), chosen_fps_indices_1]
        chosen_fps_indices_2 = farthest_point_sampling_torch(tgt_pcd, k=amount_of_interest_points)
        chosen_pcl_2 = tgt_pcd[torch.arange(batch_size).unsqueeze(1).expand(batch_size, amount_of_interest_points), chosen_fps_indices_2]


        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=src_pcd, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor, device=device)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=tgt_pcd, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor, device=device)

        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                fps_indices_1 = farthest_point_sampling_torch(src_pcd, k=(int)(src_pcd.shape[1]//scale))
                fps_indices_2 = farthest_point_sampling_torch(tgt_pcd, k=(int)(tgt_pcd.shape[1]//scale))

                downsampled_pcl_1 = src_pcd[torch.arange(batch_size).unsqueeze(1).expand(batch_size,
                                                                                    (int)(src_pcd.shape[1]//scale)), fps_indices_1]

                downsampled_pcl_2 = tgt_pcd[torch.arange(batch_size).unsqueeze(1).expand(batch_size,
                                                                                    (int)(tgt_pcd.shape[1]//scale)), fps_indices_2]

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=downsampled_pcl_1, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor, device=device)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                            pcl_src=downsampled_pcl_2, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor, device=device)

                emb_1 = torch.cat((emb_1, global_emb_1), dim=-1)
                emb_2 = torch.cat((emb_2, global_emb_2), dim=-1)


        # emb1_indices, emb2_indices = find_closest_points_torch(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences,topk=1)
        emb1_indices, emb2_indices = find_closest_points_best_buddy(emb_1, emb_2, num_neighbors=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        batch_size, num_points = emb1_indices.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_points)
        centered_points_1 = chosen_pcl_1[batch_indices, emb1_indices]
        centered_points_2 = chosen_pcl_2[batch_indices, emb2_indices]
        rot_trans_list = []
        for b in range(batch_size):
            best_rotation, best_translation, best_num_of_inliers, best_iter, corres, final_threshold = ransac_torch(centered_points_1[b], centered_points_2[b], max_iterations=num_of_ransac_iter,
                                                       threshold=0.1,
                                                       min_inliers=(int)((amount_of_interest_points*pct_of_points_2_take) // 10))

            best_rot_trans = torch.cat((best_rotation, best_translation.unsqueeze(1)), dim=1)
            rot_trans_list.append(best_rot_trans)
            final_inliers_list.append(best_num_of_inliers)
            iter_2_ransac_convergence.append(best_iter)

            # transformed_points1 = torch.matmul(src_pcd, best_rotation.T) + best_translation
            loss_rot = torch.mean(((rot[b] @ best_rotation.T) - torch.eye(3, device=device)) ** 2)
            criterion = torch.nn.MSELoss()
            loss_trans = criterion(trans[b], best_translation)
            losses_rot_list.append(loss_rot.item())
            losses_trans_list.append(loss_trans.item())

            # Update the worst losses list
            index_of_smallest_loss = torch.argmin(torch.tensor([worst_losses[i][0] for i in range(len(worst_losses))]))
            smallest_loss = worst_losses[index_of_smallest_loss][0]
            if loss_rot > smallest_loss:
                worst_losses[index_of_smallest_loss] = (loss_rot, {
                    'noisy_pointcloud_1': src_pcd[b],
                    'noisy_pointcloud_2': tgt_pcd[b],
                    'chosen_points_1': corres[0],
                    'chosen_points_2': corres[1],
                    'rotation_matrix': rot[b],
                    'best_rotation': best_rotation,
                    'best_translation': best_translation
                })
        rot_trans_tensor = torch.stack(rot_trans_list, dim=0)
        metrics = compute_metrics(sample, rot_trans_tensor)
        if len(combined_dict) > 0:
            for key in metrics.keys():
                combined_dict[key] = np.concatenate(( combined_dict[key] , metrics[key]))
        else:
            combined_dict = metrics

        if ((batch_idx+1) * batch_size) > amount_of_samples:
            break
    return worst_losses, losses_rot_list, losses_trans_list, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict

def test_multi_scale_using_embedding_predator(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, max_non_unique_correspondences=3, nn_mode=3, pct_of_points_2_take=0.75, amount_of_samples=100, batch_size=4,tri=False):
    worst_losses = [(0, None)] * num_worst_losses
    losses_rot_list = []
    losses_trans_list = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    combined_dict = {}
    # test_dataset = test_predator_data(partial_p_keep= [0.5, 0.5])
    test_dataset = test_predator_data()
    size = len(test_dataset)
    for i in range(size):
        if i > amount_of_samples:
            break
        if i%10 ==0:
            print(f'------------{i}------------')

        data = test_dataset.__getitem__(i)
        src_pcd, tgt_pcd, rot, trans, sample = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data['sample']

        chosen_fps_indices_1 = farthest_point_sampling(src_pcd, k=amount_of_interest_points)
        chosen_pcl_1 = src_pcd[chosen_fps_indices_1, :]
        chosen_fps_indices_2 = farthest_point_sampling(tgt_pcd, k=amount_of_interest_points)
        chosen_pcl_2 = tgt_pcd[chosen_fps_indices_2, :]

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=src_pcd, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=tgt_pcd, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_1 = emb_1.detach().cpu().numpy()[0]
        emb_2 = emb_2.detach().cpu().numpy()[0]
        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                fps_indices_1 = farthest_point_sampling(src_pcd, k=(int)(len(src_pcd) // scale))
                fps_indices_2 = farthest_point_sampling(tgt_pcd, k=(int)(len(tgt_pcd) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=src_pcd[fps_indices_1, :], pcl_interest=chosen_pcl_1,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=tgt_pcd[fps_indices_2, :], pcl_interest=chosen_pcl_2,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_1 = global_emb_1.detach().cpu().numpy()[0]
                global_emb_2 = global_emb_2.detach().cpu().numpy()[0]

                emb_1 = np.hstack((emb_1, global_emb_1))
                emb_2 = np.hstack((emb_2, global_emb_2))

        if nn_mode==1:
            emb1_indices, emb2_indices = find_closest_points_best_of_resolutions(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        if nn_mode == 2:
            emb1_indices, emb2_indices = find_closest_points(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        if nn_mode == 3:
            emb1_indices, emb2_indices = find_closest_points_with_dup(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take))
        if nn_mode == 4:
            emb1_indices, emb2_indices = find_closest_points_best_buddy(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points * pct_of_points_2_take),max_non_unique_correspondences=max_non_unique_correspondences)
        centered_points_1 = chosen_pcl_1[emb1_indices, :]
        centered_points_2 = chosen_pcl_2[emb2_indices, :]

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(chosen_pcl_1);
        closest_neighbor_dist = nbrs.kneighbors(chosen_pcl_1)[0][:, 1];
        mean_closest_neighbor_dist = np.mean(closest_neighbor_dist)
        # mean_closest_neighbor_dist *= 5


        # o3d_ransace_corres_with_feats_transformation, corres = ransac_pose_estimation_features(chosen_pcl_1, chosen_pcl_2, emb_1, emb_2, mutual=False, distance_threshold=mean_closest_neighbor_dist, ransac_n=3)

        failed_ransac = False
        o3d_successful = False
        # o3d_successful = True
        # for _ in range(5):
        #     o3d_ransace_corres_with_paris_transformation, corres = ransac_pose_estimation_correspondences(chosen_pcl_1,
        #                                                                                                   chosen_pcl_2,
        #                                                                                                   (np.vstack((
        #                                                                                                              emb1_indices,
        #                                                                                                              emb2_indices))).T,
        #                                                                                                   mutual=False,
        #                                                                                                   distance_threshold=mean_closest_neighbor_dist,
        #                                                                                                   ransac_n=3)
        #     corres[0] = chosen_pcl_1[corres[0]]
        #     corres[1] = chosen_pcl_2[corres[1]]
        #     o3d_successful = True
        #     best_translation = o3d_ransace_corres_with_paris_transformation[:3, 3]
        #     if np.max(np.abs(best_translation)) > 0.75:
        #         o3d_successful = False
        #     if o3d_successful == True:
        #         best_rotation = o3d_ransace_corres_with_paris_transformation[:3, :3]
        #         r_pred_euler_deg = dcm2euler(np.array([best_rotation]), seq='xyz')
        #         # check if magnitude of movement is too big for current setup
        #         if np.max((r_pred_euler_deg)) >= 0 :
        #             o3d_successful = False
        #         if np.max(np.abs(r_pred_euler_deg)) > 45:
        #             o3d_successful = False
        #     if o3d_successful == True:
        #         break
        print(o3d_successful)
        # print(o3d_successful)
        if o3d_successful == False:
            best_rotation, best_translation, best_num_of_inliers, best_iter, corres, final_threshold = ransac(
                centered_points_1, centered_points_2, max_iterations=num_of_ransac_iter,
                threshold=mean_closest_neighbor_dist, sample=sample, tri=tri,
                min_inliers=3, nn1_dist=mean_closest_neighbor_dist, max_thresh=(8 * mean_closest_neighbor_dist))
            # failed in Ransac
            if best_rotation is None:
                failed_ransac = True
                best_rotation = np.eye(3, dtype=np.float32)
                best_translation = np.zeros((3,), dtype=np.float32)

        # final_inliers_list.append(best_num_of_inliers)
        # iter_2_ransac_convergence.append(best_iter)

        # transformed_points1 = np.matmul(src_pcd, best_rotation.T) + best_translation.squeeze()
        loss = np.mean(((rot @ best_rotation.T) - np.eye(3)) ** 2)
        losses_rot_list.append(loss)
        losses_trans_list.append(np.linalg.norm(trans - best_translation))

        best_rot_trans = np.hstack((best_rotation, best_translation.reshape(3, 1)))
        metrics = compute_metrics({key: torch.tensor(np.expand_dims(val, axis=0)) for key, val in sample.items()},
                                  torch.tensor(np.expand_dims(best_rot_trans, axis=0), dtype=torch.float32))
        if len(combined_dict) > 0:
            for key in metrics.keys():
                combined_dict[key] = np.concatenate((combined_dict[key], metrics[key]))
        else:
            combined_dict = metrics

        print(final_threshold)
        print(metrics['err_r_deg'][0])
        print(best_num_of_inliers)
        print(f'++++')
        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if metrics['err_r_deg'][0] > smallest_loss:
            worst_losses[index_of_smallest_loss] = (metrics['err_r_deg'][0], {
                'noisy_pointcloud_1': src_pcd,
                'noisy_pointcloud_2': tgt_pcd,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'All_pairs_1': centered_points_1,
                'All_pairs_2': centered_points_2,
                'failed_ransac': failed_ransac,
                'pcl_id': i,
                'rotation_matrix': rot,
                'translation': trans,
                'best_rotation': best_rotation,
                'best_translation': best_translation
            })


    return worst_losses, losses_rot_list, losses_trans_list, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict

def test_embedding_dist_predator(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, max_non_unique_correspondences=3, pct_of_points_2_take=0.75, amount_of_samples=100, batch_size=4):
    mean_dist_list = []
    median_dist_list = []
    mean_distance_list = []
    median_distance_list = []
    mean_closest_neighbor_dist_list = []
    median_closest_neighbor_dist_list = []
    max_distance_list = []
    ten_pct_mean_dist_list = []
    twenty_pct_mean_dist_list = []
    thirty_pct_mean_dist_list = []
    fifty_pct_mean_dist_list = []
    ten_pct_median_dist_list = []

    test_dataset = test_predator_data(matching=True, partial_p_keep=[1.0, 1.0])
    size = len(test_dataset)
    for i in range(size):
        if i > 1:
            break
        if i%10 ==0:
            print(f'------------{i}------------')

        data = test_dataset.__getitem__(162)
        src_pcd, tgt_pcd, rot, trans, sample, matching_inds = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data['sample'], data['matching_inds']
        matching_inds = matching_inds.detach().cpu().numpy()

        chosen_fps_indices_1 = farthest_point_sampling(src_pcd, k=amount_of_interest_points)
        chosen_pcl_1 = src_pcd[chosen_fps_indices_1, :]
        chosen_fps_indices_2 = farthest_point_sampling(tgt_pcd, k=amount_of_interest_points)
        chosen_pcl_2 = tgt_pcd[chosen_fps_indices_2, :]

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=src_pcd, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=tgt_pcd, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_1 = emb_1.detach().cpu().numpy()[0]
        emb_2 = emb_2.detach().cpu().numpy()[0]
        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                fps_indices_1 = farthest_point_sampling(src_pcd, k=(int)(len(src_pcd) // scale))
                fps_indices_2 = farthest_point_sampling(tgt_pcd, k=(int)(len(tgt_pcd) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=src_pcd[fps_indices_1, :], pcl_interest=chosen_pcl_1,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=tgt_pcd[fps_indices_2, :], pcl_interest=chosen_pcl_2,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_1 = global_emb_1.detach().cpu().numpy()[0]
                global_emb_2 = global_emb_2.detach().cpu().numpy()[0]

                emb_1 = np.hstack((emb_1, global_emb_1))
                emb_2 = np.hstack((emb_2, global_emb_2))

        chosen_rotated_1 = (chosen_pcl_1 @ rot.T) + trans.squeeze()
        xyz1_indices, xyz2_indices = find_closest_points_best_buddy(chosen_rotated_1, chosen_pcl_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)

        # emb1_indices, emb2_indices = find_closest_points_best_of_resolutions(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        emb1_indices, emb2_indices = find_closest_points(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), n_neighbors=1)
        # emb1_indices, emb2_indices, dist_emb = find_closest_points_with_dup(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points * pct_of_points_2_take))
        # emb1_indices, emb2_indices = find_closest_points_best_buddy(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points))

        overlapping_chosen_1, ind1, ind2 = np.intersect1d(xyz1_indices, emb1_indices, return_indices=True)
        bbudies_pairings = np.vstack((xyz1_indices[ind1], xyz2_indices[ind1])).T
        overlap_emb_pairings = np.vstack((emb1_indices[ind2],emb2_indices[ind2])).T
        plot_4_point_clouds(chosen_pcl_1, chosen_pcl_2, chosen_pcl_1[bbudies_pairings[:, 0]],
                            chosen_pcl_2[bbudies_pairings[:, 1]], rot, trans[:, 0], "real pairings")
        plot_4_point_clouds(chosen_pcl_1, chosen_pcl_2, chosen_pcl_1[overlap_emb_pairings[:, 0]],
                            chosen_pcl_2[overlap_emb_pairings[:, 1]], rot, trans[:, 0], "emb pairings")
        exact_match = [((x[0]==y[0]) and (x[1]==y[1])) for x,y in zip(bbudies_pairings[bbudies_pairings[:,0].argsort()], overlap_emb_pairings[overlap_emb_pairings[:,0].argsort()])]
        num_exact_matches = np.count_nonzero(exact_match)
        distance_2_real_pairing = np.array([(np.linalg.norm(chosen_pcl_2[x[1]]-chosen_pcl_2[y[1]])) for x,y in zip(bbudies_pairings[bbudies_pairings[:,0].argsort()], overlap_emb_pairings[overlap_emb_pairings[:,0].argsort()])])


        emb_dist_emb = np.linalg.norm(emb_1[emb1_indices[ind2]] - emb_2[emb2_indices[ind2]], axis=1)
        # Create scatter plot
        plt.scatter(emb_dist_emb, distance_2_real_pairing)

        # Add title and labels
        plt.title('Scatter Plot Example')
        plt.xlabel('emb dist')
        plt.ylabel('geo dist')

        # Show plot
        plt.show()
        mean_dist = np.mean(distance_2_real_pairing)
        # median_dist = np.median(distance_2_real_pairing)
        ten_pct_smallest_dist = distance_2_real_pairing[(distance_2_real_pairing <= np.percentile(distance_2_real_pairing, 10))]
        twenty_pct_smallest_dist = distance_2_real_pairing[(distance_2_real_pairing <= np.percentile(distance_2_real_pairing, 20))]
        thirty_pct_smallest_dist = distance_2_real_pairing[(distance_2_real_pairing <= np.percentile(distance_2_real_pairing, 30))]
        fifty_pct_smallest_dist = distance_2_real_pairing[(distance_2_real_pairing <= np.percentile(distance_2_real_pairing, 50))]
        twenty_pct_mean_dist = np.mean(twenty_pct_smallest_dist)
        thirty_pct_mean_dist = np.mean(thirty_pct_smallest_dist)
        fifty_pct_mean_dist =  np.mean(fifty_pct_smallest_dist)
        ten_pct_mean_dist = np.mean(ten_pct_smallest_dist)
        # ten_pct_median_dist = np.median(ten_pct_smallest_dist)
        dist_matrix = distance_matrix(chosen_pcl_2, chosen_pcl_2)
        lower_triangle_indices = np.tril_indices(dist_matrix.shape[0], k=-1)
        lower_triangle = dist_matrix[lower_triangle_indices]
        mean_distance = np.mean(lower_triangle)
        # median_distance = np.median(lower_triangle)
        max_distance = np.max(lower_triangle)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(chosen_pcl_2)
        closest_neighbor_dist = nbrs.kneighbors(chosen_pcl_2)[0][:,1]
        mean_closest_neighbor_dist = np.mean(closest_neighbor_dist)
        median_closest_neighbor_dist = np.median(closest_neighbor_dist)

        mean_dist_list.append(mean_dist)
        # median_dist_list.append(median_dist)
        mean_distance_list.append(mean_distance)
        # median_distance_list.append(median_distance)
        max_distance_list.append(max_distance)
        ten_pct_mean_dist_list.append(ten_pct_mean_dist)
        twenty_pct_mean_dist_list.append(twenty_pct_mean_dist)
        thirty_pct_mean_dist_list.append(thirty_pct_mean_dist)
        fifty_pct_mean_dist_list.append(fifty_pct_mean_dist)
        # ten_pct_median_dist_list.append(ten_pct_median_dist)
        mean_closest_neighbor_dist_list.append(mean_closest_neighbor_dist)
        # median_closest_neighbor_dist_list.append(median_closest_neighbor_dist)
        # Create the plots

    plt.figure(figsize=(12, 8))

    plt.plot(mean_dist_list, label=f'mean_dist_2_real_pairing {np.mean(mean_dist_list):.2f}')
    # plt.plot(median_dist_list, label='median_dist')
    # plt.plot(mean_distance_list, label='mean_dist_orig')
    # plt.plot(median_distance_list, label='median_distance')
    plt.plot(max_distance_list, label=f'shape size {np.mean(max_distance_list):.2f}')
    plt.plot(ten_pct_mean_dist_list, label=f'10%_mean_dist {np.mean(ten_pct_mean_dist_list):.2f}')
    plt.plot(twenty_pct_mean_dist_list, label=f'20%_mean_dist {np.mean(twenty_pct_mean_dist_list):.2f}')
    plt.plot(thirty_pct_mean_dist_list, label=f'30%_mean_dist {np.mean(thirty_pct_mean_dist_list):.2f}')
    plt.plot(fifty_pct_mean_dist_list, label=f'50%_mean_dist {np.mean(fifty_pct_mean_dist_list):.2f}')
    # plt.plot(ten_pct_median_dist_list, label='10%_median_dist')
    plt.plot(mean_closest_neighbor_dist_list, label=f'mean_1NN_dist {np.mean(mean_closest_neighbor_dist_list):.2f}')
    # plt.plot(median_closest_neighbor_dist_list, label='median_closest_neighbor_dist')

    plt.title(f'Distances Over Samples; mean_10% = {np.mean(ten_pct_mean_dist_list):.2f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def test_predator_data(matching=False, partial_p_keep=[0.7, 0.7]):
    # partial_p_keep= [0.5, 0.5]
    rot_mag = 45.0
    trans_mag = 0.5
    num_points = 1024
    overlap_radius = 0.04
    train_transforms = [transforms.SplitSourceRef(),
                        transforms.RandomCrop(partial_p_keep),
                        transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                        transforms.Resampler(num_points),
                        transforms.RandomJitter(),
                        transforms.ShufflePoints()]
    test_dataset = ModelNetHdf(overlap_radius=overlap_radius, root=r'C:\\Users\\benjy\\Desktop\\curvTrans\\DeepBBS\\modelnet40_ply_hdf5_2048',
                                subset='test', categories=None, transform=train_transforms, matching=matching)
    return test_dataset
from threedmatch import *
from indoor import *
import torchvision

import yaml
def test_3dmatch_data(matching=False, partial_p_keep=[0.7, 0.7]):
    # partial_p_keep= [0.5, 0.5]
    train_set = IndoorDataset(data_augmentation=True)
    d = 10
    return train_set