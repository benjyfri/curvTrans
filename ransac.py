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
from threedmatch import *
from indoor import *
from modelnet import ModelNetHdf
import transforms
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

from scipy.spatial import KDTree

def middlePoints(pcl1, pcl2, trans_mag=0.5, rot_mag_deg=45, max_dist_from_center=0.2):
    dist_from_center = np.linalg.norm(pcl1, axis=1)
    trans_disp = ( np.sqrt(3) * trans_mag )
    rot_disp = ( np.sin( np.radians(rot_mag_deg) ) * max_dist_from_center )
    full_radial_max_displacement = trans_disp #+ rot_disp
    middle_points_indices = dist_from_center < max_dist_from_center
    kd_tree = KDTree(pcl2)
    correspondences = kd_tree.query_ball_point(pcl1[middle_points_indices], full_radial_max_displacement)
    return correspondences

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
                tri_indices = find_triangles(data1, data2, first_index, remaining_indices, 2*dist_threshold)
                if tri_indices is not None and len(set([first_index, tri_indices[0], tri_indices[1]]))==3:
                    break
            else:
                tri_indices = np.random.choice(N, size=3, replace=False)
                first_index = tri_indices[0]
                if tri_indices is not None and len(set(tri_indices))==3:
                    tri_indices = tri_indices[1:]
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

def ransac_3dmatch(data1, data2, max_iterations=1000, threshold=0.1, min_inliers=2, nn1_dist=0.05, max_thresh=1, sample=None, tri=False):
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
        # Find inliers usin np.matmul(data1, rotation.T) + translation.squeeze()
        inliers1, inliers2 = find_inliers(src_centered, dst_centered, rotation,translation, threshold)

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

def estimate_rigid_transform(src_points, dst_points):
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    H = np.matmul(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1  # Correct reflection handling
        R = np.matmul(Vt.T, U.T)  # Recompute R with modified Vt

    translation = dst_mean - (R @ src_mean)
    return R, translation
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

def test_multi_scale_using_embedding_predator_modelnet(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
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
    test_dataset = test_predator_data(matching=True)
    size = len(test_dataset)
    for i in range(size):
        if i > amount_of_samples:
            break
        if i%10 ==0:
            print(f'------------{i}------------')

        # data = test_dataset.__getitem__(83)
        data = test_dataset.__getitem__(i)
        src_pcd, tgt_pcd, GT_rot, GT_trans, sample = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data['sample']

        chosen_pcl_1 = farthest_point_sampling_o3d(src_pcd, k=amount_of_interest_points)
        chosen_pcl_2 = farthest_point_sampling_o3d(tgt_pcd, k=amount_of_interest_points)

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=src_pcd, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=tgt_pcd, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_1 = emb_1.detach().cpu().numpy()[0]
        emb_2 = emb_2.detach().cpu().numpy()[0]

        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                subsampled_1 = farthest_point_sampling_o3d(src_pcd, k=(int)(len(src_pcd) // scale))
                subsampled_2 = farthest_point_sampling_o3d(tgt_pcd, k=(int)(len(tgt_pcd) // scale))
                # fps_indices_1 = farthest_point_sampling(src_pcd, k=(int)(len(src_pcd) // scale))
                # fps_indices_2 = farthest_point_sampling(tgt_pcd, k=(int)(len(tgt_pcd) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_1, pcl_interest=chosen_pcl_1,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_2, pcl_interest=chosen_pcl_2,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_1 = global_emb_1.detach().cpu().numpy()[0]
                global_emb_2 = global_emb_2.detach().cpu().numpy()[0]

                emb_1 = np.hstack((emb_1, global_emb_1))
                emb_2 = np.hstack((emb_2, global_emb_2))

        if nn_mode==1:
            emb1_indices, emb2_indices = find_closest_points_best_of_resolutions(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
        if nn_mode == 2:
            # emb1_indices, emb2_indices = find_closest_points(emb_1[:500,:], emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
            # emb1_indices, emb2_indices = find_closest_points_beta(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=max_non_unique_correspondences)
            emb1_indices, emb2_indices = find_closest_points_best_buddy_beta(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take), max_non_unique_correspondences=1)
        if nn_mode == 3:
            emb1_indices, emb2_indices = find_closest_points_with_dup(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points*pct_of_points_2_take))
        if nn_mode == 4:
            emb1_indices, emb2_indices = find_closest_points_best_buddy(emb_1, emb_2, num_of_pairs=int(amount_of_interest_points * pct_of_points_2_take),max_non_unique_correspondences=max_non_unique_correspondences)
        centered_points_1 = chosen_pcl_1[emb1_indices, :]
        centered_points_2 = chosen_pcl_2[emb2_indices, :]


        plot_correspondence_with_classification(src_pcd, tgt_pcd, centered_points_1, centered_points_2,
                                                emb_1[emb1_indices, :], emb_2[emb2_indices, :], GT_rot, GT_trans)

        # nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(chosen_pcl_1);
        # closest_neighbor_dist = nbrs.kneighbors(chosen_pcl_1)[0][:, 1];
        # mean_closest_neighbor_dist = np.mean(closest_neighbor_dist)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(chosen_pcl_1);
        closest_neighbor_dist = nbrs.kneighbors(chosen_pcl_1)[0][:, 1];
        mean_closest_neighbor_dist = np.median(closest_neighbor_dist)
        # mean_closest_neighbor_dist *= 2

        failed_ransac = False
        o3d_successful = False

        if o3d_successful == False:
            best_rotation, best_translation, best_num_of_inliers, best_iter, corres, final_threshold = ransac(
                centered_points_1, centered_points_2, max_iterations=num_of_ransac_iter,
                threshold=mean_closest_neighbor_dist, sample=sample, tri=tri,
                min_inliers=4, nn1_dist=mean_closest_neighbor_dist, max_thresh=(2 * mean_closest_neighbor_dist))
            # failed in Ransac
            if best_rotation is None:
                failed_ransac = True
                best_rotation = np.eye(3, dtype=np.float32)
                best_translation = np.zeros((3,), dtype=np.float32)

        # final_inliers_list.append(best_num_of_inliers)
        # iter_2_ransac_convergence.append(best_iter)

        transformed_points1 = np.matmul(src_pcd, best_rotation.T) + best_translation.squeeze()
        rot_trace = np.trace(GT_rot @ best_rotation.T)
        residual_rotdeg = np.arccos(np.clip(0.5 * (rot_trace - 1), -1.0, 1.0)) * 180.0 / np.pi
        losses_rot_list.append(residual_rotdeg)
        translation_loss = np.linalg.norm(GT_trans - best_translation)
        losses_trans_list.append(translation_loss)
        corres_emb_1 = np.where((chosen_pcl_1[:, None] == corres[0]).all(axis=2))[0]
        corres_emb_2 = np.where((chosen_pcl_2[:, None] == corres[1]).all(axis=2))[0]
        plot_correspondence_with_classification(src_pcd, tgt_pcd, corres[0], corres[1],
                                                emb_1[corres_emb_1, :], emb_2[corres_emb_2, :], GT_rot, GT_trans, title=f'LOSS: {residual_rotdeg}; ')
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
                'rotation_matrix': GT_rot,
                'translation': GT_trans,
                'best_rotation': best_rotation,
                'best_translation': best_translation
            })


    return worst_losses, losses_rot_list, losses_trans_list, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict
def test_multi_scale_using_embedding_predator_3dmatch(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, max_non_unique_correspondences=3, nn_mode=3, pct_of_points_2_take=0.75, amount_of_samples=100, use_o3d_ransac=False,tri=True, thresh_multi=1):
    worst_losses = [(0, None)] * num_worst_losses
    losses_rot_list = []
    losses_trans_list = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    good_correspondences = []
    combined_dict = {}
    train_dataset = IndoorDataset(data_augmentation=True)
    size = len(train_dataset)
    for i in range(size):
        if i > amount_of_samples:
            break
        if i%10 ==0:
            print(f'------------{i}------------')

        data = train_dataset.__getitem__(i)
        src_pcd, tgt_pcd, GT_rot, GT_trans = data[0].astype(np.float32), data[1].astype(np.float32), data[4], data[5]

        '''
        (src_pcd @ GT_rot.T) + GT_trans.T = tgt_pcd
        '''
        # chosen_fps_indices_1 = farthest_point_sampling(src_pcd, k=amount_of_interest_points)
        # chosen_pcl_1 = src_pcd[chosen_fps_indices_1, :]
        # chosen_fps_indices_2 = farthest_point_sampling(tgt_pcd, k=amount_of_interest_points)
        # chosen_pcl_2 = tgt_pcd[chosen_fps_indices_2, :]

        chosen_pcl_1 = farthest_point_sampling_o3d(src_pcd, k=amount_of_interest_points)
        chosen_pcl_2 = farthest_point_sampling_o3d(tgt_pcd, k=amount_of_interest_points)

        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=src_pcd, pcl_interest=chosen_pcl_1, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=tgt_pcd, pcl_interest=chosen_pcl_2, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_1 = emb_1.detach().cpu().numpy()[0]
        emb_2 = emb_2.detach().cpu().numpy()[0]

        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                if ((int)(len(src_pcd) // scale)<41) or ((int)(len(tgt_pcd) // scale)<41):
                    break
                subsampled_1 = farthest_point_sampling_o3d(src_pcd, k=(int)(len(src_pcd) // scale))
                subsampled_2 = farthest_point_sampling_o3d(tgt_pcd, k=(int)(len(tgt_pcd) // scale))
                # fps_indices_1 = farthest_point_sampling(src_pcd, k=(int)(len(src_pcd) // scale))
                # fps_indices_2 = farthest_point_sampling(tgt_pcd, k=(int)(len(tgt_pcd) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_1, pcl_interest=chosen_pcl_1,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_2, pcl_interest=chosen_pcl_2,
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

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(centered_points_1);
        closest_neighbor_dist = nbrs.kneighbors(centered_points_1)[0][:, 1];
        mean_closest_neighbor_dist = np.mean(closest_neighbor_dist)
        dist_threshold = mean_closest_neighbor_dist * thresh_multi
        centered_registered = centered_points_1 @ GT_rot.T + GT_trans.squeeze()
        distances_corres = np.linalg.norm(centered_registered- centered_points_2,axis=1)
        amount_of_good_corres = np.count_nonzero(distances_corres<dist_threshold)
        good_correspondences.append(amount_of_good_corres)

        failed_ransac = False
        if use_o3d_ransac:
            o3d_ransace_corres_with_paris_transformation, corres = ransac_pose_estimation_correspondences(chosen_pcl_1,
                                                                                                          chosen_pcl_2,
                                                                                                          (np.vstack((
                                                                                                              emb1_indices,
                                                                                                              emb2_indices))).T,
                                                                                                          mutual=False,
                                                                                                          distance_threshold=dist_threshold,
                                                                                                          ransac_n=3)
            corres[0] = chosen_pcl_1[corres[0]]
            corres[1] = chosen_pcl_2[corres[1]]
            o3d_successful = True
            best_translation = o3d_ransace_corres_with_paris_transformation[:3, 3]
            best_rotation = o3d_ransace_corres_with_paris_transformation[:3, :3]

        else:
            best_rotation, best_translation, best_num_of_inliers, best_iter, corres, final_threshold = ransac_3dmatch(
                centered_points_1, centered_points_2, max_iterations=num_of_ransac_iter,
                threshold=dist_threshold, tri=tri,
                min_inliers=3, nn1_dist=mean_closest_neighbor_dist, max_thresh=(8 * dist_threshold))
            # failed in Ransac
            if best_rotation is None:
                failed_ransac = True
                best_rotation = np.eye(3, dtype=np.float32)
                best_translation = np.zeros((3,), dtype=np.float32)

        # final_inliers_list.append(best_num_of_inliers)
        # iter_2_ransac_convergence.append(best_iter)

        rot_trace = np.trace(GT_rot @ best_rotation.T)
        residual_rotdeg = np.arccos(np.clip(0.5 * (rot_trace - 1), -1.0, 1.0)) * 180.0 / np.pi
        losses_rot_list.append(residual_rotdeg)
        translation_loss = np.linalg.norm(GT_trans - best_translation)
        losses_trans_list.append(translation_loss)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if residual_rotdeg > smallest_loss:
            worst_losses[index_of_smallest_loss] = (residual_rotdeg, {
                'noisy_pointcloud_1': src_pcd,
                'noisy_pointcloud_2': tgt_pcd,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'All_pairs_1': centered_points_1,
                'All_pairs_2': centered_points_2,
                'failed_ransac': failed_ransac,
                'pcl_id': i,
                'rotation_matrix': GT_rot,
                'translation': GT_trans,
                'best_rotation': best_rotation,
                'best_translation': best_translation
            })


    return worst_losses, losses_rot_list, losses_trans_list, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, good_correspondences

def test_pairings_3dmatch(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, max_non_unique_correspondences=3, nn_mode=3, pct_of_points_2_take=0.75, amount_of_samples=100, use_o3d_ransac=False,tri=True, thresh_multi=1):
    train_dataset = IndoorDataset(data_augmentation=False)
    size = len(train_dataset)
    for i in range(1):
        if i > amount_of_samples:
            break
        if i%10 ==0:
            print(f'------------{i}------------')

        data = train_dataset.__getitem__(10)
        src_pcd, tgt_pcd, GT_rot, GT_trans = data[0].astype(np.float32), data[1].astype(np.float32), data[4], data[5]

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
                if ((int)(len(src_pcd) // scale)<41) or ((int)(len(tgt_pcd) // scale)<41):
                    break
                subsampled_1 = farthest_point_sampling_o3d(src_pcd, k=(int)(len(src_pcd) // scale))
                subsampled_2 = farthest_point_sampling_o3d(tgt_pcd, k=(int)(len(tgt_pcd) // scale))
                # fps_indices_1 = farthest_point_sampling(src_pcd, k=(int)(len(src_pcd) // scale))
                # fps_indices_2 = farthest_point_sampling(tgt_pcd, k=(int)(len(tgt_pcd) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_1, pcl_interest=chosen_pcl_1,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_2, pcl_interest=chosen_pcl_2,
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
        save_4_point_clouds(centered_points_1, centered_points_2, centered_points_1, centered_points_2, filename=f"2_{receptive_field}_loss.html", rotation=GT_rot, translation=GT_trans, dist_thresh=5)
        cls_1 = np.argmax(emb_1[:,:4], axis=1)
        cls_2 = np.argmax(emb_2[:,:4], axis=1)
        mask_planes = (cls_1[emb1_indices] !=0)
        save_4_point_clouds(centered_points_1, centered_points_2, centered_points_1[mask_planes], centered_points_2[mask_planes],
                            filename=f"2_no_plane_{receptive_field}_loss.html", rotation=GT_rot, translation=GT_trans, dist_thresh=5)

def test_pairings_modelnet(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, max_non_unique_correspondences=3, nn_mode=3, pct_of_points_2_take=0.75, amount_of_samples=100, use_o3d_ransac=False,tri=True, thresh_multi=1):

    test_dataset = test_predator_data()
    size = len(test_dataset)
    for i in range(1):
        if i > amount_of_samples:
            break
        if i%10 ==0:
            print(f'------------{i}------------')

        data = test_dataset.__getitem__(10)
        src_pcd, tgt_pcd, GT_rot, GT_trans, sample = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data[
            'sample']

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
                if ((int)(len(src_pcd) // scale)<41) or ((int)(len(tgt_pcd) // scale)<41):
                    break
                subsampled_1 = farthest_point_sampling_o3d(src_pcd, k=(int)(len(src_pcd) // scale))
                subsampled_2 = farthest_point_sampling_o3d(tgt_pcd, k=(int)(len(tgt_pcd) // scale))
                # fps_indices_1 = farthest_point_sampling(src_pcd, k=(int)(len(src_pcd) // scale))
                # fps_indices_2 = farthest_point_sampling(tgt_pcd, k=(int)(len(tgt_pcd) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_1, pcl_interest=chosen_pcl_1,
                                              args_shape=cls_args, scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_2, pcl_interest=chosen_pcl_2,
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
        save_4_point_clouds(centered_points_1, centered_points_2, centered_points_1, centered_points_2, filename=f"modelnet_2_{receptive_field}_loss.html", rotation=GT_rot, translation=GT_trans, dist_thresh=5)
        cls_1 = np.argmax(emb_1[:,:4], axis=1)
        cls_2 = np.argmax(emb_2[:,:4], axis=1)
        mask_planes = (cls_1[emb1_indices] !=0)
        save_4_point_clouds(centered_points_1, centered_points_2, centered_points_1[mask_planes], centered_points_2[mask_planes],
                            filename=f"modelnet_2_no_plane_{receptive_field}_loss.html", rotation=GT_rot, translation=GT_trans, dist_thresh=5)
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
    if os.name == 'nt':
        root = r'C:\\Users\\benjy\\Desktop\\curvTrans\\DeepBBS\\modelnet40_ply_hdf5_2048'
    else:
        root = r'/content/curvTrans/DeepBBS/modelnet40_ply_hdf5_2048'
    test_dataset = ModelNetHdf(overlap_radius=overlap_radius, root=root,
                                subset='train', categories=None, transform=train_transforms, matching=matching)
    return test_dataset

