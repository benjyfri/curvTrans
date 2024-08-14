import cProfile
import pstats
import faulthandler
import torch
from plotting_functions import *
from ransac import *
from experiments_utils import *
import numpy as np
from functools import partial
from modelnet import ModelNetHdf
import transforms
import pickle
import os
from benchmark_modelnet import dcm2euler
def checkPred():
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1')
    train_dataset = test_predator_data()
    size = len(train_dataset)
    ov_list = []
    size = 5
    for i in range(size):
        src_pcd, tgt_pcd, rot, trans, matching_inds, src_pcd, tgt_pcd, sample = train_dataset.__getitem__(i+85)
        if matching_inds is not None and len(matching_inds.shape)==2:
            overlap = len(np.unique(matching_inds[:, 0]))
        else:
            overlap = 0
        ov_list.append( overlap / (len(src_pcd)) )
        plot_multiclass_point_clouds([src_pcd, src_pcd[matching_inds[:, 0]]], [tgt_pcd, tgt_pcd[matching_inds[:, 1]]],
                                     rotation=rot, title=f'unique = {len(np.unique(matching_inds[:, 0]))}')

        view_stabiity(cls_args=cls_args,num_worst_losses = 3, scaling_factor=1, scales=5, receptive_field=[1,5, 10], amount_of_interest_points=300,
                                    num_of_ransac_iter=50, plot_graphs=1, given_pcls=[src_pcd,tgt_pcd, matching_inds[10,:]])
        # exit(0)
    exit(0)
    ov_list = np.array(ov_list)
    avg_overlap = np.mean(ov_list)
    print(avg_overlap)
    return ov_list
def test_predator_data():
    partial_p_keep= [0.7, 0.7]
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
    train_dataset = ModelNetHdf(overlap_radius=overlap_radius, root=r'C:\\Users\\benjy\\Desktop\\curvTrans\\DeepBBS\\modelnet40_ply_hdf5_2048',
                                subset='train', categories=None, transform=train_transforms)
    return train_dataset

def load_data(partition='test', divide_data=1):
    DATA_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes\\data'
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


def create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36'):
    cls_args_shape = configArgsPCT()
    cls_args_shape.batch_size = 1024
    cls_args_shape.num_mlp_layers = 3
    cls_args_shape.num_neurons_per_layer = 32
    cls_args_shape.sampled_points = 40
    cls_args_shape.use_second_deg = 1
    cls_args_shape.lpe_normalize = 0
    cls_args_shape.exp = name
    cls_args_shape.lpe_dim = 0
    cls_args_shape.output_dim = 4
    cls_args_shape.use_lap_reorder = 1
    cls_args_shape.lap_eigenvalues_dim = 36
    return cls_args_shape, 1, 1
def split_pointcloud_overlap(point_cloud, overlap_ratio):
    """Splits a point cloud into two with a random overlap ratio.

    Args:
      pointcloud: A numpy array of shape (2048, 3) representing the point cloud.
      r: The overlap ratio between the two split point clouds (0 < r < 1).

    Returns:
      pcl1, pcl2: Two numpy arrays representing the split point clouds.
    """

    if not (0 < overlap_ratio < 1):
        raise ValueError("Overlap ratio r must be between 0 and 1.")

    # Check for empty pointcloud
    if not point_cloud.any():
        return np.empty((0, 3)), np.empty((0, 3))

    rotation_matrix = Rotation.random().as_matrix()

    # Project the pointcloud onto the chosen vector
    projected_points = point_cloud @ rotation_matrix

    ratio = ( (1 + overlap_ratio) / 2)
    # Find the split point based on the overlap ratio
    split_point_top = np.percentile(projected_points[:,0], 100 * ratio)
    split_point_bot = np.percentile(projected_points[:,0], 100 * (1 - ratio))

    # Filter the pointcloud based on the split point
    pcl1_indices = np.where(projected_points[:,0] <= split_point_top)[0]
    pcl2_indices = np.where(projected_points[:,0] > split_point_bot)[0]
    pcl1 = point_cloud[pcl1_indices]
    pcl2 = point_cloud[pcl2_indices]
    overlapping_indices = np.where(np.logical_and((projected_points[:,0] <= split_point_top),projected_points[:,0] > split_point_bot))[0]

    return pcl1, pcl2, pcl1_indices, pcl2_indices, overlapping_indices


def load_data_and_compute_means(base_dir):
    means = []
    subdir_names = []

    # Traverse the base directory
    for root, dirs, files in os.walk(base_dir):
        if "combined_dict.pkl" in files:
            # Load the dictionary
            with open(os.path.join(root, "combined_dict.pkl"), 'rb') as f:
                data_dict = pickle.load(f)

                # Compute the mean values of 'err_r_deg' and 'err_t'
                err_r_deg_mean = np.mean(data_dict['err_r_deg'])
                err_t_mean = np.mean(data_dict['err_t'])

                means.append((err_r_deg_mean, err_t_mean))
                subdir_names.append(os.path.basename(root))

    return means, subdir_names


def plot_means(means, subdir_names):
    # Unpack the means for plotting
    err_r_deg_means, err_t_means = zip(*means)

    combined = list(zip(err_r_deg_means, err_t_means, subdir_names))

    # Sort by err_r_deg_mean
    combined.sort(key=lambda x: x[0])

    # Select the top 10% with the lowest err_r_deg_mean
    top_10_percent = combined[:max(1, len(combined) // 5)]

    # Unpack the filtered means and names
    err_r_deg_means, err_t_means, subdir_names = zip(*top_10_percent)

    # Create a 2D plot
    plt.figure(figsize=(12, 8))

    for err_r_deg_mean, err_t_mean, subdir_name in zip(err_r_deg_means, err_t_means, subdir_names):
        plt.scatter(err_r_deg_mean, err_t_mean, label=subdir_name)

    plt.xlabel('Mean err_r_deg')
    plt.ylabel('Mean err_t')
    plt.title('Mean err_r_deg vs Mean err_t for each subdirectory')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()
def check_pairings_modelnet():
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1')
    # nn_modes = [2, 3, 4]
    nn_modes = [3]
    # tri_modes = [True, False]
    tri_modes = [True]
    for nn_mode in nn_modes:
        for tri in tri_modes:
            run_name = f'mode_{nn_mode}_tri_{tri}'
            print(run_name)

            worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict = (
                test_multi_scale_using_embedding_predator_modelnet(cls_args=cls_args, num_worst_losses=3,
                                                          scaling_factor="min",
                                                          amount_of_interest_points=500,
                                                          num_of_ransac_iter=20000,
                                                          pct_of_points_2_take=1,
                                                          max_non_unique_correspondences=3,
                                                          nn_mode=nn_mode, scales=4,
                                                          receptive_field=[1, 5, 10, 15],
                                                          amount_of_samples=50, tri=tri))
            os.makedirs(run_name, exist_ok=True)
            file_path = os.path.join(run_name, 'combined_dict.pkl')
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(combined_dict, pickle_file)
            npy_file_path = os.path.join(run_name, 'losses_rot.npy')
            np.save(npy_file_path, losses_rot)
            plot_metrics(combined_dict, dir=run_name)
            plot_losses(losses=losses_rot, inliers=final_inliers_list, filename=f'rot_loss_scales_emb.png',
                        dir=run_name)
            plotWorst(worst_losses=worst_losses, dir=run_name)
def check_pairings_3dmatch():
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1')
    # nn_modes = [2, 3, 4]
    nn_modes = [3]
    # tri_modes = [True, False]
    tri_modes = [True]
    for nn_mode in nn_modes:
        for tri in tri_modes:
            run_name = f'mode_{nn_mode}_tri_{tri}'
            print(run_name)

            worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict = (
                test_multi_scale_using_embedding_predator_3dmatch(cls_args=cls_args, num_worst_losses=3,
                                                          scaling_factor="min",
                                                          amount_of_interest_points=500,
                                                          num_of_ransac_iter=20000,
                                                          pct_of_points_2_take=1,
                                                          max_non_unique_correspondences=3,
                                                          nn_mode=nn_mode, scales=4,
                                                          receptive_field=[1, 5, 10, 15],
                                                          amount_of_samples=50, tri=tri))
            os.makedirs(run_name, exist_ok=True)
            file_path = os.path.join(run_name, 'combined_dict.pkl')
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(combined_dict, pickle_file)
            npy_file_path = os.path.join(run_name, 'losses_rot.npy')
            np.save(npy_file_path, losses_rot)
            plot_metrics(combined_dict, dir=run_name)
            plot_losses(losses=losses_rot, inliers=final_inliers_list, filename=f'rot_loss_scales_emb.png',
                        dir=run_name)
            plotWorst(worst_losses=worst_losses, dir=run_name)
def check_registration_modelnet():
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1')
    scaling_factors = ["min"]
    subsamples = [500]
    receptive_fields_list = [[1, 5, 10, 15]]
    scales_list = [4]
    nn_modes = [2]
    pcts = [1]
    runsac_iterations = [20000]
    for scales, receptive_field in zip(scales_list, receptive_fields_list):
        for amount_of_interest_points in subsamples:
            for scaling_factor in scaling_factors:
                for pct_of_points_2_take in pcts:
                    for nn_mode in nn_modes:
                        for num_of_ransac_iter in runsac_iterations:
                            rfield = "_".join(map(str, receptive_field))
                            run_name = f'rfield_{rfield}_keypoints_{amount_of_interest_points}_pct_{pct_of_points_2_take}_mode_{nn_mode}_rsac_iter_{num_of_ransac_iter}_{scaling_factor}'
                            print(run_name)

                            # cProfile.runctx('test_multi_scale_using_embedding_predator_modelnet(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor, amount_of_interest_points=amount_of_interest_points, num_of_ransac_iter=num_of_ransac_iter, pct_of_points_2_take=pct_of_points_2_take, max_non_unique_correspondences=max_non_unique_correspondences, scales=scales, receptive_field=receptive_field,  amount_of_samples=20, batch_size=16 )', globals(), locals())

                            # profiler = cProfile.Profile()
                            # profiler.runctx('test_multi_scale_using_embedding_predator_modelnet(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor, amount_of_interest_points=amount_of_interest_points,num_of_ransac_iter=num_of_ransac_iter, pct_of_points_2_take=pct_of_points_2_take, max_non_unique_correspondences=max_non_unique_correspondences,scales=scales, receptive_field=receptive_field,  amount_of_samples=10)', globals(), locals())
                            # stats = pstats.Stats(profiler)
                            # stats.sort_stats(pstats.SortKey.TIME)
                            # stats.print_stats()

                            worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict = (
                                test_multi_scale_using_embedding_predator_modelnet(cls_args=cls_args,
                                                                                   num_worst_losses=3,
                                                                                   scaling_factor=scaling_factor,
                                                                                   amount_of_interest_points=amount_of_interest_points,
                                                                                   num_of_ransac_iter=num_of_ransac_iter,
                                                                                   pct_of_points_2_take=pct_of_points_2_take,
                                                                                   max_non_unique_correspondences=3,
                                                                                   nn_mode=nn_mode, scales=scales,
                                                                                   receptive_field=receptive_field,
                                                                                   amount_of_samples=50))
                            os.makedirs(run_name, exist_ok=True)
                            file_path = os.path.join(run_name, 'combined_dict.pkl')
                            with open(file_path, 'wb') as pickle_file:
                                pickle.dump(combined_dict, pickle_file)
                            npy_file_path = os.path.join(run_name, 'losses_rot.npy')
                            np.save(npy_file_path, losses_rot)
                            plot_metrics(combined_dict, dir=run_name)
                            plot_losses(losses=losses_rot, inliers=final_inliers_list,
                                        filename=f'rot_loss_scales_emb.png', dir=run_name)
                            plotWorst(worst_losses=worst_losses, dir=run_name)

def check_registration_3dmatch():
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1')
    scaling_factors = ["min", "mean"]
    subsamples = [500, 1000, 1500]
    receptive_fields_list = [ [1, 5] ,[1, 10] , [1, 5, 10, 15]]
    scales_list = [2, 2, 4]
    nn_modes = [2,3,4,5]
    pcts = [1, 0.75]
    runsac_iterations = [1000]

    for scales, receptive_field in zip(scales_list, receptive_fields_list):
        for amount_of_interest_points in subsamples:
            for scaling_factor in scaling_factors:
                for pct_of_points_2_take in pcts:
                    for nn_mode in nn_modes:
                        for num_of_ransac_iter in runsac_iterations:
                            rfield = "_".join(map(str, receptive_field))
                            run_name = f'rfield_{rfield}_keypoints_{amount_of_interest_points}_pct_{pct_of_points_2_take}_mode_{nn_mode}_rsac_iter_{num_of_ransac_iter}_{scaling_factor}'
                            print(run_name)

                            worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict = (
                                test_multi_scale_using_embedding_predator_3dmatch(cls_args=cls_args,
                                                                                   num_worst_losses=3,
                                                                                   scaling_factor=scaling_factor,
                                                                                   amount_of_interest_points=amount_of_interest_points,
                                                                                   num_of_ransac_iter=num_of_ransac_iter,
                                                                                   pct_of_points_2_take=pct_of_points_2_take,
                                                                                   max_non_unique_correspondences=3,
                                                                                   nn_mode=nn_mode, scales=scales,
                                                                                   receptive_field=receptive_field,
                                                                                   amount_of_samples=50))
                            os.makedirs(run_name, exist_ok=True)
                            file_path = os.path.join(run_name, 'combined_dict.pkl')
                            with open(file_path, 'wb') as pickle_file:
                                pickle.dump(combined_dict, pickle_file)
                            npy_file_path = os.path.join(run_name, 'losses_rot.npy')
                            np.save(npy_file_path, losses_rot)
                            plot_metrics(combined_dict, dir=run_name)
                            plot_losses(losses=losses_rot, inliers=final_inliers_list,
                                        filename=f'rot_loss_scales_emb.png', dir=run_name)
                            plotWorst(worst_losses=worst_losses, dir=run_name)


if __name__ == '__main__':
    check_registration_3dmatch()
