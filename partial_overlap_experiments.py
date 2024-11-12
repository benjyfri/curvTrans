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
from scipy.spatial.distance import cdist
from benchmark_modelnet import dcm2euler

import numpy as np
from scipy.spatial import KDTree


def find_possible_correspondences_volume(query_point, pcl1, max_angle_deg=45, max_translation=0.5):
    """
    Find all possible corresponding points in pcl1 for a query point from pcl2
    by computing the possible volume where the point could be located.

    Parameters:
    -----------
    query_point : np.ndarray
        3D point from pcl2 to find correspondences for
    pcl1 : np.ndarray
        First point cloud (Nx3 array)
    max_angle_deg : float
        Maximum rotation angle in degrees per axis
    max_translation : float
        Maximum translation distance per axis

    Returns:
    --------
    np.ndarray
        Boolean mask indicating which points in pcl1 could correspond to query_point
    """
    query_point = np.asarray(query_point)

    # Calculate maximum displacement due to rotation
    # For small angles (in radians), sin(θ) ≈ θ
    # Maximum displacement = r * θ, where r is distance from origin
    max_angle_rad = np.deg2rad(max_angle_deg)
    point_distance = np.linalg.norm(query_point)

    # The maximum displacement due to rotation is approximately:
    # distance_from_origin * sin(max_angle)
    max_rotation_displacement = point_distance * np.sin(max_angle_rad)

    # For 45 degrees or less, we can compute the maximum possible displacement
    # This creates a sphere around the point's position
    # The radius of this sphere is the max rotation displacement

    # Total maximum displacement is the sum of:
    # 1. Translation in any direction (forms a cube)
    # 2. Rotation displacement (forms a sphere)
    total_max_radius = max_rotation_displacement + max_translation

    # Create a KD-tree for efficient nearest neighbor search
    tree = KDTree(pcl1)

    # Find all points within the maximum possible radius
    indices = tree.query_ball_point(query_point, total_max_radius)

    # Create mask for the entire point cloud
    mask = np.zeros(len(pcl1), dtype=bool)
    mask[indices] = True

    return mask


def visualize_volume(query_point, max_angle_deg=45, max_translation=0.5, num_points=1000):
    """
    Generate points to visualize the possible volume where a point could be located.
    This is useful for debugging and visualization purposes.

    Returns points on the boundary and within the possible volume.
    """
    # Generate random points in a sphere
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0, 1, num_points)

    # Convert to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack((x, y, z))

    # Scale points based on maximum displacement
    point_distance = np.linalg.norm(query_point)
    max_angle_rad = np.deg2rad(max_angle_deg)
    max_rotation_displacement = point_distance * np.sin(max_angle_rad)
    total_max_radius = max_rotation_displacement + max_translation * np.sqrt(3)

    points = points * total_max_radius + query_point

    return points
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
    # center_point_idx = np.argsort(np.linalg.norm(sampled_points, axis=1))[-1]
    sampled_points_with_centroid = sampled_points_with_centroid - sampled_points_with_centroid[center_point_idx, :]
    sampled_points_with_centroid[center_point_idx, :] = (sampled_points_with_centroid[0, :]).copy()
    sampled_points_with_centroid[0, :] = np.array([[0, 0, 0]])

    return sampled_points_with_centroid

def checkSizeSynthetic():
    hdf5_file = h5py.File("train_surfaces_with_corners_very_mild_curve.h5" , 'r')
    point_clouds_group = hdf5_file['point_clouds']
    num_point_clouds = len(point_clouds_group)
    indices = list(range(num_point_clouds))
    total_sum_median = 0
    total_sum_mean = 0
    for idx in range(num_point_clouds):
        if idx%1000 ==0:
            print(f'------------{idx}------------')
        point_cloud_name = f"point_cloud_{indices[idx]}"

        info = {key: point_clouds_group[point_cloud_name].attrs[key] for key in
                point_clouds_group[point_cloud_name].attrs}
        # point_cloud = point_clouds_group[point_cloud_name]
        # point_cloud_orig = np.array(point_cloud, dtype=np.float32)
        if info['class'] <= 3:
            point_cloud = samplePoints(info['a'], info['b'], info['c'], info['d'], info['e'],
                                       count=20)
        else:
            point_cloud = sampleHalfSpacePoints(info['a'], info['b'], info['c'], info['d'], info['e'],
                                                count=20)
        distances = cdist(point_cloud, point_cloud)

        # Replace the diagonal with infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Find the minimum distance to the closest point for each point
        closest_distances = np.min(distances, axis=1)
        median_closest_distance = np.median(closest_distances)
        mean_closest_distance = np.mean(closest_distances)
        total_sum_median += np.mean(median_closest_distance)
        total_sum_mean += np.mean(mean_closest_distance)
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'SYNTHETIC')
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'AVG point distance synthetic: {total_sum_median / num_point_clouds}')
    print(f'AVG point distance synthetic: {total_sum_mean / num_point_clouds}')
def checkSizeModelnet():
    test_dataset = test_predator_data()
    total_sum = 0
    total_sum_median = 0
    total_sum_mean = 0
    size = len(test_dataset)
    for i in range(size):
        if i%1000 ==0:
            print(f'------------{i}------------')
        data = test_dataset.__getitem__(i)
        src_pcd, tgt_pcd, GT_rot, GT_trans, sample = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data['sample']
        pcl = (((get_k_nearest_neighbors_diff_pcls(src_pcd, src_pcd, k=21)).squeeze()).T)

        num_of_points = pcl.shape[1]
        cur_sum_median=0
        cur_sum_mean=0
        for i in range(num_of_points):
            point_cloud = pcl[:,i,:]

            distances = cdist(point_cloud, point_cloud)

            # Replace the diagonal with infinity to ignore self-distances
            np.fill_diagonal(distances, np.inf)

            # Find the minimum distance to the closest point for each point
            closest_distances = np.min(distances, axis=1)
            median_closest_distance = np.median(closest_distances)
            mean_closest_distance = np.mean(closest_distances)
            cur_sum_median += median_closest_distance
            cur_sum_mean += mean_closest_distance
        total_sum_median += cur_sum_median/num_of_points
        total_sum_mean += cur_sum_mean/num_of_points
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'MODELNET')
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'AVG point distance synthetic: {total_sum_median / size}')
    print(f'AVG point distance synthetic: {total_sum_mean / size}')
def example_usage():
    # Create sample point clouds
    test_dataset = test_predator_data()
    data = test_dataset.__getitem__(83)
    src_pcd, tgt_pcd, GT_rot, GT_trans, sample = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data[
        'sample']

    query_point = tgt_pcd[0,:]
    # query_point = np.array([0, 0, 0])

    # Find possible correspondences
    mask = find_possible_correspondences_volume(query_point, src_pcd)
    corresponding_points = src_pcd[mask]

    # Generate visualization points
    volume_points = visualize_volume(query_point)

    return corresponding_points, volume_points

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

def load_data_and_compute_means_3dmatch(base_dir):
    means = []
    subdir_names = []

    # Traverse the base directory
    for root, dirs, files in os.walk(base_dir):
        if "losses_rot.npy" in files:
            losses_rot = np.load(os.path.join(root, "losses_rot.npy"))
            means.append(np.mean(losses_rot))
            subdir_names.append(os.path.basename(root))
    combined = list(zip(means, subdir_names))
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
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='small_very_mild_3MLP64N2deg_lpe0eig16')
    cls_args.output_dim = 8
    cls_args.num_neurons_per_layer = 64
    cls_args.sampled_points = 20
    cls_args.lap_eigenvalues_dim = 16
    scaling_factors = ["min"]
    subsamples = [700,500]
    receptive_fields_list = [[1, 3], [1, 3, 5], [1, 3, 5, 7], [1, 7], [1, 5, 7], [1, 5, 9]]
    receptive_fields_list = [[1, 5, 9]]
    receptive_fields_list = [[1]]
    scales_list = [1]
    nn_modes = [2,3,4]
    # nn_modes = [2]
    pcts = [1]
    runsac_iterations = [500]
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
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='small_very_mild_3MLP64N2deg_lpe0eig16')
    cls_args.output_dim = 8
    cls_args.num_neurons_per_layer = 64
    cls_args.sampled_points = 20
    cls_args.lap_eigenvalues_dim = 16

    # scaling_factors = ["min", "mean"]
    scaling_factors = ["min"]
    # subsamples = [500, 1000, 1500, 2000, 3000, 5000]
    subsamples = [1000, 3000]
    # receptive_fields_list = [ [1, 5] ,[1, 10] , [1, 5, 10, 15]]
    # receptive_fields_list = [[1,5],[1,5,10],[1,10,20,30], [1,10,20,30,40,50], [1, 50, 100, 150, 200], [1,25, 50,75, 100,125, 150, 200]]

    # receptive_fields_list = [[1, 3], [1, 3, 5], [1, 3, 5, 7], [1, 7], [1, 5, 7], [1, 5, 9]]
    receptive_fields_list = [ [1, 3, 5],[1, 2, 3]]
    scales_list = [3,3]
    # scales_list = [2,3,4,6,5,8]
    # scales_list = [2,3,4,2,3,3]
    # nn_modes = [2,3,4]
    nn_modes = [2]
    pcts = [1]
    thresh_multi_options = [1,3,5]
    # tri_type =[True, False]
    tri_type =[False]
    # ransac_type =[True, False]
    ransac_type =[True]
    # ransac_type =[True]
    count = 0
    for use_o3d_ransac in ransac_type:
        for scales, receptive_field in zip(scales_list, receptive_fields_list):
            for amount_of_interest_points in subsamples:
                for scaling_factor in scaling_factors:
                    for pct_of_points_2_take in pcts:
                        for nn_mode in nn_modes:
                            for tri in tri_type:
                                for thresh_multi in thresh_multi_options:
                                    if use_o3d_ransac==True and tri==True:
                                        break
                                    # if count<90:
                                    #     count += 1
                                    #     print(count)
                                    #     continue
                                    rfield = "_".join(map(str, receptive_field))
                                    run_name = f'cls_3dmatch_rfield_{rfield}_keypoints_{amount_of_interest_points}_thresh_multi_{thresh_multi}_mode_{nn_mode}_o3drsac_{use_o3d_ransac}_{scaling_factor}_tri_{tri}'
                                    print(run_name)

                                    worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, good_correspondences = (
                                        test_multi_scale_using_embedding_predator_3dmatch(cls_args=cls_args,
                                                                                           num_worst_losses=3,
                                                                                           scaling_factor=scaling_factor,
                                                                                           amount_of_interest_points=amount_of_interest_points,
                                                                                           num_of_ransac_iter=1000,
                                                                                           use_o3d_ransac=use_o3d_ransac,
                                                                                           pct_of_points_2_take=pct_of_points_2_take,
                                                                                           max_non_unique_correspondences=3,
                                                                                           nn_mode=nn_mode, scales=scales,
                                                                                           receptive_field=receptive_field,
                                                                                           thresh_multi=thresh_multi,
                                                                                           tri=tri,
                                                                                           amount_of_samples=50))
                                    os.makedirs(run_name, exist_ok=True)
                                    file_path = os.path.join(run_name, 'combined_dict.pkl')
                                    npy_file_path = os.path.join(run_name, 'losses_rot.npy')
                                    np.save(npy_file_path, losses_rot)
                                    plot_losses(losses=losses_rot, inliers=good_correspondences,
                                                filename=f'rot_loss_scales_emb.png', dir=run_name)
                                    plotWorst(worst_losses=worst_losses, dir=run_name)
                                    for (loss, worst_loss_variables) in worst_losses:
                                        noisy_pointcloud_1 = worst_loss_variables['noisy_pointcloud_1']
                                        save_receptive_field(noisy_pointcloud_1, noisy_pointcloud_1[0], rfield=receptive_field, filename=f"rfield_{loss}.html",
                                                             dir=run_name)
                                    # exit(0)
def viewStabilityWithPartial():
    for scaling_factor in ["min"]:
        print()
        print()
        print(f'---------------------------------------------')
        # print(f'{scaling_factor}')
        print(f'---------------------------------------------')
        # cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='cur_3MLP32_eig15_lpe0_3')
        cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32_eig15_cntr1')

        cls_args.output_dim = 5
        cls_args.num_neurons_per_layer = 32
        cls_args.sampled_points = 20
        cls_args.lap_eigenvalues_dim = 15
        import itertools
        receptive_field = [list(combo) for r in range(3, 6) for combo in itertools.combinations(range(2, 10), r)]
        receptive_field = [[1,2,3],[1,3,5]]
        train_dataset = test_predator_data(matching=True)
        for pyr_layers in receptive_field:
            count = 0
            # for i in [65,83,94,95,97]:
            size = len(train_dataset)
            size = 100
            for i in range(size):
                data = train_dataset.__getitem__(i)
                src_pcd, tgt_pcd, GT_rot, GT_trans, sample = data['src_pcd'], data['tgt_pcd'], data['rot'], data['trans'], data[
                    'sample']
                pair = data['matching_inds'][0].numpy()
                #
                # view_stabiity(cls_args=cls_args, scaling_factor="min", scales=3, receptive_field=[1, 3, 7],
                #               given_pcls=[src_pcd, tgt_pcd, torch.tensor(pair)])

                noisy_pointcloud_1 = src_pcd
                noisy_pointcloud_2 = tgt_pcd


                emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1,
                                       args_shape=cls_args, scaling_factor=scaling_factor)
                emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2,
                                       args_shape=cls_args, scaling_factor=scaling_factor)

                emb_1 = emb_1.detach().cpu().numpy().squeeze()
                emb_2 = emb_2.detach().cpu().numpy().squeeze()
                plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, pair)

                if len(pyr_layers)>1:
                    # for scale in pyr_layers[1:]:
                    for scale in pyr_layers:
                        subsampled_1 = farthest_point_sampling_o3d(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1) // scale))
                        subsampled_2 = farthest_point_sampling_o3d(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2) // scale))

                        global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                                      pcl_src=subsampled_1,
                                                      pcl_interest=noisy_pointcloud_1, args_shape=cls_args,
                                                      scaling_factor=scaling_factor)

                        global_emb_2 = classifyPoints(model_name=cls_args.exp,
                                                      pcl_src=subsampled_2,
                                                      pcl_interest=noisy_pointcloud_2, args_shape=cls_args,
                                                      scaling_factor=scaling_factor)

                        global_emb_1 = global_emb_1.detach().cpu().numpy().squeeze()
                        global_emb_2 = global_emb_2.detach().cpu().numpy().squeeze()
                        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, global_emb_1, global_emb_2,pair)
                        emb_1 = np.hstack((emb_1, global_emb_1))
                        emb_2 = np.hstack((emb_2, global_emb_2))

                # Calculate distances from the random embedding to all embeddings in embedding2
                distances = np.linalg.norm(emb_2 - emb_1[pair[0]], axis=1)

                # Find indices of the 20 closest points
                closest_indices = np.argsort(distances)[:10]
                if pair[1] in closest_indices:
                    count += 1
            print(f'{pyr_layers}')
            print(f'Count: {count} out of {size} shapes are in the top 10')
if __name__ == '__main__':
    # viewStabilityWithPartial()
    checkSizeModelnet()
    checkSizeSynthetic()
    exit(0)

    # example_usage()
    # exit(0)
    checkSizeSynthetic()
    checkSyntheticData()
    exit(0)
    # check_registration_modelnet()
    # check_registration_3dmatch()
    # exit(0)

    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32_eig15_cntr01')
    cls_args.output_dim=5
    cls_args.num_neurons_per_layer = 64
    cls_args.num_neurons_per_layer = 32
    cls_args.sampled_points = 20
    cls_args.lap_eigenvalues_dim = 15
    checking_1 = [0.5]

    # view_stabiity(cls_args=cls_args, scaling_factor="min",scales=3, receptive_field=[1, 5, 9])
    # view_stabiity(cls_args=cls_args, scaling_factor="min",scales=3, receptive_field=[1, 2, 3])
    # view_stabiity(cls_args=cls_args, scaling_factor="min",scales=2, receptive_field=[2, 5, 6, 7, 8])
    # exit(0)
    for i in checking_1:
        # visualizeShapesWithEmbeddings3dMatchCorners(model_name='3MLP32_eig15_cntr01', args_shape=cls_args,
        # scaling_factor="min", rgb=False) visualizeShapesWithEmbeddings3dMatchCorners(
        # model_name='3MLP32_eig15_cntr01', args_shape=cls_args, scaling_factor="min", rgb=False)
        # visualizeShapesWithEmbeddingsCorners(model_name='3MLP32_eig15_cntr01', args_shape=cls_args,scaling_factor="min", rgb=False)
        visualizeShapesWithEmbeddings(model_name='3MLP32_eig15_cntr01', args_shape=cls_args, scaling_factor="max", rgb=True)
        visualizeShapesWithEmbeddings(model_name='3MLP32_eig15_cntr01', args_shape=cls_args, scaling_factor="median", rgb=True)
    exit(0)


    receptive_fields_list = [[1, 3], [1, 3, 5], [1, 3, 5, 7], [1, 7], [1, 5, 7],[1, 5, 9]]
    # receptive_fields_list = [[1, 3, 5, 7]]
    scales_list = [2, 3, 4, 2, 3,3]
    # scales_list = [4]
    for scales, receptive_field in zip(scales_list, receptive_fields_list):
        test_pairings_modelnet(cls_args=cls_args,num_worst_losses=3,scaling_factor="min",amount_of_interest_points=500,
                              num_of_ransac_iter=1000,use_o3d_ransac=False,pct_of_points_2_take=1,max_non_unique_correspondences=3,nn_mode=2, scales=scales,receptive_field=receptive_field,thresh_multi=5,tri=True,amount_of_samples=1)