import cProfile
import pstats

import torch
from plotting_functions import *
from ransac import *
from experiments_utils import *
import numpy as np
from functools import partial
from modelnet import ModelNetHdf
import transforms
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
if __name__ == '__main__':
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1')
    # cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_std035')
    # a = checkPred()
    worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence = (
        test_multi_scale_using_embedding_predator(cls_args=cls_args, num_worst_losses=3, scaling_factor=1,
                                           num_of_ransac_iter=50, pct_of_points_2_take=1, scales=5, receptive_field=[1,5, 10]))
    plot_losses(losses=losses, inliers=num_of_inliers, filename=f'{scaling_factor}_{max_non_unique_correspondences}_{pct_of_points_2_take}_{amount_of_interest_points}_loss_{scales}_scales_emb.png', dir="junk")
    plotWorst(worst_losses=worst_losses, model_name=f'{scaling_factor}_{amount_of_interest_points}_{scales}_scales_emb')

    exit(0)
    # cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_std001')
    # scaling_factors = [15,20,25]
    # scaling_factors = ["90pct"]
    scaling_factors = ["max" , "mean", "median", "min", "d_90"]
    # scaling_factors = ["min", "median"]
    # scaling_factors = ["median"]
    # scaling_factors = [0.5, 1, 2, 3, 4, 8]
    # scaling_factors = ["median"]
    # subsamples = [20,50,100,200,300]
    # subsamples = [200,300]
    # subsamples = [50, 150, 300]
    subsamples = [300]
    receptive_fields_list = [ [1, 20, 40] , [1, 10, 20, 30, 40]]
    # receptive_fields_list = [[1, 10, 20, 30, 40]]
    # receptive_fields_list = [[1, 20, 40] ]
    # receptive_fields_list = [[1, 20, 30] ]
    scales_list = [3,5]
    # scales_list = [5]
    # scales_list = [3]
    # bbuddies = [1, 3 ,5]
    bbuddies = [3]
    # pcts = [0.5, 0.75]
    pcts = [1]
    max_non_unique_correspondences=3
    for scales, receptive_field in zip(scales_list, receptive_fields_list):
        for amount_of_interest_points in subsamples:
            for scaling_factor in scaling_factors:
                for pct_of_points_2_take in pcts:
                    for max_non_unique_correspondences in bbuddies:
                        print(f'Scaling factor: {scaling_factor}, amount_of_interest_points: {amount_of_interest_points}, #scales: {scales}')

                        worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
                            = test_multi_scale_using_embedding(cls_args=cls_args,num_worst_losses = 3, scaling_factor=scaling_factor, scales=scales, receptive_field=receptive_field, amount_of_interest_points=amount_of_interest_points,
                                                    num_of_ransac_iter=50, shapes=range(100), pct_of_points_2_take=1, max_non_unique_correspondences=max_non_unique_correspondences)
                        plot_losses(losses=losses, inliers=num_of_inliers, filename=f'{scaling_factor}_{amount_of_interest_points}_loss_{scales}_scales_emb.png', dir="junk_001")

                        # worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
                        #     = test_multi_scale_using_embedding(cls_args=cls_args,num_worst_losses = 3, scaling_factor=scaling_factor, scales=scales, receptive_field=receptive_field, amount_of_interest_points=50,
                        #                             num_of_ransac_iter=500, shapes=range(100),create_pcls_func=partial(split_pointcloud_overlap, overlap_ratio=0.3), max_non_unique_correspondences=max_non_unique_correspondences, pct_of_points_2_take=pct_of_points_2_take)
                        #
                        # plot_losses(losses=losses, inliers=num_of_inliers, filename=f'{scaling_factor}_{max_non_unique_correspondences}_{pct_of_points_2_take}_{amount_of_interest_points}_loss_{scales}_scales_emb.png', dir="junk")
                        # plotWorst(worst_losses=worst_losses, model_name=f'{scaling_factor}_{amount_of_interest_points}_{scales}_scales_emb')
                        # visualizeShapesWithEmbeddings(model_name='3MLP32N2deg_lpe0eig36_1', args_shape=cls_args,
                        #                               scaling_factor=scaling_factor, rgb=False)
                        # view_stabiity(cls_args=cls_args,num_worst_losses = 3, scaling_factor=scaling_factor, scales=5, receptive_field=[1, 10, 20, 30], amount_of_interest_points=300,
                        #                             num_of_ransac_iter=50, plot_graphs=1,create_pcls_func=partial(split_pointcloud_overlap, overlap_ratio=0.3))
                        # exit(0)