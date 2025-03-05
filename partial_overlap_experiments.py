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
from data import samplePcl


def checkSizeSynthetic():
    hdf5_file = h5py.File("train_surfaces_05X05.h5" , 'r')
    point_clouds_group = hdf5_file['point_clouds']
    num_point_clouds = len(point_clouds_group)
    indices = list(range(num_point_clouds))
    total_sum_median = 0
    total_sum_mean = 0
    total_diam_med = 0
    total_max_dist_from_center = 0
    yay_max = 0
    counter_list = [0,0,0,0,0]
    yay = []
    angles_ang =[]
    angles_pyr =[]
    angles_cur =[]
    for idx in range(num_point_clouds):
        if idx%10000 ==0:
            print(f'------------{idx}------------')
        point_cloud_name = f"point_cloud_{indices[idx]}"

        info = {key: point_clouds_group[point_cloud_name].attrs[key] for key in
                point_clouds_group[point_cloud_name].attrs}
        # if (info['angle']>0):
        #     ang = info['angle']
        #     angles_cur.append(max(abs(info['k1']),abs(info['k2'])))
        #     if info['edge']==1:
        #         angles_pyr.append(ang)
        #     if info['edge']==2:
        #         angles_ang.append(ang)
        #
        # yay.append(max(abs(info['k1']),abs(info['k2'])))
        # # print(f"{max(abs(info['k1']),abs(info['k2']))},")
        # continue
        class_label = info['class']
        counter_list[class_label] = counter_list[class_label] + 1
        class_label = info['class']
        angle = info['angle']
        radius = info['radius']
        [min_len, max_len] = [0.45, 0.55]
        bias = 0.25
        edge_label = info['edge']
        bounds, point_cloud = samplePcl(angle=angle, radius=radius, class_label=class_label, sampled_points=20, min_len=min_len,
                                max_len=max_len, bias=bias, info=info, edge_label=edge_label)

        distances = cdist(point_cloud, point_cloud)

        max_ax = np.max(np.abs(point_cloud))
        if yay_max < max_ax:
            yay_max = max_ax
        # Replace the diagonal with infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Find the minimum distance to the closest point for each point
        closest_distances = np.min(distances, axis=1)
        median_closest_distance = np.median(closest_distances)
        mean_closest_distance = np.mean(closest_distances)
        total_sum_median += np.mean(median_closest_distance)
        total_sum_mean += np.mean(mean_closest_distance)
        total_diam_med += np.median(np.max(np.abs(point_cloud),axis=0))
        total_max_dist_from_center += np.max(np.linalg.norm(point_cloud,axis=1))
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'SYNTHETIC')
    print(f'++++++++++++++++++++++++++++++++++')
    print(counter_list)
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'max value in any axis {yay_max}')
    print(f'MEDIAN point distance synthetic: {total_sum_median / num_point_clouds}')
    print(f'MEAN point distance synthetic: {total_sum_mean / num_point_clouds}')
    print(f'mean of MEDIAN diameter synthetic: {total_diam_med / num_point_clouds}')
    print(f'total_max_dist_from_center: {total_max_dist_from_center}, num_point_clouds: {num_point_clouds}')
    print(f'MEAN max distance from center: {total_max_dist_from_center / num_point_clouds}')
def checkDiameterPCLSynthetic():
    hdf5_file = h5py.File("train_surfaces_05X05.h5" , 'r')
    point_clouds_group = hdf5_file['point_clouds']
    num_point_clouds = len(point_clouds_group)
    for label in [0,1,2,3,4]:
        all_diameter_vals = []
        all_k1_vals = []
        all_k2_vals = []
        for idx in range(num_point_clouds // 5):
            # if idx%10000 ==0:
            #     print(f'------------{idx}------------')
            full_idx = idx + (label * (num_point_clouds // 5))
            point_cloud_name = f"point_cloud_{full_idx}"

            info = {key: point_clouds_group[point_cloud_name].attrs[key] for key in
                    point_clouds_group[point_cloud_name].attrs}
            class_label = info['class']
            angle = info['angle']
            radius = info['radius']
            all_k1_vals.append(info['k1'])
            all_k2_vals.append(info['k2'])
            [min_len, max_len] = [0.45, 0.55]
            bias = 0.25
            edge_label = info['edge']
            bounds, point_cloud = samplePcl(angle=angle, radius=radius,class_label=class_label,sampled_points=20,min_len=min_len,max_len=max_len, bias=bias, info=info, edge_label=edge_label)
            cur_diameter = (np.max(np.linalg.norm(point_cloud,axis=1)))
            all_diameter_vals.append(cur_diameter)
        all_diameter_vals = np.array(all_diameter_vals)
        print(f'++++++++++++++++++++++++++++++++++')
        print(f'label: {label}')
        print(f'++++++++++++++++++++++++++++++++++')
        print(f'Mean: {np.mean(all_diameter_vals)}')
        print(f'Median: {np.median(all_diameter_vals)}')
        print(f'Max: {np.max(all_diameter_vals)}')
        print(f'Min: {np.min(all_diameter_vals)}')
        print(f'Min K1: {np.min(np.abs(all_k1_vals))}')
        print(f'max K1: {np.max(np.abs(all_k1_vals))}')
        print(f'Min K2: {np.min(np.abs(all_k2_vals))}')
        print(f'max K2: {np.max(np.abs(all_k2_vals))}')

def checkSizeModelnet():
    test_dataset = test_predator_data()
    total_sum = 0
    total_sum_median = 0
    total_sum_mean = 0
    total_diam_med = 0
    total_max_dist_from_center = 0
    total_vol_mean = 0
    yay_max = 0
    vol_list = []
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
        cur_vol_sum=0
        cur_sum_max_dist_from_center=0
        cur_diam_med = np.median(np.max(np.abs(pcl),axis=0))
        total_diam_med += cur_diam_med
        for i in range(num_of_points):
            point_cloud = pcl[:,i,:]
            axis_size = (np.abs(np.max(point_cloud, axis=0) -np.min(point_cloud, axis=0)))
            volume = axis_size[0]*axis_size[1]*axis_size[2]
            vol_list.append(volume)
            cur_vol_sum +=volume
            distances = cdist(point_cloud, point_cloud)

            max_ax = np.max(np.abs(point_cloud))
            if yay_max < max_ax:
                yay_max = max_ax

            # Replace the diagonal with infinity to ignore self-distances
            np.fill_diagonal(distances, np.inf)

            # Find the minimum distance to the closest point for each point
            closest_distances = np.min(distances, axis=1)
            median_closest_distance = np.median(closest_distances)
            mean_closest_distance = np.mean(closest_distances)
            cur_sum_median += median_closest_distance
            cur_sum_mean += mean_closest_distance
            cur_sum_max_dist_from_center += np.max(np.linalg.norm(point_cloud, axis=1))
        total_sum_median += cur_sum_median/num_of_points
        total_sum_mean += cur_sum_mean/num_of_points
        total_max_dist_from_center += cur_sum_max_dist_from_center/num_of_points
        total_vol_mean += cur_vol_sum/num_of_points
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'MODELNET')
    print(f'++++++++++++++++++++++++++++++++++')
    print(f'max value in any axis {yay_max}')
    print(f'MEDIAN point distance modelnet: {total_sum_median / size}')
    print(f'MEAN point distance modelnet: {total_sum_mean / size}')
    print(f'mean of MEDIAN diameter modelnet: {total_diam_med / size}')
    print(f'total_max_dist_from_center: {total_max_dist_from_center}, size: {size}')
    print(f'MEAN max distance from center: {total_max_dist_from_center / size}')
    print(f'MEAN volume of each local patch: {total_vol_mean / size}')
    vol_list = np.array(vol_list)
    print(f'vol mean {np.mean(vol_list)}')
    print(f'vol median {np.median(vol_list)}')
    print(f'vol max {np.max(vol_list)}')
    print(f'vol min {np.min(vol_list)}')

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
        # root = r'C:\\Users\\benjy\\Desktop\\curvTrans\\DeepBBS\\modelnet40_ply_hdf5_2048'
        root = r'C:\\Users\\Owner\\PycharmProjects\\curvTrans\\bbsWithShapes\\data\\modelnet40_ply_hdf5_2048'
    else:
        root = r'/content/curvTrans/DeepBBS/modelnet40_ply_hdf5_2048'
    test_dataset = ModelNetHdf(overlap_radius=overlap_radius, root=root,
                                subset='train', categories=None, transform=train_transforms, matching=matching)
    return test_dataset

def load_data(partition='test', divide_data=1):
    DATA_DIR = r'C:\\Users\\Owner\\PycharmProjects\\curvTrans\\bbsWithShapes\\data'
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
    cls_args_shape.num_mlp_layers = 5
    cls_args_shape.num_neurons_per_layer = 64
    cls_args_shape.sampled_points = 20
    cls_args_shape.use_second_deg = 1
    cls_args_shape.lpe_normalize = 0
    cls_args_shape.exp = name
    cls_args_shape.lpe_dim = 0
    cls_args_shape.output_dim = 5
    cls_args_shape.use_lap_reorder = 1
    cls_args_shape.lap_eigenvalues_dim = 0
    return cls_args_shape, 1, 1

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
def check_registration_modelnet(model_name):
    remove_planes = [False, True]
    remove_dups = [False, True]
    subsamples = [700,350]
    receptive_fields_list = [[1, 3], [1, 3, 5], [1, 3, 5, 7], [1, 7], [1, 5, 7], [1, 5, 9]]
    scales_list = [2,3,4,2,3,3]
    pcts = [1]
    runsac_iterations = [1000]
    use_triangles=[False,True]
    models_names = [model_name]
    nn_modes = [2,4]
    for nn_mode in nn_modes:
        for scales, receptive_field in zip(scales_list, receptive_fields_list):
            for amount_of_interest_points in subsamples:
                for avoid_planes in remove_planes:
                    for avoid_diff_classification in remove_dups:
                        for tri in use_triangles:
                            for model_name in models_names:
                                for num_of_ransac_iter in runsac_iterations:
                                    rfield = "_".join(map(str, receptive_field))
                                    run_name = f'rfield_{rfield}_keypoints_{amount_of_interest_points}_tri_{tri}_nn_mode_{nn_mode}_{avoid_planes}_{avoid_diff_classification}_{model_name}'
                                    print(run_name)

                                    # cProfile.runctx('test_multi_scale_using_embedding_predator_modelnet(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor, amount_of_interest_points=amount_of_interest_points, num_of_ransac_iter=num_of_ransac_iter, pct_of_points_2_take=pct_of_points_2_take, max_non_unique_correspondences=max_non_unique_correspondences, scales=scales, receptive_field=receptive_field,  amount_of_samples=20, batch_size=16 )', globals(), locals())

                                    # profiler = cProfile.Profile()
                                    # profiler.runctx('test_multi_scale_using_embedding_predator_modelnet(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor, amount_of_interest_points=amount_of_interest_points,num_of_ransac_iter=num_of_ransac_iter, pct_of_points_2_take=pct_of_points_2_take, max_non_unique_correspondences=max_non_unique_correspondences,scales=scales, receptive_field=receptive_field,  amount_of_samples=10)', globals(), locals())
                                    # stats = pstats.Stats(profiler)
                                    # stats.sort_stats(pstats.SortKey.TIME)
                                    # stats.print_stats()
                                    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name=model_name)
                                    cls_args.num_neurons_per_layer = 64
                                    cls_args.num_mlp_layers = 5
                                    cls_args.output_dim = 4
                                    worst_losses, losses_rot, losses_trans, final_thresh_list, final_inliers_list, point_distance_list, iter_2_ransac_convergence, combined_dict = (
                                        test_multi_scale_using_embedding_predator_modelnet_geo(cls_args=cls_args,
                                                                                           tri=tri,
                                                                                           num_worst_losses=3,
                                                                                           scaling_factor="1",
                                                                                           amount_of_interest_points=amount_of_interest_points,
                                                                                           num_of_ransac_iter=num_of_ransac_iter,
                                                                                           pct_of_points_2_take=1,
                                                                                           max_non_unique_correspondences=3,
                                                                                           nn_mode=nn_mode, scales=scales,
                                                                                           receptive_field=receptive_field,
                                                                                           amount_of_samples=50,
                                                                                           avoid_planes=avoid_planes, avoid_diff_classification=avoid_diff_classification))
                                    dir_path = os.path.join("0101run", run_name)
                                    os.makedirs(dir_path, exist_ok=True)
                                    file_path = os.path.join(dir_path, 'combined_dict.pkl')
                                    with open(file_path, 'wb') as pickle_file:
                                        pickle.dump(combined_dict, pickle_file)
                                    # npy_file_path = os.path.join(dir_path, 'losses_rot.npy')
                                    # np.save(npy_file_path, losses_rot)
                                    plot_metrics(combined_dict, dir=dir_path)
                                    mean = np.mean(losses_rot)
                                    median = np.median(losses_rot)
                                    file_name = f"rot_loss_mean_{mean:.2f}_median_{median:.2f}.npy"

                                    np.save(os.path.join(dir_path, file_name), losses_rot)
                                    plot_losses(losses=losses_rot, inliers=final_inliers_list,
                                                filename=f'rot_loss_scales_emb.png', dir=dir_path)
                                    plotWorst(worst_losses=worst_losses, dir=dir_path)

def check_registration_3dmatch(model_name):
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name=model_name)
    cls_args.output_dim = 5
    cls_args.num_neurons_per_layer = 64

    # scaling_factors = ["min", "mean"]
    scaling_factors = [1]
    # subsamples = [500, 1000, 1500, 2000, 3000, 5000]
    subsamples = [1000, 3000]

    # receptive_fields_list = [ [1, 5] ,[1, 10] , [1, 5, 10, 15]]
    # receptive_fields_list = [[1,5],[1,5,10],[1,10,20,30], [1,10,20,30,40,50], [1, 50, 100, 150, 200], [1,25, 50,75, 100,125, 150, 200]]

    # receptive_fields_list = [[1, 3], [1, 3, 5], [1, 3, 5, 7], [1, 7], [1, 5, 7], [1, 5, 9]]
    receptive_fields_list = [ [1,10,20,30,40]]
    receptive_fields_list = [ [1,3,5]]
    scales_list = [3]
    # scales_list = [2,3,4,6,5,8]
    # scales_list = [2,3,4,2,3,3]
    # nn_modes = [2,3,4]
    nn_modes = [4]
    pcts = [0.5]
    thresh_multi_options = [1,3,5,10]
    thresh_multi_options = [5]
    # tri_type =[True, False]
    tri_type =[True]
    # ransac_type =[True, False]
    ransac_type =[False]
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
                                                                                           num_of_ransac_iter=10000,
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
    a_3MLP32_eig15_cntr04_harder_receptive_field = [5,6,7,9]
    for scaling_factor in ["axis", "min"]:
        # for model_name in ["3MLP32_eig15_cntr05_std007", "3MLP32_eig15_cntr04_harder.pt"]:
        for model_name in ["3MLP32_eig15_cntr04_harder"]:
            print()
            print()
            print(f'---------------------------------------------')
            print(f'{model_name} usin scaling: {scaling_factor}')
            print(f'---------------------------------------------')
            cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name=model_name)

            cls_args.output_dim = 5
            cls_args.num_neurons_per_layer = 32
            cls_args.sampled_points = 20
            cls_args.lap_eigenvalues_dim = 15
            import itertools
            receptive_field = [list(combo) for r in range(2, 6) for combo in itertools.combinations(range(2, 10), r)]
            # receptive_field = [[1],[1, 2, 3], [1, 3, 5]]
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


                    emb_1 , scaling_fac = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1,
                                           args_shape=cls_args, scaling_factor=scaling_factor)
                    emb_2 , scaling_fac = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2,
                                           args_shape=cls_args, scaling_factor=scaling_factor)

                    emb_1 = emb_1.detach().cpu().numpy().squeeze()
                    emb_2 = emb_2.detach().cpu().numpy().squeeze()
                    # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, pair)

                    if len(pyr_layers)>1:
                        # for scale in pyr_layers[1:]:
                        for scale in pyr_layers:
                            subsampled_1 = farthest_point_sampling_o3d(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1) // scale))
                            subsampled_2 = farthest_point_sampling_o3d(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2) // scale))

                            global_emb_1 , scaling_fac = classifyPoints(model_name=cls_args.exp,
                                                          pcl_src=subsampled_1,
                                                          pcl_interest=noisy_pointcloud_1, args_shape=cls_args,
                                                          scaling_factor=scaling_factor)

                            global_emb_2 , scaling_fac = classifyPoints(model_name=cls_args.exp,
                                                          pcl_src=subsampled_2,
                                                          pcl_interest=noisy_pointcloud_2, args_shape=cls_args,
                                                          scaling_factor=scaling_factor)

                            global_emb_1 = global_emb_1.detach().cpu().numpy().squeeze()
                            global_emb_2 = global_emb_2.detach().cpu().numpy().squeeze()
                            # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, global_emb_1, global_emb_2,pair)
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
import os
import re

def find_lowest_mean_image_dir(base_path):
    lowest_mean = float('inf')
    lowest_mean_dir = None

    # Regular expression to extract mean value from the file name
    pattern = re.compile(r"err_r_deg_Mean_(\d+\.\d+)_Median_")

    # Iterate through each subdirectory
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            # Search for the PNG file in the directory
            for file in os.listdir(subdir_path):
                if file.endswith('.png'):
                    match = pattern.search(file)
                    if match:
                        mean_value = float(match.group(1))
                        if mean_value < lowest_mean:
                            lowest_mean = mean_value
                            lowest_mean_dir = subdir

    # Print the directory with the lowest mean and the mean value
    if lowest_mean_dir is not None:
        print(f"The directory with the lowest mean is: {lowest_mean_dir}")
        print(f"The lowest mean value is: {lowest_mean}")
    else:
        print("No valid image files were found.")


def testPretrainedModel(args, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = BasicPointCloudDataset(file_path='train_surfaces_05X05.h5', args=args)

    model = shapeClassifier(args)
    model.load_state_dict(torch.load(f'models_weights/{model_name}.pt'))
    model.to(device)
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')
    # Set the model to evaluation mode
    model.eval()
    count = 0
    total_acc_loss = 0.0
    label_correct = {label: 0 for label in range(5)}
    label_total = {label: 0 for label in range(5)}
    wrong_preds = {label: [] for label in range(5)}
    wrong_idx = {label: [] for label in range(5)}
    wrong_pcl = {label: [] for label in range(5)}
    wrong_pred_class = {label: [] for label in range(5)}
    wrong_K1_values = {label: [] for label in range(5)}
    wrong_K2_values = {label: [] for label in range(5)}

    with torch.no_grad():
        for batch in test_dataloader:
            pcl, info = batch['point_cloud'].to(device), batch['info']
            label = info['class'].to(device).long()
            output = model((pcl.permute(0, 2, 1)).unsqueeze(2))
            output = (output[:, :5]).squeeze()
            preds = output.max(dim=1)[1]
            total_acc_loss += torch.mean((preds == label).float()).item()

            # Collect data for wrong predictions
            for i, (pred, actual_label) in enumerate(zip(preds, label.cpu().numpy())):
                if pred != actual_label:
                    wrong_preds[actual_label].append(pred.item())
                    wrong_idx[actual_label].append(info['idx'][i].item())
                    wrong_pcl[actual_label].append(pcl[i, :, :])
                    wrong_pred_class[actual_label].append(preds[i])
                    wrong_K1_values[actual_label].append((info['k1'][i].item()))
                    wrong_K2_values[actual_label].append((info['k2'][i].item()))

            count += 1

            # Update per-label statistics
            for label_name in range(5):
                correct_mask = (preds == label_name) & (label == label_name)
                label_correct[label_name] += correct_mask.sum().item()
                label_total[label_name] += (label == label_name).sum().item()

    label_accuracies = {
        label: label_correct[label] / label_total[label]
        for label in range(5)
        if label_total[label] != 0
    }
    for label, accuracy in label_accuracies.items():
        print(f"Accuracy for label {label}: {accuracy:.4f}")

    # for label in range(4):
    for label in range(5):
        if len(wrong_preds[label]) > 0:
            print(f"Label {label}:")
            print(f"  - Most frequent wrong prediction: {max(wrong_preds[label], key=wrong_preds[label].count)}")
            print(f"  - Average K1 for wrong predictions: {np.mean(wrong_K1_values[label])}")
            print(f"  - Average K for wrong predictions: {np.mean(wrong_K2_values[label])}")
            print(f"  - median K1 for wrong predictions: {np.median(wrong_K1_values[label])}")
            print(f"  - median K for wrong predictions: {np.median(wrong_K2_values[label])}")
            print(f"+++++")
            argmax_K1_index = (np.argmax(np.abs(wrong_K1_values[label])))
            print(f"  - biggest abs wrong K1 pcl idx: {wrong_idx[label][argmax_K1_index]}")
            print(f"  - biggest abs wrong K1 pcl val: {wrong_K1_values[label][argmax_K1_index]}")
            np.save(f"{label}_max_K1_pcl_{wrong_pred_class[label][argmax_K1_index]}_{wrong_idx[label][argmax_K1_index]}.npy",
                    (wrong_pcl[label][argmax_K1_index]).cpu().numpy())

            argmin_K1_index = (np.argmin(np.abs(wrong_K1_values[label])))
            print(f"  - smallest abs wrong K1 pcl idx: {wrong_idx[label][argmin_K1_index]}")
            print(f"  - smallest abs wrong K1 pcl val: {wrong_K1_values[label][argmin_K1_index]}")
            np.save(f"{label}_min_K1_pcl_{wrong_pred_class[label][argmin_K1_index]}_{wrong_idx[label][argmin_K1_index]}.npy",
                    (wrong_pcl[label][argmin_K1_index]).cpu().numpy())

            argmax_K2_index = (np.argmax(np.abs(wrong_K2_values[label])))
            print(f"  - biggest abs wrong K2 pcl idx: {wrong_idx[label][argmax_K2_index]}")
            print(f"  - biggest abs wrong K2 pcl val: {wrong_K2_values[label][argmax_K2_index]}")
            np.save(f"{label}_max_K2_pcl_{wrong_pred_class[label][argmax_K2_index]}_{wrong_idx[label][argmax_K2_index]}.npy",
                    (wrong_pcl[label][argmax_K2_index]).cpu().numpy())

            argmin_K2_index = (np.argmin(np.abs(wrong_K2_values[label])))
            print(f"  - smallest abs wrong K2 pcl idx: {wrong_idx[label][argmin_K2_index]}")
            print(f"  - smallest abs wrong K2 pcl val: {wrong_K2_values[label][argmin_K2_index]}")
            np.save(f"{label}_min_K2_pcl_{wrong_pred_class[label][argmin_K2_index]}_{wrong_idx[label][argmin_K2_index]}.npy",
                    (wrong_pcl[label][argmin_K2_index]).cpu().numpy())

def find_top_three_directories_with_lowest_means(base_dir, k=3):
    # Pattern to match the required filename format
    filename_pattern = re.compile(r"rot_loss_mean_(?P<mean>[-+]?[0-9]*\.?[0-9]+)_median_[-+]?[0-9]*\.?[0-9]+\.npy")

    # Initialize a list to store directories and their mean values
    directories_with_means = []

    # Iterate over all items in the base directory
    for item in os.listdir(base_dir):
        # Construct the full path
        item_path = os.path.join(base_dir, item)

        # Check if the item is a directory and starts with 'rfield'
        if os.path.isdir(item_path) and item.startswith("rfield"):
            # Iterate over the files in the directory
            for file in os.listdir(item_path):
                # Check if the file matches the required pattern
                match = filename_pattern.match(file)
                if match:
                    # Extract the mean value from the filename
                    mean = float(match.group("mean"))

                    # Append the directory and mean value to the list
                    directories_with_means.append((item_path, mean))

    # Sort the list by mean value
    directories_with_means.sort(key=lambda x: x[1])

    # Return the top three directories and their means
    return directories_with_means[:k]

if __name__ == '__main__':
    # base_directory = r"C:\Users\benjy\Desktop\curvTrans\0101run"
    # base_directory = r"C:\Users\Owner\PycharmProjects\curvTrans\0101run"
    # result = find_top_three_directories_with_lowest_means(base_directory, k=10)
    # for r in result:
    #     print(r)
    # exit(0)
    # model_name = "a_cntr01_std005_64"
    model_name = "ZZ_cntr0_std01_long"
    # model_name = "ZZ_cntr0_std01_long_no_edges"

    # viewStabilityWithPartial()
    # checkSizeModelnet()
    # checkSizeSynthetic()
    # exit(0)

    # example_usage()
    # exit(0)
    # checkSizeSynthetic()
    # # checkSyntheticData()
    # checkDiameterPCLSynthetic()
    # exit(0)
    # check_registration_modelnet(model_name)
    # # # check_registration_3dmatch(model_name)
    # exit(0)


    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name=model_name)
    cls_args.output_dim=5
    # cls_args.output_dim=10
    cls_args.num_neurons_per_layer = 64
    cls_args.num_mlp_layers = 5

    # checkDiameterPCLSynthetic()
    # testPretrainedModel(cls_args, model_name)
    # exit()
    #
    # import time
    # start_time = time.time()
    # check3dStability(cls_args=cls_args, scaling_factor="1",scales=3, receptive_field=[1, 5, 10])
    # end_time = time.time()
    #
    # # Calculate and print elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Function execution time: {elapsed_time:.4f} seconds")
    # view_stabiity(cls_args=cls_args, scaling_factor=1,scales=3, receptive_field=[1, 5, 10], add_noise=True)
    # exit(0)


    # visualizeShapesWithEmbeddings3dMatchCorners(model_name=model_name, args_shape=cls_args, scaling_factor=2.5, rgb=False)
    # # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor="one", rgb=False, add_noise=False)
    # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor=0.9, rgb=False, add_noise=False)
    # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor="0.5", rgb=False, add_noise=True)
    # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor="1", rgb=False, add_noise=True)
    # vis2(model_name=model_name, args_shape=cls_args,scaling_factor="1", rgb=False, add_noise=True)
    # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor="1.5", rgb=False, add_noise=False)
    # claude_plotting(model_name=model_name, args_shape=cls_args,scaling_factor="1", rgb=False, add_noise=False)
    claude_plotting_static(model_name=model_name, args_shape=cls_args,scaling_factor="1")
    # # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor=1.2, rgb=False, add_noise=False)
    # # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor=0.77, rgb=False, add_noise=True)
    # # visualizeShapesWithEmbeddingsCorners(model_name=model_name, args_shape=cls_args,scaling_factor=0.8, rgb=False, add_noise=True)
    # # visualizeShapesWithEmbeddings(model_name=model_name, args_shape=cls_args, scaling_factor=0.9, rgb=True, add_noise=True)
    # visualizeShapesWithEmbeddings(model_name=model_name, args_shape=cls_args, scaling_factor="1", rgb=True, add_noise=True)
    # # visualizeShapesWithEmbeddings(model_name=model_name, args_shape=cls_args, scaling_factor=1.1, rgb=True, add_noise=True)
    # # exit(0)
    #
