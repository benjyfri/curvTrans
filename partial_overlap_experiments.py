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
from ransac import *
from experiments_utils import *
import numpy as np
from functools import partial
from experiment_shapes import visualizeShapesWithEmbeddings
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

def check_overlap():
    pcls, label = load_data()
    shapes = [86, 162, 174, 176, 179]
    # shapes = np.arange(100)
    overlap_avg = 0
    overlap_max = 0.33
    overlap_min = 0.28
    divider = 1.53
    while (overlap_avg < overlap_min) or (overlap_avg > overlap_max):

        sum = 0
        for k in shapes:
            overlap = 0.3
            pointcloud = pcls[k][:]
            # pcl1, pcl2, overlap = get_two_point_clouds_with_neighbors(pointcloud, (int)(len(pointcloud)//divider))
            pcl1, pcl2 = split_point_cloud(pointcloud, overlap)
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
    k = (int)(len(point_cloud) // k)
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
    return random_point_cloud, farthest_point_cloud


def split_point_cloud(point_cloud, overlap_ratio):
    """
    Splits a 3D point cloud into two parts based on a random axis and given overlap ratio.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (2048, 3).
        overlap_ratio (float): Ratio of overlap between the two point clouds.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two point clouds split based on the overlap ratio.
    """
    # Choose a random axis
    axis = np.random.choice(3)

    # Sort points based on the chosen axis
    sorted_indices = np.argsort(point_cloud[:, axis])
    sorted_points = point_cloud[sorted_indices]

    # Calculate the number of points for overlap
    total_points = point_cloud.shape[0]
    overlap_points = int(total_points * overlap_ratio)
    split_points = (total_points - overlap_points) // 2

    # Find the values of a and b
    a = sorted_points[split_points, axis]
    b = sorted_points[split_points + overlap_points, axis]

    # Create the two point clouds
    first_point_cloud = point_cloud[point_cloud[:, axis] <= b]
    second_point_cloud = point_cloud[point_cloud[:, axis] >= a]

    return first_point_cloud, second_point_cloud
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
if __name__ == '__main__':
    cls_args, _, _ = create_3MLP32N2deg_lpe0eig36_args(name='3MLP32N2deg_lpe0eig36_1_4')
    scaling_factor = 17
    scaling_factor = 15
    # check_overlap()
    # visualizeShapesWithEmbeddings(model_name='3MLP32N2deg_lpe0eig36', args_shape=cls_args, scaling_factor=scaling_factor)
    shapes = range(20,25)
    shapes = [1,5,10,15,20,25,30,35]
    shapes = [1,5,10]
    shapes = [15,20,25]
    shapes = [20]
    for scaling_factor in shapes:
        # worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
        #     = test_multi_scale_classification(cls_args=cls_args,num_worst_losses = 3, scaling_factor=scaling_factor, scales=2, receptive_field=[1, 4], amount_of_interest_points=100,
        #                             interest_point_choice=2, num_of_ransac_iter=50, shapes=[86, 179], plot_graphs=1,create_pcls_func=partial(split_point_cloud, overlap_ratio=0.3))

        # worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
        #     = test_multi_scale_classification(cls_args=cls_args,num_worst_losses = 3, scaling_factor=scaling_factor, scales=1, receptive_field=[1, 4], amount_of_interest_points=1000,
        #                             interest_point_choice=2, num_of_ransac_iter=50, plot_graphs=1)
        # visualizeShapesWithEmbeddings(model_name='3MLP32N2deg_lpe0eig36_1_4', args_shape=cls_args,
        #                               scaling_factor=scaling_factor)
        view_stabiity(model_name='3MLP32N2deg_lpe0eig36_1_4', args_shape=cls_args,
                                      scaling_factor=scaling_factor)
        # view_stabiity(model_name=None, args_shape=None, scaling_factor=None)