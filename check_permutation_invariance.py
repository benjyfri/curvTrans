import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
# from plotting_functions import *

def createLap(point_cloud, normalized, graph_weight_mode):
    distances = cdist(point_cloud, point_cloud)
    lower_triangle_indices = np.tril_indices(distances.shape[0], k=-1)
    lower_triangle = distances[lower_triangle_indices]
    if graph_weight_mode == 0:
        weights = np.exp(-distances)
    elif graph_weight_mode == 1:
        weights = np.exp(-distances ** 2)
    elif graph_weight_mode == 2:
        rbf_weight = (distances ** 2) / np.max(lower_triangle)
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 3:
        rbf_weight = (distances ** 2) / np.min(lower_triangle)
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 4:
        rbf_weight = (distances ** 2) / np.mean(lower_triangle)
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 5:
        rbf_weight = (distances ** 2) / np.median(lower_triangle)
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 6:
        weights = distances
    elif graph_weight_mode == 7:
        weights = (distances ** 2)
    elif graph_weight_mode == 8:
        rbf_weight = (distances ** 2) / 5
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 9:
        rbf_weight = (distances ** 2) / 10
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 10:
        rbf_weight = (distances ** 2) / 15
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 11:
        rbf_weight = (distances ** 2) / 20
        weights = np.exp(-rbf_weight)
    elif graph_weight_mode == 12:
        weights = np.exp(-distances / 5)
    elif graph_weight_mode == 13:
        weights = np.exp(-distances / 10)
    elif graph_weight_mode == 14:
        weights = np.exp(-distances / 15)
    elif graph_weight_mode == 15:
        weights = np.exp(-distances / 20)

    column_sums = weights.sum(axis=1)
    # print(column_sums)
    diag_matrix = np.diag(column_sums)
    laplacian = diag_matrix - weights

    if normalized:
        inv_D_sqrt = np.diag(np.sqrt(1.0 / (column_sums + 1e-7)))
        identity = np.eye(weights.shape[1])
        laplacian = identity - (inv_D_sqrt @ weights @ inv_D_sqrt)

    return laplacian


def top_k_smallest_eigenvectors(graph, k=0):
    if k < 1:
        k = 1
    eigenvalues, eigenvectors = eigh(graph)
    return eigenvectors[:, 1:k + 1], eigenvalues


def sort_by_first_eigenvector(eigenvectors):
    abs_eigenvector = np.abs(eigenvectors[:, 0])
    sorted_indices = np.argsort(abs_eigenvector[1:])
    sorted_indices = np.concatenate(([0], 1 + sorted_indices))

    sorted_eigvecs = eigenvectors[sorted_indices]
    return sorted_indices, sorted_eigvecs


import matplotlib.pyplot as plt


def plot_same_index_list(all_runs_same_index_list):
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('tab10', len(all_runs_same_index_list))

    idx = 0
    for run_name, same_index_list in all_runs_same_index_list.items():
        # Line plot for each run
        # plt.plot(same_index_list, label=run_name+f'; {np.mean(same_index_list):.2f}')
        plt.scatter(np.arange(len(same_index_list)), same_index_list, color=colors(idx), alpha=0.8, label=run_name+f'; {np.mean(same_index_list):.2f}')
        idx +=1

    plt.title("Same Index Amount Distribution for Different Runs")
    plt.xlabel("Index")
    plt.ylabel("Same Index Amount")
    plt.legend()
    plt.grid(True)
    plt.savefig("same_index_distribution.png")
    plt.show()
def check(graph_weight_mode=0):
    hdf5_file = h5py.File("train_surfaces_40_stronger_boundaries.h5" , 'r')
    point_clouds_group = hdf5_file['point_clouds']
    num_point_clouds = len(point_clouds_group)
    normalize_lap = [True, False]
    noise_size = [0, 0.001, 0.01, 0.1, 1]
    # noise_size = [0]
    all_runs_same_index_list = {}
    for nlap in normalize_lap:
        for std in noise_size:
            run_name = "nlap: " + str(nlap) + "_std_" + str(std) +f'_mode:{graph_weight_mode}'
            print(f'Run:  {run_name}')
            same_index_list = []
            for i in range(num_point_clouds):
                # if i> 30:
                #     break
                # if i%100 == 0:
                #     print(f'pcl: {i}')
                point_cloud_name = f"point_cloud_{i}"
                # Load metadata from attributes
                info = {key: point_clouds_group[point_cloud_name].attrs[key] for key in
                        point_clouds_group[point_cloud_name].attrs}
                point_cloud = np.array(point_clouds_group[point_cloud_name])

                l = createLap(point_cloud, nlap, graph_weight_mode=graph_weight_mode)
                # Compute LPE embedding
                eigvecs, eigenvals = top_k_smallest_eigenvectors(l)
                indices_orig, fixed_eigs = sort_by_first_eigenvector(eigvecs)

                pcl_size = len(point_cloud)
                rot = R.random().as_matrix()
                # print(rot)
                point_cloud1 = np.matmul(point_cloud, rot.T)
                # point_cloud1 = point_cloud
                noise = np.random.normal(0, std, point_cloud.shape)
                # noise = np.random.normal(0, 0, point_cloud.shape)
                noisy_point_cloud = point_cloud1 + noise
                permuted_indices = np.concatenate(([0], (1 + np.random.permutation(pcl_size-1))))
                # permuted_indices = np.arange(41)
                noisy_point_cloud = noisy_point_cloud[permuted_indices]


                l2 = createLap(noisy_point_cloud, nlap, graph_weight_mode=graph_weight_mode)
                # Compute LPE embedding
                eigvecs2, eigenvals2 = top_k_smallest_eigenvectors(l2)
                indices_noised, fixed_eigs2 = sort_by_first_eigenvector(eigvecs2)

                base = np.arange(pcl_size)

                original_reordered_indices = base[indices_orig]
                noised_reordered_indices = (base[permuted_indices])[indices_noised]
                same_order = np.array_equal(original_reordered_indices, noised_reordered_indices)
                same_index_amount = np.count_nonzero(original_reordered_indices == noised_reordered_indices)
                same_index_list.append(same_index_amount)

            all_runs_same_index_list[run_name] = same_index_list
            print(f'mean: {np.mean(same_index_list)}')
    # plot_same_index_list(all_runs_same_index_list)

if __name__ == '__main__':
    check(graph_weight_mode=0)
    check(graph_weight_mode=1)
    check(graph_weight_mode=2)
    check(graph_weight_mode=3)
    check(graph_weight_mode=4)
    check(graph_weight_mode=5)
    check(graph_weight_mode=6)
    check(graph_weight_mode=7)
    check(graph_weight_mode=8)
    check(graph_weight_mode=9)
    check(graph_weight_mode=10)
    check(graph_weight_mode=11)
    check(graph_weight_mode=12)
    check(graph_weight_mode=13)
    check(graph_weight_mode=14)
    check(graph_weight_mode=15)

