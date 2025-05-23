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
from data import samplePcl

def createLap(point_cloud, normalized, graph_weight_mode):
    distances = torch.cdist(point_cloud, point_cloud)
    if graph_weight_mode == 0:
        weights = torch.exp(-distances)
    elif graph_weight_mode == -1:
        batch_size = point_cloud.shape[0]
        rbf_weight = distances / (torch.max(distances.view(batch_size, -1), dim=1).values).view(batch_size, 1, 1)
        weights = torch.exp(-rbf_weight)
    else:
        rbf_weight = (-distances ** 2) / graph_weight_mode
        weights = torch.exp(rbf_weight)
    column_sums = weights.sum(dim=1)
    diag_matrix = torch.diag_embed(column_sums)
    laplacian = diag_matrix - weights
    if normalized:
        inv_sqrt_col_sum = torch.rsqrt(column_sums + 1e-7)  # Safer reciprocal sqrt
        laplacian = (
                torch.eye(weights.size(1), device=weights.device)  # Identity matrix
                - (inv_sqrt_col_sum.unsqueeze(2) * weights * inv_sqrt_col_sum.unsqueeze(1))
        )
    return laplacian

def top_k_smallest_eigenvectors(graph, k=0):
    if k < 1:
        k = 1
    eigenvalues, eigenvectors = torch.linalg.eigh(graph)
    return eigenvectors[:, :, 1:k + 1], eigenvalues


def sort_by_first_eigenvector(eigenvectors):
    first_eig_vec = eigenvectors[:, :, 0]

    abs_tensor = torch.abs(first_eig_vec)  # Absolute values
    max_indices = torch.argmax(abs_tensor, dim=1)  # Indices of max abs value

    # Step 2: Get the sign of the max absolute values
    max_values = first_eig_vec[torch.arange(eigenvectors.shape[0]), max_indices]  # Gather max values
    signs = torch.sign(max_values)  # Compute the sign of the max values

    # Step 3: Multiply each row by its corresponding sign
    result = first_eig_vec * signs.unsqueeze(1)

    sorted_indices = torch.argsort(result[:, 1:])
    zeros_tensor = torch.zeros(size=(eigenvectors.shape[0], 1), device=eigenvectors.device, dtype=int)
    sorted_indices = torch.cat((zeros_tensor, 1 + sorted_indices), dim=1)

    sorted_eigvecs = torch.gather(eigenvectors, 1,
                                  sorted_indices.unsqueeze(-1).expand(-1, -1, eigenvectors.size(-1)))
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
    hdf5_file = h5py.File("train_surfaces_05X05.h5", 'r')
    point_clouds_group = hdf5_file['point_clouds']
    num_point_clouds = len(point_clouds_group)
    indices = list(range(num_point_clouds))
    normalize_lap = [True, False]
    noise_size = [0.0,0.01,0.03,0.05,0.07, 0.1]
    all_runs_same_index_list = {}
    for std in noise_size:
        print()
        for nlap in normalize_lap:
            run_name = "nlap: " + str(nlap) + "_std_" + str(std)+ f'_mode:{graph_weight_mode}'
            print(f'Run:  {run_name}')
            same_index_list = []
            for idx in range(num_point_clouds//10):
                # if (idx*50) % 10000 == 0:
                #     print(f'------------{idx*50}------------')
                point_cloud_name = f"point_cloud_{indices[idx*10]}"

                info = {key: point_clouds_group[point_cloud_name].attrs[key] for key in
                        point_clouds_group[point_cloud_name].attrs}
                class_label = info['class']
                angle = info['angle']
                radius = info['radius']
                [min_len, max_len] = [0.45, 0.55]
                bias = 0.25
                edge_label = info['edge']
                bounds, point_cloud = samplePcl(angle=angle, radius=radius, class_label=class_label, sampled_points=20,
                                                min_len=min_len,
                                                max_len=max_len, bias=bias, info=info, edge_label=edge_label)


                l = createLap(torch.from_numpy(point_cloud).float().unsqueeze(0), nlap, graph_weight_mode=graph_weight_mode)
                # Compute LPE embedding
                eigvecs, eigenvals = top_k_smallest_eigenvectors(l)
                indices_orig, fixed_eigs = sort_by_first_eigenvector(eigvecs)
                indices_orig = indices_orig.squeeze().cpu().numpy()
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


                l2 = createLap(torch.from_numpy(noisy_point_cloud).float().unsqueeze(0), nlap, graph_weight_mode=graph_weight_mode)
                # Compute LPE embedding
                eigvecs2, eigenvals2 = top_k_smallest_eigenvectors(l2)
                indices_noised, fixed_eigs2 = sort_by_first_eigenvector(eigvecs2)
                indices_noised = indices_noised.squeeze().cpu().numpy()

                base = np.arange(pcl_size)

                original_reordered_indices = base[indices_orig]
                noised_reordered_indices = (base[permuted_indices])[indices_noised]
                same_order = np.array_equal(original_reordered_indices, noised_reordered_indices)
                same_index_amount = np.count_nonzero(original_reordered_indices == noised_reordered_indices)
                same_index_list.append(same_index_amount)

            all_runs_same_index_list[run_name] = same_index_list
            print(f'mean: {np.mean(same_index_list)}')
    # plot_same_index_list(all_runs_same_index_list)


import matplotlib.pyplot as plt


def plot_runs_old():
    x_axis = [0, 0.01, 0.03, 0.05, 0.07, 0.1]

    runs = {
        "mode_0_5": ([21, 17.47, 13.23, 10.69, 9.07, 7.3], [21, 17.43, 13.18, 10.47, 8.68, 6.97]),
        "mode_1": ([21, 17.23, 13.06, 10.64, 8.9, 7.19], [21, 17.43, 13.31, 10.72, 8.94, 7.26]),
        "mode_2": ([21, 17.2, 12.89, 10.47, 8.87, 7.03], [21, 17.53, 13.27, 10.71, 9.12, 7.3]),
        "mode_5": ([21, 17.21, 12.97, 10.43, 8.67, 7.03], [21, 17.47, 13.19, 10.78, 9.12, 7.29])
    }

    colors = ["blue", "green", "red", "purple"]  # Assign unique colors

    plt.figure(figsize=(8, 6))

    for (label, (unnormalized, normalized)), color in zip(runs.items(), colors):
        plt.plot(x_axis, normalized, label=f"{label} (normalized)", color=color, linestyle="-")
        plt.plot(x_axis, unnormalized, label=f"{label} (unnormalized)", color=color, linestyle="--")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Comparison of Normalized and Unnormalized Runs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_runs():
    x_axis = [0, 0.01, 0.03, 0.05, 0.07, 0.1]

    runs = {
        "mode_0_5": ([21, 17.47, 13.23, 10.69, 9.07, 7.3], [21, 17.43, 13.18, 10.47, 8.68, 6.97]),
        "mode_1": ([21, 17.23, 13.06, 10.64, 8.9, 7.19], [21, 17.43, 13.31, 10.72, 8.94, 7.26]),
        "mode_2": ([21, 17.2, 12.89, 10.47, 8.87, 7.03], [21, 17.53, 13.27, 10.71, 9.12, 7.3]),
        "mode_5": ([21, 17.21, 12.97, 10.43, 8.67, 7.03], [21, 17.47, 13.19, 10.78, 9.12, 7.29])
    }

    colors = ["blue", "green", "red", "purple"]  # Assign unique colors

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, ((label, (unnormalized, normalized)), color) in zip(axes, zip(runs.items(), colors)):
        ax.plot(x_axis, normalized, label=f"{label} (normalized)", color=color, linestyle="-", marker='o')
        ax.plot(x_axis, unnormalized, label=f"{label} (unnormalized)", color=color, linestyle="--", marker='s')
        ax.set_title(label)
        ax.legend()
        ax.grid(True)

    fig.suptitle("Comparison of Normalized and Unnormalized Runs")
    fig.supxlabel("X-axis")
    fig.supylabel("Y-axis")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # check(graph_weight_mode=0.5)
    # check(graph_weight_mode=1)
    # check(graph_weight_mode=2)
    # check(graph_weight_mode=5)
    # Call the function to plot the runs
    plot_runs()

