from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import MLP, TransformerNetwork, TransformerNetworkPCT, shapeClassifier
from data import BasicPointCloudDataset
from scipy.spatial.transform import Rotation
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import os
import urllib.request
import zipfile
import shutil
import faiss
import plotly.graph_objects as go
from train import configArgsPCT
import numpy as np
from scipy.spatial import KDTree
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from scipy.linalg import eig
import matplotlib.pyplot as plt
def rotate_to_z_axis(arr):
    # Compute covariance matrix
    covariance_matrix = np.cov(arr, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Find the index of the smallest eigenvalue
    min_eigenvalue_index = np.argmin(eigenvalues)

    # Extract the eigenvector corresponding to the smallest eigenvalue
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]

    # Normalize the eigenvector
    min_eigenvector /= np.linalg.norm(min_eigenvector)

    # Compute rotation matrix
    z_axis = np.array([0, 0, 1])
    rotation_matrix = np.cross(min_eigenvector, z_axis)
    angle = np.arccos(np.dot(min_eigenvector, z_axis))
    rotation_matrix = np.array([[0, -rotation_matrix[2], rotation_matrix[1]],
                                 [rotation_matrix[2], 0, -rotation_matrix[0]],
                                 [-rotation_matrix[1], rotation_matrix[0], 0]])

    # Rotate the array
    rotated_arr = np.dot(arr, np.eye(3) + np.sin(angle) * rotation_matrix + (1 - np.cos(angle)) * np.dot(rotation_matrix, rotation_matrix))

    return rotated_arr
def plot_4_point_clouds(point_cloud1, point_cloud2, point_cloud3, point_cloud4):
  """
  Plot four point clouds in an interactive 3D plot with Plotly.

  Args:
      point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
      point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
      point_cloud3 (np.ndarray): Third point cloud of shape (41, 3)
      point_cloud4 (np.ndarray): Fourth point cloud of shape (41, 3)
  """
  fig = go.Figure()

  # Define a color list for four point clouds
  colors = ['grey', 'yellow', 'red', 'blue']

  # Add traces for each point cloud with corresponding color
  for i, point_cloud in enumerate(
      [point_cloud1, point_cloud2, point_cloud3, point_cloud4]):
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers', marker=dict(size=2, color=colors[i]),
        name=f'Point Cloud {i+1}'
    ))

  for i in range(len(point_cloud3)):
      fig.add_trace(go.Scatter3d(
          x=[point_cloud3[i, 0], point_cloud4[i, 0]],
          y=[point_cloud3[i, 1], point_cloud4[i, 1]],
          z=[point_cloud3[i, 2], point_cloud4[i, 2]],
          mode='lines', line=dict(color='green', width=2),
          showlegend=False
      ))

  fig.update_layout(
      scene=dict(
          xaxis=dict(title='X'),
          yaxis=dict(title='Y'),
          zaxis=dict(title='Z'),
      ),
      margin=dict(r=20, l=10, b=10, t=10)
  )

  fig.show()

def save_4_point_clouds(point_cloud1, point_cloud2, point_cloud3, point_cloud4, filename="plot.html"):
  """
  Plot four point clouds in an interactive 3D plot with Plotly and save it.

  Args:
      point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
      point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
      point_cloud3 (np.ndarray): Third point cloud of shape (41, 3)
      point_cloud4 (np.ndarray): Fourth point cloud of shape (41, 3)
      filename (str, optional): Filename to save the plot. Defaults to "plot.png".
  """
  fig = go.Figure()

  # Define a color list for four point clouds
  colors = ['grey', 'yellow', 'red', 'blue']

  # Add traces for each point cloud with corresponding color
  for i, point_cloud in enumerate(
      [point_cloud1, point_cloud2, point_cloud3, point_cloud4]):
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers', marker=dict(size=2, color=colors[i]),
        name=f'Point Cloud {i+1}'
    ))

  for i in range(len(point_cloud3)):
      fig.add_trace(go.Scatter3d(
          x=[point_cloud3[i, 0], point_cloud4[i, 0]],
          y=[point_cloud3[i, 1], point_cloud4[i, 1]],
          z=[point_cloud3[i, 2], point_cloud4[i, 2]],
          mode='lines', line=dict(color='green', width=2),
          showlegend=False
      ))

  fig.update_layout(
      scene=dict(
          xaxis=dict(title='X'),
          yaxis=dict(title='Y'),
          zaxis=dict(title='Z'),
      ),
      margin=dict(r=20, l=10, b=10, t=10)
  )

  # Save the figure as a png image
  fig.write_html(filename)
def plot_point_clouds(point_cloud1, point_cloud2, title=""):
    """
    Plot two point clouds in an interactive 3D plot with Plotly.

    Args:
        point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
        point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
        mode='markers', marker=dict(size=2,color='red'), name='Point Cloud 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
        mode='markers', marker=dict(size=2,color='blue'), name='Point Cloud 2'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=10),
        title=title
    )

    fig.show()
def save_point_clouds(point_cloud1, point_cloud2, title="", filename="plot.html"):
    """
    Plot two point clouds in an interactive 3D plot with Plotly and save it.

    Args:
        point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
        point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
        title (str, optional): Title for the plot. Defaults to "".
        filename (str, optional): Filename to save the plot. Defaults to "plot.png".
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
        mode='markers', marker=dict(size=2,color='red'), name='Point Cloud 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
        mode='markers', marker=dict(size=2,color='blue'), name='Point Cloud 2'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=10),
        title=title
    )

    # Save the figure as a png image
    fig.write_html(filename)

def rotate_pointcloud_z_normal(point_cloud):
  """
  Rotates a point cloud such that the smallest eigenvalue (normal) of the covariance
  points in the z-axis direction.

  Args:
      pointcloud (np.ndarray): Point cloud of shape (41, 3).

  Returns:
      np.ndarray: Rotated point cloud with the smallest eigenvalue aligned to the z-axis.
  """

  cov_matrix = np.cov(point_cloud.T)

  # Compute eigenvectors and eigenvalues
  eigvals, eigvecs = eig(cov_matrix)

  # Find the index of the smallest eigenvalue
  min_eigval_index = np.argmin(eigvals)

  # Extract the corresponding eigenvector
  normal_vector = eigvecs[:, min_eigval_index]

  # Find rotation angles to align the normal vector with the z-axis
  theta = np.arctan2(normal_vector[1], normal_vector[0])
  phi = np.arccos(normal_vector[2] / np.linalg.norm(normal_vector))

  # Create rotation matrix
  rot_matrix = np.array([[np.cos(theta) * np.sin(phi), -np.sin(theta), np.cos(theta) * np.cos(phi)],
                         [np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)],
                         [-np.cos(phi), 0, np.sin(phi)]])

  # Apply rotation to the point cloud
  rotated_point_cloud = np.dot(rot_matrix, point_cloud.T).T

  return rotated_point_cloud
def get_k_nearest_neighbors(point_cloud, k):
    """
    Returns the k nearest neighbors for each point in the point cloud.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (pcl_size, 3)
        k (int): Number of nearest neighbors to return

    Returns:
        np.ndarray: Array of shape (1, 3, pcl_size, k) containing the k nearest neighbors for each point
    """
    pcl_size = point_cloud.shape[0]
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(point_cloud)
    distances, indices = neigh.kneighbors(point_cloud)

    neighbors_centered = np.empty((1, 3, pcl_size, k), dtype=point_cloud.dtype)
    neighbors_non_centered = np.empty((1, 3, pcl_size, k), dtype=point_cloud.dtype)

    for i in range(pcl_size):
        # orig = point_cloud[indices[i, 1:]] - point_cloud[indices[i, 1:]][0,:]
        orig = point_cloud[indices[i, 1:]] - point_cloud[indices[i, 1:]][0,:]
        # rotated1 = rotate_pointcloud_z_normal(orig)
        # rotated = rotate_to_z_axis(orig)
        # plot_point_clouds(rotated1, rotated)
        neighbors_centered[0, :, i, :] = orig.T
        neighbors_non_centered[0, :, i, :] = (point_cloud[indices[i, 1:]]).T

    return neighbors_centered, neighbors_non_centered


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx




def get_graph_feature(x, k=41, large_k=None):
    # x = x.squeeze()
    if large_k is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        idx = knn(x, k=large_k)
        idx = idx[:, :, torch.randperm(large_k)[:k]]
    batch_size, num_points, _ = idx.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature
def checkOnShapes(model_name='MLP5layers64Nlpe10xyz2deg40points', input_data=None, args_shape=None):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'{model_name}.pt'))
    model.eval()
    neighbors_centered, neighbors_non_centered = get_k_nearest_neighbors(input_data, 41)
    src_knn_pcl = torch.tensor(neighbors_centered)
    x_scale_src = torch.max(abs(src_knn_pcl[:, 0, :, :]))
    src_knn_pcl[:, 0, :, :] = src_knn_pcl[:, 0, :, :] / x_scale_src
    src_knn_pcl[:, 1, :, :] = src_knn_pcl[:, 1, :] / x_scale_src
    src_knn_pcl[:, 2, :, :] = src_knn_pcl[:, 2, :, :] / x_scale_src
    src_knn_pcl = 23 * src_knn_pcl
    output = model(src_knn_pcl)
    return output
def load_data(partition='test', divide_data=1):
    BASE_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes'
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

def checkData():
    args = configArgsPCT()
    args.std_dev = 0.05
    args.rotate_data = 1
    # args.batch_size = 40000
    max_list_x = []
    min_list_x = []
    max_list_y = []
    min_list_y = []
    max_list_z = []
    min_list_z = []
    train_dataset = BasicPointCloudDataset(file_path="train_surfaces_40_stronger_boundaries.h5", args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    with tqdm(train_dataloader) as tqdm_bar:
        for batch in tqdm_bar:
            pcl = batch['point_cloud']
            pcl = pcl.reshape(pcl.shape[0]*pcl.shape[1], -1)
            max_list_x.append(torch.max(pcl[:,0]).item())
            min_list_x.append(torch.min(pcl[:,0]).item())
            max_list_y.append(torch.max(pcl[:,1]).item())
            min_list_y.append(torch.min(pcl[:,1]).item())
            max_list_z.append(torch.max(pcl[:,2]).item())
            min_list_z.append(torch.min(pcl[:,2]).item())
            print('yay')
    print(f'-----------X--------------')
    print(f'MIN mean: {np.mean(min_list_x)}')
    print(f'MIN median: {np.median(min_list_x)}')
    print(f'MIN MIN: {np.min(min_list_x)}')
    print(f'MAX mean: {np.mean(max_list_x)}')
    print(f'MAX median: {np.median(max_list_x)}')
    print(f'MAX MAX: {np.max(max_list_x)}')
    print(f'-----------Y--------------')
    print(f'MIN mean: {np.mean(min_list_y)}')
    print(f'MIN median: {np.median(min_list_y)}')
    print(f'MIN MIN: {np.min(min_list_y)}')
    print(f'MAX mean: {np.mean(max_list_y)}')
    print(f'MAX median: {np.median(max_list_y)}')
    print(f'MAX MAX: {np.max(max_list_y)}')
    print(f'-----------Z--------------')
    print(f'MIN mean: {np.mean(min_list_z)}')
    print(f'MIN median: {np.median(min_list_z)}')
    print(f'MIN MIN: {np.min(min_list_z)}')
    print(f'MAX mean: {np.mean(max_list_z)}')
    print(f'MAX median: {np.median(max_list_z)}')
    print(f'MAX MAX: {np.max(max_list_z)}')
def visualizeShapesWithEmbeddings(model_name=None, args_shape=None):
    pcls, label = load_data()
    # for k in range(179,180): #human
    # for k in range(176,177): #hat
    # for k in range(174,175): #toilet
    # for k in range(162,163): #vase
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    shapes = [162, 174, 176, 179]
    # shapes = [179]
    for k in shapes:
        pointcloud = pcls[k][:]
        noisy_pointcloud = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors = checkOnShapes(model_name=model_name,
                                    input_data=pointcloud, args_shape=args_shape)
        colors = colors.detach().cpu().numpy()
        src_knn_pcl = src_knn_pcl.detach().cpu().numpy()

        colors_normalized = colors.copy()
        colors_normalized[:, 0] = ((colors[:, 0] - colors[:, 0].min()) / (
                    colors[:, 0].max() - colors[:, 0].min())) * 255
        colors_normalized[:, 1] = ((colors[:, 1] - colors[:, 1].min()) / (
                    colors[:, 1].max() - colors[:, 1].min())) * 255
        colors_normalized[:, 2] = ((colors[:, 2] - colors[:, 2].min()) / (
                    colors[:, 2].max() - colors[:, 2].min())) * 255
        colors_normalized[:, 3] = ((colors[:, 3] - colors[:, 3].min()) / (
                    colors[:, 3].max() - colors[:, 3].min())) * 255

        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        # Create and display 4 subplots with different colors and colorbars
        # for i in range(4):
        # for i in range(4):
        #     data = [
        #         go.Scatter3d(
        #             x=pointcloud[:, 0],
        #             y=pointcloud[:, 1],
        #             z=pointcloud[:, 2],
        #             mode='markers',
        #             marker=dict(
        #                 size=2,
        #                 opacity=0.8,
        #                 color=colors[:, i],
        #                 colorbar=dict(
        #                     title=f'{names[i]} Embedding',
        #                     y=1.1,
        #                     x=51
        #                 )
        #             ),
        #             name=f'Embedding Channel {i + 1}'
        #         ),
        #         #     go.Scatter3d(
        #         #         x=pointcloud[:, 0],
        #         #         y=pointcloud[:, 1],
        #         #         z=pointcloud[:, 2],
        #         #         mode='markers',
        #         #         marker=dict(
        #         #             size=2,
        #         #             opacity=0.8,
        #         #             color=['rgb(' + ', '.join(map(str, rgb)) + ')' for rgb in colors_normalized],  # Set RGB values
        #         # colorbar = dict(
        #         #     title='RGB Embeddings',
        #         #     y=1.1,
        #         #     x=0.95  # Adjust position of colorbar
        #         # )
        #         # ),
        #         # name = 'RGB Embeddings'
        #         # ),
        #         #     go.Scatter3d(
        #         #         x=src_knn_pcl[0, :],
        #         #         y=src_knn_pcl[1, :],
        #         #         z=src_knn_pcl[2, :],
        #         #         mode='markers',
        #         #         marker=dict(
        #         #             size=2,
        #         #             opacity=0.8,
        #         #             color='red'  # Set the color to red
        #         #         ),
        #         #         name='src_knn_pcl'
        #         #     )
        #     ]
        #     fig = go.Figure(data=data, layout=layout)
        #     fig.show()
        data_rgb = [
            go.Scatter3d(
                x=pointcloud[:, 0],
                y=pointcloud[:, 1],
                z=pointcloud[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    opacity=0.8,
                    color=['rgb(' + ', '.join(map(str, rgb)) + ')' for rgb in colors_normalized],  # Set RGB values
                ),
                name='RGB Embeddings'
            )
        ]

        # Your existing code

        # Plotting the RGB embeddings separately
        fig_rgb = go.Figure(data=data_rgb, layout=layout)
        fig_rgb.show()

        # Plot the maximum value embedding with specified colors
        max_embedding_index = np.argmax(colors, axis=1)
        max_embedding_colors = np.array(['red', 'blue', 'green', 'pink'])[max_embedding_index]

        data_max_embedding = []
        colors_shape = ['red', 'blue', 'green', 'pink']
        names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
        for color, name in zip(colors_shape, names):
            indices = np.where(max_embedding_colors == color)[0]
            data_max_embedding.append(
                go.Scatter3d(
                    x=pointcloud[indices, 0],
                    y=pointcloud[indices, 1],
                    z=pointcloud[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        opacity=0.8,
                        color=color
                    ),
                    name=f'Max Value Embedding - {name}'
                )
            )
        fig_max_embedding = go.Figure(data=data_max_embedding, layout=layout)
        fig_max_embedding.show()


def find_max_difference_indices(array, k=200):
    # Find the maximum values along axis 1
    max_values = np.max(array, axis=1)
    max_indices = np.argmax(array, axis=1)
    # Find the maximum values ignoring the maximum value itself along axis 1
    max_values_without_max = np.max(np.where(array == max_values[:, np.newaxis], -np.inf, array), axis=1)
    diff_from_max = (max_values - max_values_without_max)
    # kth_val = np.partition(diff_from_max, -k)[-k]
    # # Find the indices where the difference between the maximum value and the next maximum is at least 3
    # good_class_indices = np.where(diff_from_max >= kth_val)[0]
    good_class_indices = np.argpartition(diff_from_max, -k)[-k:]
    if len(good_class_indices) != k:
        raise Exception(f'Wrong size! k = {k}, actual size = {len(good_class_indices)}')
    return max_values, max_indices, diff_from_max, good_class_indices
def checkEmbeddingStability():
    pcls, label = load_data()
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    # plane_min = []
    # plane_max = []
    # peak_min = []
    # peak_max = []
    # ridge_min = []
    # ridge_max = []
    # saddle_min = []
    # saddle_max = []
    #
    # plane_diff_median_list = []
    # peak_diff_median_list = []
    # ridge_diff_median_list = []
    # saddle_diff_median_list  = []

    max_val_class_same_list = []
    max_val_diff_list = []
    reg_val_class_same_list = []
    reg_val_diff_list = []
    shapes = np.arange(pcls.shape[0])
    # shapes = np.arange(100) + 30
    for k in shapes:
        if k % 10 ==0:
            print(f'--------{k}--------')
        pointcloud = pcls[k][:]
        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)
        # plot_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2)
        colors_1 = checkOnShapes(model_name='MLP3layers32Nxyz2degRotStd005',
                                            input_data=noisy_pointcloud_1)
        colors_1 = colors_1.detach().cpu().numpy()
        colors_2 = checkOnShapes(model_name='MLP3layers32Nxyz2degRotStd005',
                                            input_data=noisy_pointcloud_2)
        colors_2 = colors_2.detach().cpu().numpy()

        max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(colors_1, threshold=10)
        max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(colors_2, threshold=10)
        reg_num_of_same_class = np.sum((max_indices_1 == max_indices_2))
        percentage = (reg_num_of_same_class / len(max_values_1))
        reg_val_diff_change_mean = np.mean((diff_from_max_1 - diff_from_max_2))
        reg_val_class_same_list.append(percentage)
        reg_val_diff_list.append(reg_val_diff_change_mean)


        union_indices = np.union1d(good_class_indices_1, good_class_indices_2)
        max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(colors_1[union_indices],
                                                                                                         threshold=10)
        max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(colors_2[union_indices],
                                                                                                         threshold=10)
        # plot_point_clouds(noisy_pointcloud_1, noisy_pointcloud_1[good_class_indices_1])
        # plot_point_clouds(noisy_pointcloud_2, noisy_pointcloud_2[good_class_indices_2])

        num_of_same_class = np.sum((max_indices_1==max_indices_2))
        percentage = (num_of_same_class / len(union_indices) )
        max_val_diff_change_mean =np.mean( (diff_from_max_1-diff_from_max_2) )
        max_val_class_same_list.append(percentage)
        max_val_diff_list.append(max_val_diff_change_mean)

        # plane_1, peak_1, ridge_1, saddle_1 = colors_1[:,0], colors_1[:,1], colors_1[:,2], colors_1[:,3]
        # plane_2, peak_2, ridge_2, saddle_2 = colors_2[:,0], colors_2[:,1], colors_2[:,2], colors_2[:,3]
        #
        # plane_diff = abs(plane_1 - plane_2)
        # plane_diff_median = np.median(plane_diff)
        # peak_diff = abs(peak_1 - peak_2)
        # peak_diff_median = np.median(peak_diff)
        # ridge_diff = abs(ridge_1 - ridge_2)
        # ridge_diff_median = np.median(ridge_diff)
        # saddle_diff = abs(saddle_1 - saddle_2)
        # saddle_diff_median = np.median(saddle_diff)
        # planes_top_10_indices, planes_top_10_values_orig, planes_top_10_values_second, planes_bottom_10_indices, planes_bottom_10_values, planes_bottom_10_values_second = top_and_bottom_values(plane_1, plane_2)
        # peaks_top_10_indices, peaks_top_10_values_orig, peaks_top_10_values_second, peaks_bottom_10_indices, peaks_bottom_10_values, peaks_bottom_10_values_second = top_and_bottom_values(
        #     peak_1, peak_2)
        # ridges_top_10_indices, ridges_top_10_values_orig, ridges_top_10_values_second, ridges_bottom_10_indices, ridges_bottom_10_values, ridges_bottom_10_values_second = top_and_bottom_values(
        #     ridge_1, ridge_2)
        # saddles_top_10_indices, saddles_top_10_values_orig, saddles_top_10_values_second, saddles_bottom_10_indices, saddles_bottom_10_values, saddles_bottom_10_values_second = top_and_bottom_values(
        #     saddle_1, saddle_2)
        #
        # plane_min.append(np.mean(abs(planes_top_10_values_orig - planes_top_10_values_second)))
        # plane_max.append(np.mean(abs(planes_bottom_10_values - planes_bottom_10_values_second)))
        # peak_min.append(np.mean(abs(peaks_top_10_values_orig - peaks_top_10_values_second)))
        # peak_max.append(np.mean(abs(peaks_bottom_10_values - peaks_bottom_10_values_second)))
        # ridge_min.append(np.mean(abs(ridges_top_10_values_orig - ridges_top_10_values_second)))
        # ridge_max.append(np.mean(abs(ridges_bottom_10_values - ridges_bottom_10_values_second)))
        # saddle_min.append(np.mean(abs(saddles_top_10_values_orig - saddles_top_10_values_second)))
        # saddle_max.append(np.mean(abs(saddles_bottom_10_values - saddles_bottom_10_values_second)))
        #
        # plane_diff_median_list.append(plane_diff_median)
        # peak_diff_median_list.append(peak_diff_median)
        # ridge_diff_median_list.append(ridge_diff_median)
        # saddle_diff_median_list.append(saddle_diff_median)

    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # axs[0, 0].plot(plane_min, label='Min')
    # axs[0, 0].plot(plane_max, label='Max')
    # axs[0, 0].plot(plane_diff_median_list, label='diff-median')
    # axs[0, 0].set_title('Plane')
    # axs[0, 0].legend()
    #
    # axs[0, 1].plot(peak_min, label='Min')
    # axs[0, 1].plot(peak_max, label='Max')
    # axs[0, 1].plot(peak_diff_median_list, label='diff-median')
    # axs[0, 1].set_title('Peak')
    # axs[0, 1].legend()
    #
    # axs[1, 0].plot(ridge_min, label='Min')
    # axs[1, 0].plot(ridge_max, label='Max')
    # axs[1, 0].plot(ridge_diff_median_list, label='diff-median')
    # axs[1, 0].set_title('Ridge')
    # axs[1, 0].legend()
    #
    # axs[1, 1].plot(saddle_min, label='Min')
    # axs[1, 1].plot(saddle_max, label='Max')
    # axs[1, 1].plot(saddle_diff_median_list, label='diff-median')
    # axs[1, 1].set_title('Saddle')
    # axs[1, 1].legend()
    #
    # plt.tight_layout()
    # plt.show()
    plt.figure(figsize=(10, 5))

    # Plot max_val_class_same_list
    plt.subplot(1, 2, 1)
    plt.plot(max_val_class_same_list, label='Percentage of Same Class Max Values')
    plt.plot(reg_val_class_same_list, label='Percentage of Same Class reg Values')
    plt.xlabel('Iterations')
    plt.ylabel('Percentage')
    plt.title('Percentage of Same Class Max Values vs. Iterations')
    plt.legend()

    # Plot max_val_diff_list
    plt.subplot(1, 2, 2)
    plt.plot(max_val_diff_list, label='Mean Difference in Max Values')
    plt.plot(reg_val_diff_list, label='Mean Difference in reg Values')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Difference')
    plt.title('Mean Difference in Max Values vs. Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()


def findRotTrans(model_name=None, args_shape=None, max_non_unique_correspondences=1):
    pcls, label = load_data()
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    shapes = np.arange(pcls.shape[0])
    # shapes = [162, 174, 176, 179]  # or any other range or selection of indices
    shapes = np.arange(1000) # or any other range or selection of indices
    num_worst_losses = 3  # Number of worst losses to track
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    num_of_inliers = []
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud #+ np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl #+ np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        colors_1 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_1, args_shape=args_shape)
        colors_2 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_2,args_shape=args_shape)
        #use only 32 embedding used for contrastive loss
        if args_shape.contrastive_mid_layer:
            colors_1 = colors_1[1]
            colors_2 = colors_2[1]
        colors_1 = colors_1.detach().cpu().numpy()
        colors_2 = colors_2.detach().cpu().numpy()


        if np.isnan(np.sum(colors_1)) or np.isnan(np.sum(colors_2)):
            print(f'oish')
            continue

        max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(colors_1,
                                                                                                         k=200)
        max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(colors_2,
                                                                                                         k=200)

        best_point_desc_pcl1 = colors_1[good_class_indices_1, :]
        best_point_desc_pcl2 = colors_1[good_class_indices_2, :]

        source_indices, target_indices = find_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
                                                             num_neighbors=40, max_non_unique_correspondences=max_non_unique_correspondences)
        # source_indices, target_indices = find_unique_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
        #                                                      num_neighbors=40)
        chosen_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :]
        chosen_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :]
        centered_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :] - np.mean(noisy_pointcloud_2)
        best_rotation, inliers = ransac(centered_points_1, centered_points_2, max_iterations=1000, threshold=0.1,
                                  min_inliers=10)
        num_of_inliers.append(len(inliers))
        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        losses.append(loss)
        # print(f'rotations etc:')
        # print(rotation_matrix @ best_rotation)
        # print(loss)
        # print(f'------------{k}------------')

        # Update the worst losses list
        for i, (worst_loss, _) in enumerate(worst_losses):
            if loss > worst_loss:
                worst_losses[i] = (loss, {
                    'noisy_pointcloud_1': noisy_pointcloud_1,
                    'noisy_pointcloud_2': noisy_pointcloud_2,
                    'chosen_points_1': chosen_points_1,
                    'chosen_points_2': chosen_points_2,
                    'rotation_matrix': rotation_matrix,
                    'best_rotation': best_rotation
                })
                break

    return worst_losses, losses, num_of_inliers
from scipy.spatial.distance import cdist

def find_closest_points(embeddings1, embeddings2, num_neighbors=40, max_non_unique_correspondences=1):
    # Initialize NearestNeighbors instance
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2)

    # Find the indices and distances of the closest points in embeddings2 for each point in embeddings1
    distances, indices = nbrs.kneighbors(embeddings1)
    duplicates = np.zeros(len(embeddings1))
    for i,index in enumerate(indices):
        if duplicates[index] >= max_non_unique_correspondences:
            distances[i]= np.inf
        duplicates[index] += 1
    # count_unnique_dup = np.count_nonzero(duplicates)
    # count_unnique =len(np.unique(indices))
    # if count_unnique != count_unnique_dup:
    #     print("FUCK!!!")
    smallest_distances_indices = np.argsort(distances.flatten())[:num_neighbors]
    emb1_indices = smallest_distances_indices.squeeze()
    emb2_indices = indices[smallest_distances_indices].squeeze()
    # a_count_yay = len(np.unique(indices[smallest_distances_indices]))
    return emb1_indices, emb2_indices

def random_rotation_translation(pointcloud):
  """
  Performs a random 3D rotation on a point cloud after centering it.

  Args:
      pointcloud: A NumPy array of shape (N, 3) representing the point cloud.

  Returns:
      A new NumPy array of shape (N, 3) representing the rotated point cloud.
  """
  # Center the point cloud by subtracting the mean of its coordinates
  center = np.mean(pointcloud, axis=0)
  centered_cloud = pointcloud - center

  # Generate random rotation angles for each axis
  theta_x = np.random.rand() * 2 * np.pi
  theta_y = np.random.rand() * 2 * np.pi
  theta_z = np.random.rand() * 2 * np.pi
  rotation_matrix = (Rotation.from_euler("xyz", [theta_x, theta_y, theta_z], degrees=False)).as_matrix()
  # Apply rotation to centered pointcloud
  rotated_cloud = (centered_cloud @ rotation_matrix)

  return ( rotated_cloud + center) , rotation_matrix


from scipy import spatial

from scipy.spatial.transform import Rotation


def ransac(data1, data2, max_iterations=1000, threshold=0.1, min_inliers=2):
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
    best_inliers = None
    best_rotation = None
    best_translation = None

    # src_mean = np.mean(data1, axis=0)
    # dst_mean = np.mean(data2, axis=0)

    src_centered = data1 #- src_mean
    dst_centered = data2 #- dst_mean

    for iteration in range(max_iterations):
        # Randomly sample 3 corresponding points
        indices = np.random.choice(N, size=4, replace=False)
        src_points = src_centered[indices]
        dst_points = dst_centered[indices]

        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        inliers1, inliers2 = find_inliers(src_centered, dst_centered, rotation, threshold)

        # Update best model if we have enough inliers
        if len(inliers1) >= min_inliers and (best_inliers is None or len(inliers1) > len(best_inliers)):
            best_inliers = inliers1
            best_rotation = rotation
            # best_translation = translation
    if best_inliers == None:
        return ransac(data1, data2, max_iterations=max_iterations, threshold=threshold + 0.1, min_inliers=min_inliers)
    transformed_1 = np.matmul(src_centered, best_rotation)
    # plot_point_clouds(src_centered, transformed_1, "src_centered vs fixed")
    # plot_point_clouds(transformed_1,dst_centered, "fixed vs target" )
    # print(len(best_inliers))
    # print(f'threshold is: {threshold}')
    return best_rotation, best_inliers


def estimate_rigid_transform(src_points, dst_points):
    """
    Estimates the rotation and translation that aligns two sets of 3D points.

    Args:
        src_points (np.ndarray): Array of shape (3, 3) containing the source points.
        dst_points (np.ndarray): Array of shape (3, 3) containing the destination points.

    Returns:
        rotation (np.ndarray): Array of shape (3, 3) representing the rotation matrix.
        translation (np.ndarray): Array of shape (3,) representing the translation vector.
    """

    H = np.matmul(src_points.T, dst_points)
    U, _, Vt = np.linalg.svd(H)
    # R = np.matmul(Vt.T, U.T)
    R = np.matmul(Vt, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        # R = np.matmul(Vt.T, U.T)
        R = np.matmul(Vt, U.T)



    return R


def find_inliers(data1, data2, rotation, threshold):
    """
    Finds the inliers in two sets of 3D points given a rotation and translation.

    Args:
        data1 (np.ndarray): Array of shape (N, 3) containing the first set of 3D points.
        data2 (np.ndarray): Array of shape (N, 3) containing the second set of 3D points.
        rotation (np.ndarray): Array of shape (3, 3) representing the rotation matrix.
        translation (np.ndarray): Array of shape (3,) representing the translation vector.
        threshold (float): Maximum distance for a point to be considered an inlier.

    Returns:
        inliers1 (list): List of indices of the inliers in data1.
        inliers2 (list): List of indices of the inliers in data2.
    """
    inliers1 = []
    inliers2 = []
    for i in range(data1.shape[0]):
        point1 = data1[i]
        point2 = data2[i]
        # transformed = np.matmul(point1, rotation)
        transformed = np.matmul(rotation, point1)
        distance = np.linalg.norm(transformed - point2)
        if distance < threshold:
            inliers1.append(i)
            inliers2.append(i)

    return inliers1, inliers2
def top_and_bottom_values(arr, arr_2, k = 10):
    # Get indices of sorted elements
    sorted_indices = np.argsort(arr)

    # Get top 10 indices and values
    top_10_indices = sorted_indices[-k:]
    top_10_values_orig = arr[top_10_indices]
    top_10_values_second = arr_2[top_10_indices]

    # Get bottom 10 indices and values
    bottom_10_indices = sorted_indices[:k]
    bottom_10_values = arr[bottom_10_indices]
    bottom_10_values_second = arr_2[bottom_10_indices]

    return top_10_indices, top_10_values_orig,top_10_values_second, bottom_10_indices, bottom_10_values, bottom_10_values_second
def plotWorst(worst_losses, model_name=""):
    for (loss,worst_loss_variables) in worst_losses:
        noisy_pointcloud_1 = worst_loss_variables['noisy_pointcloud_1']
        noisy_pointcloud_2 = worst_loss_variables['noisy_pointcloud_2']
        chosen_points_1 = worst_loss_variables['chosen_points_1']
        chosen_points_2 = worst_loss_variables['chosen_points_2']
        rotation_matrix = worst_loss_variables['rotation_matrix']
        best_rotation = worst_loss_variables['best_rotation']

        # plot_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2)
        save_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, title="", filename=model_name+f"_{loss:.3f}_orig.html")
        # plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2)
        save_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, filename=model_name+f"_{loss:.3f}_correspondence.html")

        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        save_point_clouds(transformed_points1, noisy_pointcloud_2 - center2, title="", filename=model_name+f"_{loss:.3f}_fixed.html")


def plot_losses(losses, inliers, filename="loss_plot.png"):
    """
    Plots the given list of losses and corresponding number of inliers,
    and saves the plot as an image.

    Args:
        losses (list): List of losses to plot.
        inliers (list): List of corresponding number of inliers.
        filename (str, optional): Filename to save the plot image. Defaults to "loss_plot.png".
    """
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)

    plt.figure(figsize=(8, 6))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot((40-np.array(inliers)), marker='o', linestyle='-', color='blue', label='Non_Inliers')
    ax2.plot(losses, marker='s', linestyle='-', color='red', label='Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('#Non_Inliers', color='blue')
    ax2.set_ylabel('Loss', color='red')

    ax1.grid(True)
    plt.title(f'mean loss = {mean_loss:.5f}, median loss = {median_loss:.5f}')

    # Save the plot as an image
    plt.savefig(filename)

    # You can optionally remove the following line if you don't want to display the plot as well
    plt.show()


if __name__ == '__main__':
    # checkData()
    # visualizeShapesWithEmbeddings()
    # visualizeShapesWithEmbeddings()
    args_shape = configArgsPCT()
    args_shape.batch_size = 1024
    args_shape.use_mlp = 1
    args_shape.num_mlp_layers = 3
    args_shape.num_neurons_per_layer = 32
    args_shape.sampled_points = 40
    args_shape.use_second_deg = 1
    args_shape.lpe_normalize = 1
    args_shape.lpe_dim = 0
    args_shape.num_mlp_layers = 3
    args_shape.contrastive = 1
    visualizeShapesWithEmbeddings(model_name='MLP3layers32Nxyz2degRotStd005ContrFromOrig10', args_shape=args_shape)
    # for i in range(1,5):
    for i in range(1,2):
        # model_name = 'MLP3layers32Nxyz2degRotStd005'
        model_name = 'MLP3layers32Nxyz2degRotStd005Contr32FromOrig10'
        args_shape = configArgsPCT()
        args_shape.batch_size = 1024
        args_shape.exp = model_name
        args_shape.use_mlp = 1
        args_shape.lpe_dim = 0
        args_shape.num_mlp_layers = 3
        args_shape.num_neurons_per_layer = 32
        args_shape.sampled_points = 40
        args_shape.use_second_deg = 1
        args_shape.lpe_normalize = 1
        args_shape.contrastive_mid_layer = 1
        # worst_losses, losses, num_of_inliers = findRotTrans(model_name=model_name, args_shape=args_shape,max_non_unique_correspondences=i)
        # plot_losses(losses, num_of_inliers, filename=f'{i}_'+"orig")
        # plotWorst(worst_losses, model_name=f'{i}_'+"orig")
        # model_name = 'MLP3layers32Nlpe3xyz2degRotStd005ContrFromOrig10'
        model_name = 'MLP3layers32Nlpe3xyz2degRotStd005Contr32FromOrig10'
        args_shape.exp = model_name
        args_shape.lpe_dim = 3
        args_shape.contrastive = 1
        # worst_losses, losses, num_of_inliers = findRotTrans(model_name=model_name, args_shape=args_shape,max_non_unique_correspondences=i)
        # plot_losses(losses, num_of_inliers, filename=f'{i}_'+"contrast_lpe3")
        # plotWorst(worst_losses, model_name=f'{i}_'+"contrast_lpe3")
        # visualizeShapesWithEmbeddings(model_name=model_name, args_shape=args_shape)
        # model_name = 'MLP3layers32Nxyz2degRotStd005ContrFromOrig10'
        model_name = 'MLP3layers32Nlpe6xyz2degRotStd005Contr4New'
        model_name = 'MLP3layers32Nlpe6xyz2degRotStd005Contr32NEWstdfun2'
        args_shape.exp = model_name
        args_shape.lpe_dim = 6
        args_shape.num_mlp_layers = 3
        args_shape.contrastive = 1
        worst_losses, losses, num_of_inliers  = findRotTrans(model_name = model_name, args_shape=args_shape,max_non_unique_correspondences=i)
        plot_losses(losses, num_of_inliers, filename=f'{i}_32_'+"contrast_lpe6")
        plotWorst(worst_losses, model_name=f'{i}_'+"contrast_lpe6")


