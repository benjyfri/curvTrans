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

def plot_point_clouds(point_cloud1, point_cloud2):
    """
    Plot two point clouds in an interactive 3D plot with Plotly.

    Args:
        point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
        point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
        mode='markers', marker=dict(color='red'), name='Point Cloud 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
        mode='markers', marker=dict(color='blue'), name='Point Cloud 2'
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
def checkOnShapes(model_name='MLP5layers64Nlpe10xyz2deg40points', input_data=None):
    args_shape = configArgsPCT()
    args_shape.batch_size = 1024
    args_shape.exp = model_name
    args_shape.use_mlp = 1
    args_shape.lpe_dim = 6
    args_shape.num_mlp_layers = 3
    args_shape.num_neurons_per_layer = 64
    args_shape.sampled_points = 40
    args_shape.use_second_deg = 0
    args_shape.lpe_normalize = 1
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'{model_name}.pt'))
    model.eval()
    neighbors_centered, neighbors_non_centered = get_k_nearest_neighbors(input_data, 41)
    src_knn_pcl = torch.tensor(neighbors_centered)
    src_knn_pcl_non_centered = torch.tensor(neighbors_non_centered)
    # src_knn_pcl = src_knn_pcl - src_knn_pcl[:,:,:,0].unsqueeze(3)
    # input_data = torch.tensor(input_data)
    # input_data = input_data.view(input_data.shape[1], input_data.shape[0])
    #
    #
    # src_knn = get_graph_feature(input_data.unsqueeze(0))
    # src_knn_pcl = src_knn[:, :3, :, :]
    # src_knn_pcl = src_knn[:, :3, :, :] - src_knn[:, 3:, :, :]
    x_scale_src = torch.max(abs(src_knn_pcl[:, 0, :, :]))
    # y_scale_src = torch.max(abs(src_knn_pcl[:, 1, :, :]))
    # z_scale_src = torch.max(abs(src_knn_pcl[:, 2, :, :]))
    # z_scale_src = x_scale_src / 2 + y_scale_src / 2
    src_knn_pcl[:, 0, :, :] = src_knn_pcl[:, 0, :, :] / x_scale_src
    src_knn_pcl[:, 1, :, :] = src_knn_pcl[:, 1, :] / x_scale_src
    src_knn_pcl[:, 2, :, :] = src_knn_pcl[:, 2, :, :] / x_scale_src
    src_knn_pcl = 20 * src_knn_pcl
    output = model(src_knn_pcl)
    preds = output.max(dim=1)[1]
    return output, src_knn_pcl_non_centered[0,:,0,:]
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
if __name__ == '__main__':
    # checkData()
    args = configArgsPCT()
    pcls, label = load_data()
    # for k in range(160,190):
    # for k in range(160,161):
    # for k in range(179,180): #human
    # for k in range(176,177): #hat
    # for k in range(174,175): #toilet
    # for k in range(162,163): #vase
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    shapes = [160, 162, 174,176,179]
    # shapes = [179]
    for k in shapes:
        pointcloud = pcls[k][:]

        colors, src_knn_pcl = checkOnShapes(model_name='MLP3layers64Nlpe6xyzRotStd005', input_data=pointcloud)
        colors = colors.detach().cpu().numpy()
        src_knn_pcl = src_knn_pcl.detach().cpu().numpy()

        colors_normalized = colors.copy()
        colors_normalized[:,0] = ((colors[:,0] - colors[:,0].min()) / (colors[:,0].max() - colors[:,0].min())) * 255
        colors_normalized[:,1] = ((colors[:,1] - colors[:,1].min()) / (colors[:,1].max() - colors[:,1].min())) * 255
        colors_normalized[:,2] = ((colors[:,2] - colors[:,2].min()) / (colors[:,2].max() - colors[:,2].min())) * 255
        colors_normalized[:,3] = ((colors[:,3] - colors[:,3].min()) / (colors[:,3].max() - colors[:,3].min())) * 255

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
        for i in range(4):
            data = [
                go.Scatter3d(
                    x=pointcloud[:, 0],
                    y=pointcloud[:, 1],
                    z=pointcloud[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        opacity=0.8,
                        color=colors[:, i],
                        colorbar=dict(
                            title=f'{names[i]} Embedding',
                            y=1.1,
                            x = 51
                        )
                    ),
                    name=f'Embedding Channel {i + 1}'
                ),
            #     go.Scatter3d(
            #         x=pointcloud[:, 0],
            #         y=pointcloud[:, 1],
            #         z=pointcloud[:, 2],
            #         mode='markers',
            #         marker=dict(
            #             size=2,
            #             opacity=0.8,
            #             color=['rgb(' + ', '.join(map(str, rgb)) + ')' for rgb in colors_normalized],  # Set RGB values
            # colorbar = dict(
            #     title='RGB Embeddings',
            #     y=1.1,
            #     x=0.95  # Adjust position of colorbar
            # )
            # ),
            # name = 'RGB Embeddings'
            # ),
                go.Scatter3d(
                    x=src_knn_pcl[0, :],
                    y=src_knn_pcl[1, :],
                    z=src_knn_pcl[2, :],
                    mode='markers',
                    marker=dict(
                        size=2,
                        opacity=0.8,
                        color='red'  # Set the color to red
                    ),
                    name='src_knn_pcl'
                )
            ]
            fig = go.Figure(data=data, layout=layout)
            fig.show()
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
