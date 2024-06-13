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

def plot_multiclass_point_clouds(point_clouds_1, point_clouds_2, rotation=None, title=""):
    """
    Plot pairs of sub point clouds in an interactive 3D plot with Plotly.

    Args:
        point_clouds_1 (list of np.ndarray): List of sub point clouds from pcl1.
        point_clouds_2 (list of corresponding np.ndarray): List of sub point clouds from pcl2.
        rotation (np.ndarray): Rotation matrix to apply to the point clouds.
        title (str): Title of the plot.
    """
    fig = go.Figure()

    # Filter out empty sub point clouds
    point_clouds_1 = [pc for pc in point_clouds_1 if len(pc) > 0]
    point_clouds_2 = [np.array(pc) for pc in point_clouds_2 if len(pc) > 0]

    if rotation is not None:
        center = np.mean(np.vstack(point_clouds_1), axis=0)
        point_clouds_1 = [np.matmul((pc - center), rotation) for pc in point_clouds_1]
        axis = np.argmin(np.max(np.vstack(point_clouds_1), axis=0))
        for pc in point_clouds_1:
            pc[:, axis] += 1.5

    # Define a broad range of colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(cmap.N)]

    # Plot each point cloud from point_clouds_1 and point_clouds_2 with corresponding color
    for i, point_cloud1 in enumerate(point_clouds_1):
        fig.add_trace(go.Scatter3d(
            x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
            mode='markers', marker=dict(size=2, color=f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})', opacity=0.5),
            name=f'Class {i+1}'
        ))
    for i, point_cloud2 in enumerate(point_clouds_2):
        fig.add_trace(go.Scatter3d(
            x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
            mode='markers', marker=dict(size=2, color=f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})', opacity=0.5),
            name=f'Class {i+1}'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=100)
    )

    fig.show()
def plot_8_point_clouds(cls1_1, cls1_2, cls1_3, cls1_4, cls2_1, cls2_2, cls2_3, cls2_4, rotation=None, title=""):
    """
    Plot four pairs of sub point clouds in an interactive 3D plot with Plotly.

    Args:
        cls1_1, cls1_2, cls1_3, cls1_4 (np.ndarray): Sub point clouds from pcl1.
        cls2_1, cls2_2, cls2_3, cls2_4 (np.ndarray): Corresponding sub point clouds from pcl2.
        rotation (np.ndarray): Rotation matrix to apply to the point clouds.
        title (str): Title of the plot.
    """
    fig = go.Figure()

    # Combine the inputs into lists for easier iteration
    pcl1 = [cls1_1, cls1_2, cls1_3, cls1_4]
    pcl1 = [x for x in pcl1 if len(x)>0]

    pcl2 = [cls2_1, cls2_2, cls2_3, cls2_4]
    pcl2 = [np.array(x) for x in pcl2 if len(x) > 0]

    if rotation is not None:
        center = np.mean(np.vstack(pcl1), axis=0)
        pcl1 = [np.matmul((pcl - center), rotation) for pcl in pcl1]
        axis = np.argmin(np.max(np.vstack(pcl1), axis=0))
        for pcl in pcl1:
            pcl[:, axis] += 1.5

    # Define a color list for four point clouds
    colors = ['blue', 'orange', 'brown', 'red']
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    # Plot each point cloud from pcl1 and pcl2 with corresponding color
    # Plot each point cloud from pcl1 and pcl2 with corresponding color
    for i, (point_cloud1) in enumerate(pcl1):
        fig.add_trace(go.Scatter3d(
            x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
            mode='markers', marker=dict(size=2, color=colors[i], opacity=0.5),
            name=f'PCL1 - {names[i]}'
        ))
    for i, (point_cloud2) in enumerate(pcl2):
        fig.add_trace(go.Scatter3d(
            x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
            mode='markers', marker=dict(size=2, color=colors[i], opacity=0.5),
            name=f'PCL2 - {names[i]}'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=100)
    )

    fig.show()

def plot_4_point_clouds(point_cloud1, point_cloud2, point_cloud3, point_cloud4, rotation=None, title=""):
  """
  Plot four point clouds in an interactive 3D plot with Plotly.

  Args:
      point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
      point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
      point_cloud3 (np.ndarray): Third point cloud of shape (41, 3)
      point_cloud4 (np.ndarray): Fourth point cloud of shape (41, 3)
  """
  fig = go.Figure()
  if rotation is not None:
      center = np.mean(point_cloud1, axis=0)
      point_cloud1 = np.matmul((point_cloud1 - center), rotation.T)
      point_cloud3 = np.matmul((point_cloud3 - center), rotation.T)
      axis = np.argmin(np.max(point_cloud1, axis=0))

      point_cloud1[:, axis] = point_cloud1[:, axis] + 1.5
      point_cloud3[:, axis] = point_cloud3[:, axis] + 1.5

  # Define a color list for four point clouds
  colors = ['blue', 'orange', 'brown', 'red']

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
          mode='lines', line=dict(color='black', width=2),
          showlegend=False
      ))

  fig.update_layout(
      title=title,
      scene=dict(
          xaxis=dict(title='X'),
          yaxis=dict(title='Y'),
          zaxis=dict(title='Z'),
      ),
      margin=dict(r=20, l=10, b=10, t=100)
  )

  fig.show()

def save_4_point_clouds(point_cloud1, point_cloud2, point_cloud3, point_cloud4, filename="plot.html", rotation=None, title=""):
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
  colors = ['blue', 'orange', 'brown', 'red']

  if rotation is not None:
      center = np.mean(point_cloud1, axis=0)
      point_cloud1 = np.matmul((point_cloud1 - center), rotation)
      point_cloud3 = np.matmul((point_cloud3 - center), rotation)
      axis = np.argmin(np.max(point_cloud1, axis=0))

      point_cloud1[:, axis] = point_cloud1[:, axis] + 1.5
      point_cloud3[:, axis] = point_cloud3[:, axis] + 1.5

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
          mode='lines', line=dict(color='black', width=2),
          showlegend=False
      ))

  fig.update_layout(
      title=title,
      scene=dict(
          xaxis=dict(title='X'),
          yaxis=dict(title='Y'),
          zaxis=dict(title='Z'),
      ),
      margin=dict(r=20, l=10, b=10, t=100)
  )

  # Save the figure as a png image
  fig.write_html(filename)

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

def plot_point_clouds(point_cloud1, point_cloud2, title=""):
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
    # Each point cloud should be centered around first point which is at the origin
    for i in range(pcl_size):
        orig = point_cloud[indices[i, 1:]] - point_cloud[indices[i, 1:]][0,:]
        neighbors_centered[0, :, i, :] = orig.T
        neighbors_non_centered[0, :, i, :] = (point_cloud[indices[i, 1:]]).T

    return neighbors_centered, neighbors_non_centered
def find_mean_diameter_for_specific_coordinate(specific_coordinates):
    pairwise_distances = torch.cdist(specific_coordinates.unsqueeze(2), specific_coordinates.unsqueeze(2))
    largest_dist = pairwise_distances.view(specific_coordinates.shape[0], -1).max(dim=1).values
    mean_distance = torch.mean(largest_dist)
    return mean_distance

def checkOnShapes(model_name=None, input_data=None, args_shape=None, scaling_factor=None):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'{model_name}.pt'))
    model.eval()
    neighbors_centered, neighbors_non_centered = get_k_nearest_neighbors(input_data, 41)
    src_knn_pcl = torch.tensor(neighbors_centered)
    x_scale_src = find_mean_diameter_for_specific_coordinate(src_knn_pcl[0,0,:,:])
    y_scale_src = find_mean_diameter_for_specific_coordinate(src_knn_pcl[0,1,:,:])
    z_scale_src = find_mean_diameter_for_specific_coordinate(src_knn_pcl[0,2,:,:])
    scale = torch.mean(torch.stack((x_scale_src, y_scale_src, z_scale_src), dim=0))
    #scale KNN point clouds to be of size 1
    src_knn_pcl = src_knn_pcl / scale
    #use 23 as it is around the size of the synthetic point clouds
    src_knn_pcl = scaling_factor * src_knn_pcl
    output = model(src_knn_pcl.permute(2,1,0,3))
    return output
def load_data(partition='test', divide_data=1):
    BASE_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes'
    DATA_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes\\data'
    # DATA_DIR = r'/content/curvTrans/bbsWithShapes/data'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        print(f'++++++++{h5_name}++++++++')
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
def visualizeShapesWithEmbeddings(model_name=None, args_shape=None, scaling_factor=None):
    pcls, label = load_data()
    shapes = [86, 174, 179]
    # shapes = [86]
    for k in shapes:
        pointcloud = pcls[k][:]
        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud #+ np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        surface_labels , colors = checkOnShapes(model_name=model_name,
                                    input_data=pointcloud, args_shape=args_shape, scaling_factor=scaling_factor)
        colors = colors.detach().cpu().numpy()
        colors = colors[:,:4]

        colors_normalized = colors.copy()
        colors_normalized[:, 0] = ((colors[:, 0] - colors[:, 0].min()) / (
                    colors[:, 0].max() - colors[:, 0].min())) * 255
        colors_normalized[:, 1] = ((colors[:, 1] - colors[:, 1].min()) / (
                    colors[:, 1].max() - colors[:, 1].min())) * 255
        colors_normalized[:, 2] = ((colors[:, 2] - colors[:, 2].min()) / (
                    colors[:, 2].max() - colors[:, 2].min())) * 255
        colors_normalized[:, 3] = ((colors[:, 3] - colors[:, 3].min()) / (
                    colors[:, 3].max() - colors[:, 3].min())) * 255
        colors_normalized = np.clip(colors_normalized, 0, 255).astype(np.uint8)
        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
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

        # Plot the maximum value embedding with specified colors
        max_embedding_index = surface_labels
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
    good_class_indices = np.argpartition(diff_from_max, -k)[-k:]
    if len(good_class_indices) != k:
        raise Exception(f'Wrong size! k = {k}, actual size = {len(good_class_indices)}')
    # sorted_indices = np.argsort(max_values)
    # k_largest_indices = sorted_indices[-k:]
    # good_class_indices = k_largest_indices
    return max_values, max_indices, diff_from_max, good_class_indices

def findRotTrans(model_name=None, args_shape=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None):
    pcls, label = load_data()
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
    shapes = np.arange(pcls.shape[0])
    shapes = np.arange(1000) # or any other range or selection of indices
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    iter_2_ransac_convergence = []
    num_of_inliers = []
    shape_size_list = []
    shortest_dist_list = []
    avg_dist_list = []
    dist_from_orig = []
    # shapes = [86, 162, 174, 176, 179]
    shapes = np.arange(1000)
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_2,args_shape=args_shape, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue
        #
        # max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
        #                                                                                                  k=200)
        # max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
        #                                                                                                  k=200)

        good_class_indices_1 = farthest_point_sampling(emb_1[:,:4],k=200)
        good_class_indices_2 = farthest_point_sampling(emb_2[:,:4],k=200)

        best_point_desc_pcl1 = emb_1[good_class_indices_1, :]
        best_point_desc_pcl2 = emb_2[good_class_indices_2, :]

        source_indices, target_indices = find_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
                                                             num_neighbors=40, max_non_unique_correspondences=max_non_unique_correspondences)


        chosen_indices_pcl1 = good_class_indices_1[source_indices]
        chosen_indices_pcl2 = good_class_indices_2[target_indices]
        chosen_points_1 = noisy_pointcloud_1[chosen_indices_pcl1, :]
        chosen_points_2 = noisy_pointcloud_2[chosen_indices_pcl2, :]

        amount_same_index = np.sum(chosen_indices_pcl1 == chosen_indices_pcl2)
        original_points_chosen = pointcloud[chosen_indices_pcl1, :]
        original_corresponding_points = pointcloud[chosen_indices_pcl2, :]
        pairwise_distances = cdist(pointcloud, pointcloud)
        shape_size = np.max(pairwise_distances)
        shape_size_list.append(shape_size)
        correspondences_dist = (pairwise_distances[chosen_indices_pcl1,chosen_indices_pcl2])
        dist_from_orig.append(np.mean(correspondences_dist))
        mask = ~np.eye(pairwise_distances.shape[0], dtype=bool)
        flattened_arr = pairwise_distances[mask]
        avg_dist_list.append(np.mean(flattened_arr))
        shortest_dist_list.append(np.min(flattened_arr))
        centered_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :] - np.mean(noisy_pointcloud_2)
        best_rotation, inliers, best_iter = ransac(centered_points_1, centered_points_2, max_iterations=1000, threshold=0.1,
                                  min_inliers=10)
        num_of_inliers.append(len(inliers))
        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        losses.append(loss)
        iter_2_ransac_convergence.append(best_iter)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if loss > smallest_loss:
            worst_losses[index_of_smallest_loss] = (loss, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': chosen_points_1,
                'chosen_points_2': chosen_points_2,
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })

    return worst_losses, losses, num_of_inliers, iter_2_ransac_convergence, shape_size_list, dist_from_orig, shortest_dist_list, avg_dist_list

def find_closest_points(embeddings1, embeddings2, num_neighbors=40, max_non_unique_correspondences=3):
    classification_1 = np.argmax(embeddings1[:,:4], axis=1)
    classification_2 = np.argmax(embeddings2[:,:4], axis=1)

    # Initialize NearestNeighbors instance
    # nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2[:,4:])
    # nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2[:,:4])

    # Find the indices and distances of the closest points in embeddings2 for each point in embeddings1
    # distances, indices = nbrs.kneighbors(embeddings1[:,4:])
    # distances, indices = nbrs.kneighbors(embeddings1)
    distances, indices = nbrs.kneighbors(embeddings1[:,:4])
    duplicates = np.zeros(len(embeddings1))
    for i,index in enumerate(indices):
        if duplicates[index] >= max_non_unique_correspondences:
            distances[i]= np.inf
        duplicates[index] += 1
    same_class = (classification_1==(classification_2[indices].squeeze()))
    distances[~same_class] = np.inf

    smallest_distances_indices = np.argsort(distances.flatten())[:num_neighbors]
    emb1_indices = smallest_distances_indices.squeeze()
    emb2_indices = indices[smallest_distances_indices].squeeze()
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
    best_iter = 0
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
            best_iter = iteration
    if best_inliers == None:
        return ransac(data1, data2, max_iterations=max_iterations, threshold=threshold + 0.1, min_inliers=min_inliers)
    return best_rotation, best_inliers, best_iter

def classification_only_ransac(cls1_1,cls1_2,cls1_3,cls1_4, cls2_1,cls2_2,cls2_3,cls2_4,
                               max_iterations=1000, threshold=0.1, min_inliers=2):
    # N = data1.shape[0]
    best_num_of_inliers = 0
    corres = None
    best_rotation = None
    best_translation = None

    # Choose one array based on the probabilities
    pcl_1 = []
    pcl_2 = []
    for pcl1,pcl2 in zip([cls1_1, cls1_2, cls1_3, cls1_4], [cls2_1, cls2_2, cls2_3, cls2_4]):
        if len(pcl1)>0 and len(pcl2)>0:
            pcl_1.append(pcl1)
            pcl_2.append(pcl2)

    sizes = [len(cls) for cls in pcl_1]
    total_size = sum(sizes)
    probabilities = [(1 - (size / total_size)) for size in sizes if size > 0]
    if len(probabilities) > 1:
        probabilities = [(prob / (len(probabilities) - 1)) for prob in probabilities]
    else:
        probabilities = [1]
    best_iter = 0
    for iteration in range(max_iterations):
        # if iteration%10==0:
        #     print(f'Iteration {iteration}')
        # Randomly sample 3 corresponding points
        chosen_classes = random.choices(range(len(probabilities)), weights=probabilities, k=3)
        src_points = np.array([random.choice(pcl_1[cls]) for cls in chosen_classes])
        dst_points = np.array([random.choice(pcl_2[cls]) for cls in chosen_classes])
        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        inliers_1, inliers_2 = find_inliers_classification(cls1_1, cls1_2, cls1_3, cls1_4, cls2_1, cls2_2, cls2_3, cls2_4, rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return classification_only_ransac(cls1_1,cls1_2,cls1_3,cls1_4, cls2_1,cls2_2,cls2_3,cls2_4, max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_num_of_inliers, best_iter, corres, threshold

def multiclass_classification_only_ransac(cls_1, cls_2, max_iterations=1000, threshold=0.1, min_inliers=2):
    # N = data1.shape[0]
    best_num_of_inliers = 0
    corres = None
    best_rotation = None
    best_translation = None

    # Choose one array based on the probabilities
    pcl_1 = []
    pcl_2 = []
    for pcl1,pcl2 in zip(cls_1, cls_2):
        if len(pcl1)>0 and len(pcl2)>0:
            pcl_1.append(pcl1)
            pcl_2.append(pcl2)

    sizes = [len(cls) for cls in pcl_1]
    total_size = sum(sizes)
    probabilities = [(1 - (size / total_size)) for size in sizes if size > 0]
    if len(probabilities) > 1:
        probabilities = [(prob / (len(probabilities) - 1)) for prob in probabilities]
    else:
        probabilities = [1]
    best_iter = 0
    for iteration in range(max_iterations):
        # if iteration%10==0:
        #     print(f'Iteration {iteration}')
        # Randomly sample 3 corresponding points
        chosen_classes = random.choices(range(len(probabilities)), weights=probabilities, k=3)
        src_points = np.array([random.choice(pcl_1[cls]) for cls in chosen_classes])
        dst_points = np.array([random.choice(pcl_2[cls]) for cls in chosen_classes])
        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        inliers_1, inliers_2 = find_inliers_classification_multiclass(cls_1, cls_2, rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return multiclass_classification_only_ransac(cls_1, cls_2, max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_num_of_inliers, best_iter, corres, threshold

def random_only_ransac(cls1_1,cls1_2,cls1_3,cls1_4, cls2_1,cls2_2,cls2_3,cls2_4,
                               max_iterations=1000, threshold=0.1, min_inliers=2):
    # N = data1.shape[0]
    best_num_of_inliers = 0
    corres = None
    best_rotation = None
    best_translation = None

    # Choose one array based on the probabilities
    yay1 = [pcl for pcl in [cls1_1, cls1_2, cls1_3, cls1_4] if len(pcl) > 0]
    yay2 = [pcl for pcl in [cls2_1, cls2_2, cls2_3, cls2_4] if len(pcl) > 0]
    pcl_1 = np.vstack(yay1)
    # pcl_1 = np.random.permutation(pcl_1)
    pcl_2 = np.vstack(yay2)
    # pcl_2 = np.random.permutation(pcl_2)

    best_iter = 0
    for iteration in range(max_iterations):
        src_points = np.array([random.choice(pcl_1) for i in range(3)])
        dst_points = np.array([random.choice(pcl_2) for j in range(3)])
        # Estimate rotation and translation
        rotation = estimate_rigid_transform(src_points, dst_points)
        # translation = dst_mean - np.matmul(src_mean, rotation)
        # Find inliers
        dummy_1 = pcl_1.tolist()
        dummy_2 = pcl_2.tolist()
        inliers_1, inliers_2 = find_inliers_classification(dummy_1, [], [], [], dummy_2, [], [], [], rotation=rotation, threshold=threshold)

        # Update best model if we have enough inliers
        if len(inliers_1) >= min_inliers and ( (best_num_of_inliers == 0 ) or (len(inliers_1) > best_num_of_inliers) ):
            best_num_of_inliers = len(inliers_1)
            corres = [inliers_1, inliers_2]
            best_rotation = rotation
            # best_translation = translation
            best_iter = iteration
    if best_num_of_inliers == 0:
        return random_only_ransac(dummy_1, [], [], [], dummy_2, [], [], [], max_iterations=max_iterations,
                                                   threshold=threshold + 0.1,
                                                   min_inliers=min_inliers)
    return best_rotation, best_num_of_inliers, best_iter, corres, threshold


def estimate_rigid_transform(src_points, dst_points):
    H = np.matmul(src_points.T, dst_points)
    U, _, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt, U.T)
    return R


def find_inliers(data1, data2, rotation, threshold):
    """
    Finds the inliers in two sets of 3D points given a rotation and translation.
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
def find_inliers_classification(cls1_1, cls1_2, cls1_3, cls1_4, cls2_1, cls2_2, cls2_3, cls2_4, rotation, threshold=0.1):
    """
    Finds the inliers in two sets of labeled 3D points for each class using 1-nearest neighbor.
    """
    inliers_1 = []
    inliers_2 = []

    # Combine the inputs into lists for easier iteration
    orig_cls1 = []
    cls2 = []
    for pcl1,pcl2 in zip([cls1_1, cls1_2, cls1_3, cls1_4], [cls2_1, cls2_2, cls2_3, cls2_4]):
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
    # print(f'inliers_1:')
    # print([x.shape for x in inliers_1])
    # print(f'inliers_2:')
    # print([x.shape for x in inliers_2])
    return np.vstack(inliers_1), np.vstack(inliers_2)
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
    # print(f'inliers_1:')
    # print([x.shape for x in inliers_1])
    # print(f'inliers_2:')
    # print([x.shape for x in inliers_2])
    return np.vstack(inliers_1), np.vstack(inliers_2)
def plotWorst(worst_losses, model_name=""):
    count = 0
    for (loss,worst_loss_variables) in worst_losses:
        noisy_pointcloud_1 = worst_loss_variables['noisy_pointcloud_1']
        noisy_pointcloud_2 = worst_loss_variables['noisy_pointcloud_2']
        chosen_points_1 = worst_loss_variables['chosen_points_1']
        chosen_points_2 = worst_loss_variables['chosen_points_2']
        rotation_matrix = worst_loss_variables['rotation_matrix']
        best_rotation = worst_loss_variables['best_rotation']
        save_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, title="", filename=model_name+f"_{loss:.3f}_orig_{count}_loss.html")
        save_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, filename=model_name+f"_{loss:.3f}_correspondence_{count}_loss.html", rotation=rotation_matrix)

        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        save_point_clouds(transformed_points1, noisy_pointcloud_2 - center2, title="", filename=model_name+f"_{loss:.3f}_{count}_loss.html")
        count = count + 1


def plot_and_save_with_stats(numbers, name):
    # Calculate mean and median
    mean_value = np.mean(numbers)
    median_value = np.median(numbers)

    # Create the plot
    plt.figure()
    plt.plot(numbers)
    plt.grid(True)
    plt.title(f'Mean: {mean_value:.2f}, Median: {median_value:.2f}')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Save the plot
    plt.savefig(f'{name}_plot.png')
    plt.show()
def plot_losses(losses, inliers, filename="loss_plot.png"):
    """
    Plots the given list of losses and corresponding number of inliers,
    and saves the plot as an image
    """
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)

    plt.figure(figsize=(8, 6))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot((np.array(inliers)), marker='o', linestyle='-', color='blue', label='Inliers')
    ax2.plot(losses, marker='s', linestyle='-', color='red', label='Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('#Inliers', color='blue')
    ax2.set_ylabel('Loss', color='red')

    ax1.grid(True)
    plt.title(f'mean loss = {mean_loss:.5f}, median loss = {median_loss:.5f}')

    # Save the plot as an image
    plt.savefig(filename)

    # You can optionally remove the following line if you don't want to display the plot as well
    plt.show()



def plot_distances(Max_dist, min_dist, avg_dist_list, dist_from_orig, filename="dist_plot.png"):
    # Create x-axis values (assuming len(Max_dist) = len(min_dist) = len(dist_from_orig))
    x = list(range(len(Max_dist)))

    # Plot Max_dist
    plt.plot(x, Max_dist, label=f'Max Distances {np.mean(Max_dist):.3f}', color='blue')

    # Plot min_dist
    plt.plot(x, min_dist, label=f'Min Distances {np.mean(min_dist):.3f}', color='red')

    # Plot min_dist
    plt.plot(x, avg_dist_list, label=f'AVG Distances {np.mean(avg_dist_list):.3f}', color='purple')

    # Plot dist_from_orig
    plt.plot(x, dist_from_orig, label=f'Dist from Origin {np.mean(dist_from_orig):.3f}', color='green')

    plt.grid(True)
    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Distance')
    # Calculate mean and median of dist_from_orig
    # mean_dist = np.mean(dist_from_orig)
    # median_dist = np.median(dist_from_orig)

    # Add mean and median to the title
    title = f'Distances Plot'
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    # Show the plot
    plt.show()

def test_coress_dis(model_name=None, args_shape=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None):
    pcls, label = load_data()
    shape_size_list = []
    shortest_dist_list = []
    avg_dist_list = []
    dist_from_orig = []
    # diameter_20_nn_list = []
    shapes = [86, 162, 174, 176, 179]
    # shapes = np.arange(100)
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=10)
        emb_2 = checkOnShapes(model_name=model_name,
                                    input_data=noisy_pointcloud_2,args_shape=args_shape, scaling_factor=10)
        # diameter_20_nn_list.append(diameter_20_nn)
        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue

        # max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
        #                                                                                                  k=200)
        # max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
        #                                                                                                  k=200)

        good_class_indices_1 = farthest_point_sampling(emb_1[:, :4], k=200)
        good_class_indices_2 = farthest_point_sampling(emb_2[:, :4], k=200)

        best_point_desc_pcl1 = emb_1[good_class_indices_1, :]
        best_point_desc_pcl2 = emb_2[good_class_indices_2, :]

        source_indices, target_indices = find_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
                                                             num_neighbors=40, max_non_unique_correspondences=1)


        chosen_indices_pcl1 = good_class_indices_1[source_indices]
        chosen_indices_pcl2 = good_class_indices_2[target_indices]
        chosen_points_1 = noisy_pointcloud_1[chosen_indices_pcl1, :]
        chosen_points_2 = noisy_pointcloud_2[chosen_indices_pcl2, :]

        amount_same_index = np.sum(chosen_indices_pcl1 == chosen_indices_pcl2)
        original_points_chosen = pointcloud[chosen_indices_pcl1, :]
        original_corresponding_points = pointcloud[chosen_indices_pcl2, :]
        pairwise_distances = cdist(pointcloud, pointcloud)
        shape_size = np.max(pairwise_distances)
        shape_size_list.append(shape_size)
        correspondences_dist = (pairwise_distances[chosen_indices_pcl1,chosen_indices_pcl2])
        flat_array = correspondences_dist.flatten()
        sorted_values = np.sort(flat_array)
        best_30 = sorted_values[:(len(correspondences_dist)//10) *3]
        best_30 = np.append(best_30, shape_size)
        # best_30 = np.append(best_30, diameter_20_nn)
        # Step 5: Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(best_30)), best_30)
        plt.grid(True)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Smallest 30% Values in the Array (Ordered)')
        plt.show()
        dist_from_orig.append(np.mean(best_30))
        mask = ~np.eye(pairwise_distances.shape[0], dtype=bool)
        flattened_arr = pairwise_distances[mask]
        avg_dist_list.append(np.mean(flattened_arr))
        shortest_dist_list.append(np.min(flattened_arr))
        centered_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :] - np.mean(noisy_pointcloud_2)
        best_rotation, inliers, best_iter = ransac(centered_points_1, centered_points_2, max_iterations=1000,
                                                   threshold=0.05,
                                                   min_inliers=5)
        center = np.mean(noisy_pointcloud_1, axis=0)
        center2 = np.mean(noisy_pointcloud_2, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, rotation=rotation_matrix,
                            title=f'Loss is {loss:.3f}; <br>Shape size: {np.max(pairwise_distances)}; <br>best_30: {np.mean(best_30)};')

    return 1, 1, 1, 1, shape_size_list, dist_from_orig, shortest_dist_list, avg_dist_list

def test_classification(cls_args=None,contr_args=None,smooth_args=None, max_non_unique_correspondences=3, num_worst_losses = 3, scaling_factor=None, point_choice=0, num_of_ransac_iter=100, random_pairing=0):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    worst_point_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    shapes = [86, 162, 174, 176, 179]
    # shapes = np.arange(1000)
    num_of_points_to_sample = 100
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_2,args_shape=cls_args, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue
        if point_choice == 0:
            max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
                                                                                                             k=num_of_points_to_sample)
            max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
                                                                                                             k=num_of_points_to_sample)
        if point_choice == 1:
            max_values_1 = np.max(emb_1[:, :4], axis=1)
            good_class_indices_1 = np.argsort(max_values_1)[-num_of_points_to_sample:][::-1]
            max_values_2 = np.max(emb_2[:, :4], axis=1)
            good_class_indices_2 = np.argsort(max_values_2)[-num_of_points_to_sample:][::-1]
        if point_choice == 2:
            good_class_indices_1 = farthest_point_sampling(emb_1[:, :4], k=num_of_points_to_sample)
            good_class_indices_2 = farthest_point_sampling(emb_2[:, :4], k=num_of_points_to_sample)

        classification_pcl1 = np.argmax((emb_1[good_class_indices_1, :]), axis=1)
        classification_pcl2 = np.argmax((emb_2[good_class_indices_2, :]), axis=1)

        # # learned_emb_1 = checkOnShapes(model_name=contr_args.exp,
        # #                       input_data=noisy_pointcloud_1[good_class_indices_1, :], args_shape=contr_args, scaling_factor=scaling_factor)
        # # learned_emb_2 = checkOnShapes(model_name=contr_args.exp,
        # #                       input_data=noisy_pointcloud_2[good_class_indices_2, :] , args_shape=contr_args, scaling_factor=scaling_factor)
        #
        # learned_emb_1 = checkOnShapes(model_name=smooth_args.exp,
        #                       input_data=noisy_pointcloud_1[good_class_indices_1, :], args_shape=smooth_args, scaling_factor=scaling_factor)
        # learned_emb_2 = checkOnShapes(model_name=smooth_args.exp,
        #                       input_data=noisy_pointcloud_2[good_class_indices_2, :] , args_shape=smooth_args, scaling_factor=scaling_factor)
        #
        # learned_emb_1 = learned_emb_1.detach().cpu().numpy()
        # learned_emb_2 = learned_emb_2.detach().cpu().numpy()
        #
        # best_point_desc_pcl1 = np.hstack((emb_1[good_class_indices_1, :], learned_emb_1))
        # best_point_desc_pcl2 = np.hstack((emb_2[good_class_indices_2, :], learned_emb_2))
        #
        # source_indices, target_indices = find_closest_points(best_point_desc_pcl1, best_point_desc_pcl2,
        #                                                      num_neighbors=40,
        #                                                      max_non_unique_correspondences=1)
        #
        # chosen_indices_pcl1 = good_class_indices_1[source_indices]
        # chosen_indices_pcl2 = good_class_indices_2[target_indices]
        # chosen_points_1 = noisy_pointcloud_1[chosen_indices_pcl1, :]
        # chosen_points_2 = noisy_pointcloud_2[chosen_indices_pcl2, :]
        #
        # centered_points_1 = noisy_pointcloud_1[good_class_indices_1[source_indices], :] - np.mean(noisy_pointcloud_1)
        # centered_points_2 = noisy_pointcloud_2[good_class_indices_2[target_indices], :] - np.mean(noisy_pointcloud_2)
        # best_rotation, inliers, best_iter = ransac(centered_points_1, centered_points_2, max_iterations=1000,
        #                                            threshold=0.1,
        #                                            min_inliers=10)

        centered_points_1 = noisy_pointcloud_1[good_class_indices_1, :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2, :] - np.mean(noisy_pointcloud_2)

        # Concatenate along the second axis (axis=1)
        result_1 = np.concatenate((classification_pcl1.reshape(len(classification_pcl1), 1), centered_points_1), axis=1)
        result_2 = np.concatenate((classification_pcl2.reshape(len(classification_pcl2), 1), centered_points_2), axis=1)
        cls1_0 = [pcl[1:] for pcl in result_1 if pcl[0]==0]
        cls1_1 = [pcl[1:] for pcl in result_1 if pcl[0]==1]
        cls1_2 = [pcl[1:] for pcl in result_1 if pcl[0]==2]
        cls1_3 = [pcl[1:] for pcl in result_1 if pcl[0]==3]
        cls2_0 = [pcl[1:] for pcl in result_2 if pcl[0]==0]
        cls2_1 = [pcl[1:] for pcl in result_2 if pcl[0]==1]
        cls2_2 = [pcl[1:] for pcl in result_2 if pcl[0]==2]
        cls2_3 = [pcl[1:] for pcl in result_2 if pcl[0]==3]
        plot_8_point_clouds( cls1_0, cls1_1, cls1_2, cls1_3, cls2_0, cls2_1, cls2_2, cls2_3,  rotation=rotation_matrix)
        if random_pairing == 1:
            best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = random_only_ransac(
                cls1_0, cls1_1, cls1_2, cls1_3, cls2_0, cls2_1, cls2_2, cls2_3, max_iterations=num_of_ransac_iter,
                                                       threshold=0.1,
                                                       min_inliers=num_of_points_to_sample/10)
        else:
            best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = classification_only_ransac(
                cls1_0, cls1_1, cls1_2, cls1_3, cls2_0, cls2_1, cls2_2, cls2_3, max_iterations=num_of_ransac_iter,
                threshold=0.1,
                min_inliers=num_of_points_to_sample / 10)
        final_thresh_list.append(final_threshold)
        final_inliers_list.append(best_num_of_inliers)
        iter_2_ransac_convergence.append(best_iter)

        # center = np.mean(noisy_pointcloud_1, axis=0)
        # transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        # loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        # losses.append(loss)
        #
        # kdtree = cKDTree(transformed_points1)
        #
        # # Query the KDTree with points from pcl2
        # distances, indices = kdtree.query(noisy_pointcloud_2)
        #
        # point_distance = np.mean(distances)
        # point_distance_list.append(point_distance)
        #
        # plot_point_clouds(transformed_points1, noisy_pointcloud_2, f'loss is: {loss}; best iter: {best_iter}')
        # plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, rotation=rotation_matrix.T)

        plot_point_clouds(transformed_points1, noisy_pointcloud_2, f'loss is: {loss}; inliers: {best_num_of_inliers}; threshold: {final_threshold}')
        plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, corres[0], corres[1], rotation=rotation_matrix.T)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if loss > smallest_loss:
            worst_losses[index_of_smallest_loss] = (loss, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_point_losses[i][0] for i in range(len(worst_point_losses))])
        smallest_loss = worst_point_losses[index_of_smallest_loss][0]
        if point_distance > smallest_loss:
            worst_point_losses[index_of_smallest_loss] = (point_distance, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
    return worst_losses, losses, final_thresh_list, final_inliers_list, point_distance_list, worst_point_losses, iter_2_ransac_convergence

def test_multi_scale_classification(cls_args=None,num_worst_losses = 3, scaling_factor=None, point_choice=0, num_of_ransac_iter=100, subsampled_points=100):
    pcls, label = load_data()
    worst_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    worst_point_losses = [(0, None)] * num_worst_losses  # Initialize with (loss, variables)
    losses = []
    point_distance_list = []
    final_thresh_list = []
    final_inliers_list = []
    iter_2_ransac_convergence = []
    # shapes = [86, 162, 174, 176, 179]
    shapes = [86, 162]
    shapes = np.arange(100)
    num_of_points_to_sample = subsampled_points
    for k in shapes:
        if k%10 ==0:
            print(f'------------{k}------------')
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=noisy_pointcloud_2,args_shape=cls_args, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        if np.isnan(np.sum(emb_1)) or np.isnan(np.sum(emb_2)):
            print(f'oish')
            continue
        if point_choice == 0:
            max_values_1, max_indices_1, diff_from_max_1, good_class_indices_1 = find_max_difference_indices(emb_1[:,:4],
                                                                                                             k=num_of_points_to_sample)
            max_values_2, max_indices_2, diff_from_max_2, good_class_indices_2 = find_max_difference_indices(emb_2[:,:4],
                                                                                                             k=num_of_points_to_sample)
        if point_choice == 1:
            max_values_1 = np.max(emb_1[:, :4], axis=1)
            good_class_indices_1 = np.argsort(max_values_1)[-num_of_points_to_sample:][::-1]
            max_values_2 = np.max(emb_2[:, :4], axis=1)
            good_class_indices_2 = np.argsort(max_values_2)[-num_of_points_to_sample:][::-1]
        if point_choice == 2:
            good_class_indices_1 = farthest_point_sampling(emb_1[:, :4], k=num_of_points_to_sample)
            good_class_indices_2 = farthest_point_sampling(emb_2[:, :4], k=num_of_points_to_sample)

        classification_pcl1 = np.argmax((emb_1[good_class_indices_1, :]), axis=1)
        classification_pcl2 = np.argmax((emb_2[good_class_indices_2, :]), axis=1)

        centered_points_1 = noisy_pointcloud_1[good_class_indices_1, :] - np.mean(noisy_pointcloud_1)
        centered_points_2 = noisy_pointcloud_2[good_class_indices_2, :] - np.mean(noisy_pointcloud_2)


        global_emb_1 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=centered_points_1, args_shape=cls_args, scaling_factor=scaling_factor)
        global_emb_2 = checkOnShapes(model_name=cls_args.exp,
                                    input_data=centered_points_2,args_shape=cls_args, scaling_factor=scaling_factor)

        global_emb_1 = global_emb_1.detach().cpu().numpy()
        global_emb_2 = global_emb_2.detach().cpu().numpy()

        global_classification_pcl1 = np.argmax((global_emb_1), axis=1)
        global_classification_pcl2 = np.argmax((global_emb_2), axis=1)

        pcl1_classes = [[] for _ in range(16)]

        # Iterate over the point cloud and classification arrays
        for i in range(len(centered_points_1)):
            index = classification_pcl1[i] * 4 + global_classification_pcl1[i]
            pcl1_classes[index].append(centered_points_1[i,:])

        pcl2_classes = [[] for _ in range(16)]

        # Iterate over the point cloud and classification arrays
        for i in range(len(centered_points_1)):
            index = classification_pcl2[i] * 4 + global_classification_pcl2[i]
            pcl2_classes[index].append(centered_points_2[i,:])

        # plot_multiclass_point_clouds(pcl1_classes, pcl2_classes, rotation=rotation_matrix, title="")

        best_rotation, best_num_of_inliers, best_iter, corres, final_threshold = multiclass_classification_only_ransac(
            pcl1_classes, pcl2_classes, max_iterations=num_of_ransac_iter,
            threshold=0.1,
            min_inliers=num_of_points_to_sample / 10)
        final_thresh_list.append(final_threshold)
        final_inliers_list.append(best_num_of_inliers)
        iter_2_ransac_convergence.append(best_iter)

        center = np.mean(noisy_pointcloud_1, axis=0)
        transformed_points1 = np.matmul((noisy_pointcloud_1 - center), best_rotation.T)
        loss = np.mean(((rotation_matrix @ best_rotation) - np.eye(3)) ** 2)
        losses.append(loss)

        kdtree = cKDTree(transformed_points1)

        # Query the KDTree with points from pcl2
        distances, indices = kdtree.query(noisy_pointcloud_2)

        point_distance = np.mean(distances)
        point_distance_list.append(point_distance)

        # plot_point_clouds(transformed_points1, noisy_pointcloud_2, f'loss is: {loss}; inliers: {best_num_of_inliers}; threshold: {final_threshold}')
        # plot_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, corres[0], corres[1], rotation=rotation_matrix.T)

        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_losses[i][0] for i in range(len(worst_losses))])
        smallest_loss = worst_losses[index_of_smallest_loss][0]
        if loss > smallest_loss:
            worst_losses[index_of_smallest_loss] = (loss, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
        # Update the worst losses list
        index_of_smallest_loss = np.argmin([worst_point_losses[i][0] for i in range(len(worst_point_losses))])
        smallest_loss = worst_point_losses[index_of_smallest_loss][0]
        if point_distance > smallest_loss:
            worst_point_losses[index_of_smallest_loss] = (point_distance, {
                'noisy_pointcloud_1': noisy_pointcloud_1,
                'noisy_pointcloud_2': noisy_pointcloud_2,
                'chosen_points_1': corres[0],
                'chosen_points_2': corres[1],
                'rotation_matrix': rotation_matrix,
                'best_rotation': best_rotation
            })
    return worst_losses, losses, final_thresh_list, final_inliers_list, point_distance_list, worst_point_losses, iter_2_ransac_convergence


def view_stabiity(model_name=None, args_shape=None, scaling_factor=None):
    pcls, label = load_data()
    shapes = [82, 83, 86, 174]
    for k in shapes:
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = checkOnShapes(model_name=model_name,
                              input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=scaling_factor)
        emb_2 = checkOnShapes(model_name=model_name,
                              input_data=noisy_pointcloud_2, args_shape=args_shape, scaling_factor=scaling_factor)
        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()
        # embeddings_1 = []
        # embeddings_2 = []
        # for i in range(2,5):
        #     scaling_factor = i * 10
        #
        #     emb_1 = checkOnShapes(model_name=model_name,
        #                                 input_data=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=scaling_factor)
        #
        #     emb_2 = checkOnShapes(model_name=model_name,
        #                                 input_data=noisy_pointcloud_2, args_shape=args_shape, scaling_factor=scaling_factor)
        #
        #     emb_1 = emb_1.detach().cpu().numpy()
        #     emb_2 = emb_2.detach().cpu().numpy()
        #     embeddings_1.append(emb_1[:,4:])
        #     embeddings_2.append(emb_2[:,4:])
        # emb_1 = np.hstack((embeddings_1))
        # emb_2 = np.hstack((embeddings_2))
        if np.isnan(np.sum(emb_1)):
            print(f'oish')
            continue
        # plot_point_cloud_with_colors_by_dist(noisy_pointcloud_1, emb_1)
        # plot_point_cloud_with_colors_by_dist(noisy_pointcloud_1, emb_1[:,:4])
        # plot_point_cloud_with_colors_by_dist(noisy_pointcloud_1, emb_1[:,4:])

        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2)
        # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1[:,:4], emb_2[:,:4])
        # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1[:,4:], emb_2[:,4:])

def plot_point_cloud_with_colors_by_dist(point_cloud, embedding):
    # Generate random index to choose a point
    random_index = np.random.randint(len(point_cloud))
    random_index = 10
    random_point = point_cloud[random_index]
    random_embedding = embedding[random_index]

    # Calculate distances from the random embedding to all other embeddings
    distances = np.linalg.norm(embedding - random_embedding, axis=1)

    # Find indices of the 10 closest points
    closest_indices = np.argsort(distances)[:10]

    # Define color scale based on distances
    max_distance = distances.max()
    min_distance = distances.min()
    colors = [(d - min_distance) / (max_distance - min_distance) for d in distances]
    colors = 1 / np.array(colors)
    colors = np.log(colors)

    # Plot point cloud with colors based on distances
    fig = go.Figure(data=[go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale='plasma',
            opacity=0.8,
            colorbar=dict(title='Distance')
        )
    )])

    # Add the 10 closest points as a separate trace for better visibility
    closest_points = point_cloud[closest_indices]
    fig.add_trace(go.Scatter3d(
        x=closest_points[:, 0],
        y=closest_points[:, 1],
        z=closest_points[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='green',  # 10 closest points are red
            opacity=1
        )
    ))

    # Add the randomly chosen point as a separate trace for better visibility
    fig.add_trace(go.Scatter3d(
        x=[random_point[0]],
        y=[random_point[1]],
        z=[random_point[2]],
        mode='markers',
        marker=dict(
            size=8,
            color='red',  # Chosen point is red
            opacity=1
        )
    ))
    fig.show()

def plot_point_cloud_with_colors_by_dist_2_pcls(point_cloud1, point_cloud2, embedding1, embedding2):
    # Generate random index to choose a point from point_cloud1
    random_index = np.random.randint(len(point_cloud1))
    random_index = 10
    random_point = point_cloud2[random_index]
    random_embedding = embedding1[random_index]

    # Calculate distances from the random embedding to all embeddings in embedding2
    distances = np.linalg.norm(embedding2 - random_embedding, axis=1)

    # Find indices of the 20 closest points
    closest_indices = np.argsort(distances)[:20]

    # Define color scale based on distances
    max_distance = distances.max()
    min_distance = distances.min()
    colors = [(d - min_distance) / (max_distance - min_distance) for d in distances]
    colors = 1 / np.array(colors)
    colors = np.log(colors)

    # Plot point cloud 2 with colors based on distances
    fig = go.Figure(data=[go.Scatter3d(
        x=point_cloud2[:, 0],
        y=point_cloud2[:, 1],
        z=point_cloud2[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale='plasma',
            opacity=0.8,
            colorbar=dict(title='Distance')
        )
    )])

    # Add the 20 closest points as a separate trace for better visibility
    closest_points = point_cloud2[closest_indices]
    fig.add_trace(go.Scatter3d(
        x=closest_points[:, 0],
        y=closest_points[:, 1],
        z=closest_points[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='green',  # 20 closest points are green
            opacity=1
        )
    ))

    # Add the randomly chosen point from point_cloud1 as a separate trace for better visibility
    fig.add_trace(go.Scatter3d(
        x=[random_point[0]],
        y=[random_point[1]],
        z=[random_point[2]],
        mode='markers',
        marker=dict(
            size=8,
            color='red',  # Chosen point is red
            opacity=1
        )
    ))

    fig.show()
def fit_surface_quadratic_constrained(points):
    """
    Fits a quadratic surface constrained to f = 0 to a centered point cloud.

    Args:
      points: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      numpy array of shape (5,) representing the surface coefficients
        [a, b, c, d, e], where:
          z = a * x**2 + b * y**2 + c * x * y + d * x + e * y
    """

    # Center the points around the mean
    centroid = points[0,:]
    centered_points = points - centroid

    # Design matrix without f term
    X = np.c_[centered_points[:, 0] ** 2, centered_points[:, 1] ** 2,
              centered_points[:, 0] * centered_points[:, 1],
    centered_points[:, 0], centered_points[:, 1]]

    # Extract z-coordinates as target vector
    z = centered_points[:, 2]

    # Solve the linear system with f coefficient constrained to 0
    coeffs = np.linalg.lstsq(X, z, rcond=None)[0]

    a, b, c, d, e = coeffs

    K = (4 * (a * b) - ((c ** 2))) / ((1 + d ** 2 + e ** 2) ** 2)
    H = (a * (1 + e ** 2) - d * e * c + b * (1 + d ** 2)) / (((d ** 2) + (e ** 2) + 1) ** 1.5)


    gaussian = 0
    if K > 0.05:
        gaussian = 1
    if K < -0.05:
        gaussian = -1

    mean = 0
    if H > 0.05:
        gaussian = 1
    if H < -0.05:
        gaussian = -1

    if gaussian == 0 and mean == 0:
        return 0
    if gaussian == 1:
        return 1
    if gaussian == 0:
        return 2
    return 3
def fix_orientation(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    rotation_axis = np.cross(np.array([0, 0, 1]), normal_at_centroid)

    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), normal_at_centroid))
    cosine_angle = np.arccos(np.dot(np.array([0, 0, 1]), normal_at_centroid))

    rotation_matrix = np.array([
        [1 + (1 - cosine_angle) * (rotation_axis[0] ** 2),
         (1 - cosine_angle) * rotation_axis[0] * rotation_axis[1] - rotation_angle * rotation_axis[2],
         (1 - cosine_angle) * rotation_axis[0] * rotation_axis[2] + rotation_angle * rotation_axis[1]],
        [(1 - cosine_angle) * rotation_axis[1] * rotation_axis[0] + rotation_angle * rotation_axis[2],
         1 + (1 - cosine_angle) * (rotation_axis[1] ** 2),
         (1 - cosine_angle) * rotation_axis[1] * rotation_axis[2] - rotation_angle * rotation_axis[0]],
        [(1 - cosine_angle) * rotation_axis[2] * rotation_axis[0] - rotation_angle * rotation_axis[1],
         (1 - cosine_angle) * rotation_axis[2] * rotation_axis[1] + rotation_angle * rotation_axis[0],
         1 + (1 - cosine_angle) * (rotation_axis[2] ** 2)]
    ])

    # Apply the rotation to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)

    return rotated_point_cloud + centroid
def plot_point_clouds(point_cloud1, point_cloud2, title):
    """
    Plot two point clouds in an interactive 3D plot with Plotly.

    Args:
        point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
        point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
        title (str): Title of the plot
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
        mode='markers', marker=dict(color='red'),opacity=0.8, name='Point Cloud 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
        mode='markers', marker=dict(color='blue'),opacity=0.8, name='Point Cloud 2'
    ))
    # Add a separate trace for the point (0, 0, 0) in bright pink
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(color='rgb(255, 105, 180)'), name='Origin (0, 0, 0)'
    ))

    fig.update_layout(
        title=title,  # Set the title
        title_y=0.9,  # Adjust the y position of the title
        scene=dict(
            xaxis=dict(title='X', range=[min(point_cloud1[:,0].min(), point_cloud2[:,0].min()),
                                         max(point_cloud1[:,0].max(), point_cloud2[:,0].max())]),
            yaxis=dict(title='Y', range=[min(point_cloud1[:,1].min(), point_cloud2[:,1].min()),
                                         max(point_cloud1[:,1].max(), point_cloud2[:,1].max())]),
            zaxis=dict(title='Z', range=[min(point_cloud1[:,2].min(), point_cloud2[:,2].min()),
                                         max(point_cloud1[:,2].max(), point_cloud2[:,2].max())])
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig.show()
def random_rotation(point_cloud):
    """
    Applies a random rotation to a 3D point cloud.

    Args:
        point_cloud: A NumPy array of shape (N, 3) representing the point cloud.

    Returns:
        A NumPy array of shape (N, 3) representing the rotated point cloud.
    """

    is_rotation = False
    while not is_rotation:
        # Generate random rotation angles around x, y, and z axes
        theta_x = np.random.uniform(0, 2 * np.pi)
        theta_y = np.random.uniform(0, 2 * np.pi)
        theta_z = np.random.uniform(0, 2 * np.pi)

        # Rotation matrices around x, y, and z axes
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), -np.sin(theta_x)],
                       [0, np.sin(theta_x), np.cos(theta_x)]])

        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                       [0, 1, 0],
                       [-np.sin(theta_y), 0, np.cos(theta_y)]])

        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                       [np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])

        # Combine rotation matrices
        R = np.matmul(Rz, np.matmul(Ry, Rx))

        # Check if rotation is valid (determinant close to 1)
        is_rotation = np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    # Apply rotation to point cloud
    rotated_point_cloud = np.matmul(point_cloud, R.T)
    return rotated_point_cloud


def read_bin_file(bin_file):
    """
    Read a .bin file and return a numpy array of shape (N, 3) where N is the number of points.

    Args:
        bin_file (str): Path to the .bin file.

    Returns:
        np.ndarray: Numpy array containing the point cloud data.
    """
    # Load the binary file
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

    # We only need the first three columns (x, y, z)
    return points[:, :3]
def create_args(cls_model_name, contrastive_model_name, smoothing_model_name):
    cls_args_shape = configArgsPCT()
    cls_args_shape.batch_size = 1024
    cls_args_shape.num_mlp_layers = 3
    cls_args_shape.num_neurons_per_layer = 32
    cls_args_shape.sampled_points = 40
    cls_args_shape.use_second_deg = 1
    cls_args_shape.lpe_normalize = 1
    cls_args_shape.exp = cls_model_name
    cls_args_shape.lpe_dim = 6
    cls_args_shape.output_dim = 4
    contrastive_args_shape = configArgsPCT()
    contrastive_args_shape.batch_size = 1024
    contrastive_args_shape.num_mlp_layers = 3
    contrastive_args_shape.num_neurons_per_layer = 32
    contrastive_args_shape.sampled_points = 40
    contrastive_args_shape.use_second_deg = 1
    contrastive_args_shape.lpe_normalize = 1
    contrastive_args_shape.exp = contrastive_model_name
    contrastive_args_shape.lpe_dim = 6
    contrastive_args_shape.output_dim = 8
    smoothing_args_shape = configArgsPCT()
    smoothing_args_shape.batch_size = 1024
    smoothing_args_shape.num_mlp_layers = 3
    smoothing_args_shape.num_neurons_per_layer = 32
    smoothing_args_shape.sampled_points = 40
    smoothing_args_shape.use_second_deg = 1
    smoothing_args_shape.lpe_normalize = 1
    smoothing_args_shape.exp = smoothing_model_name
    smoothing_args_shape.lpe_dim = 6
    smoothing_args_shape.output_dim = 8
    return cls_args_shape, contrastive_args_shape, smoothing_args_shape
if __name__ == '__main__':
    args_shape = configArgsPCT()
    cls_args, contr_args,smooth_args =  create_args(cls_model_name='3MLP32Ncls', contrastive_model_name='3MLP32Ncontr', smoothing_model_name='3MLP32Nsmooth')
    scaling_factor = 10
    # view_stabiity(model_name=contr_args.exp, args_shape=contr_args, scaling_factor=scaling_factor)
    # view_stabiity(model_name=smooth_args.exp, args_shape=smooth_args, scaling_factor=scaling_factor)
    # for point_choice in range(3):
    #     worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
    #         = test_classification(cls_args=cls_args, contr_args=contr_args, smooth_args=smooth_args, max_non_unique_correspondences=5,
    #                               scaling_factor=scaling_factor, point_choice=point_choice, num_of_ransac_iter=100,
    #                               random_pairing=0)
    #     print(f'--------------------{point_choice}--------------------')
    #     print(f'loss avg: {np.mean(losses)}')
    #     plotWorst(worst_losses, model_name=f'_{point_choice}_')

    all_num_of_inliers = []
    all_mean_losses = []
    all_iter_2_ransac_convergence = []
    labels = []

    # range_check_1 = [500]
    range_check_1 = np.arange(700, 1100, 100)
    range_check_2 = np.arange(100, 1100, 100)
    # range_check_2 = np.arange(800, 1100, 100)
    # range_check_2 = np.arange(100, 1100, 100)
    # for point_choice in [0,1,2]:
    for point_choice in [2]:
        for ransac_iter in range_check_1:
            for amount_of_points_to_subsample in range_check_2:
                print(f'----------------------')
                print(f'Point choice: {point_choice}, ransac iter: {ransac_iter}, #subsample: {amount_of_points_to_subsample}')
                worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
                    = test_multi_scale_classification(cls_args=cls_args, num_worst_losses=3, scaling_factor=scaling_factor, point_choice=point_choice,
                                                num_of_ransac_iter=ransac_iter, subsampled_points=amount_of_points_to_subsample)
                np.save(f'multi_output/{point_choice}_{ransac_iter}_{amount_of_points_to_subsample}_losses.npy', losses)
                np.save(f'multi_output/{point_choice}_{ransac_iter}_{amount_of_points_to_subsample}_point_distance_list.npy', point_distance_list)
                np.save(f'multi_output/{point_choice}_{ransac_iter}_{amount_of_points_to_subsample}_num_of_inliers.npy', num_of_inliers)
                np.save(f'multi_output/{point_choice}_{ransac_iter}_{amount_of_points_to_subsample}_iter_2_ransac_convergence.npy', iter_2_ransac_convergence)
                # Calculate mean loss
                mean_loss = np.mean(losses)

                # Store the results
                all_num_of_inliers.append(num_of_inliers)
                all_mean_losses.append(mean_loss)
                all_iter_2_ransac_convergence.append(iter_2_ransac_convergence)
                labels.append(f'PC: {point_choice}, RI: {ransac_iter}, SP: {amount_of_points_to_subsample}')

    # Plot mean loss vs number of inliers
    plt.figure(figsize=(12, 6))
    for i in range(len(all_num_of_inliers)):
        plt.plot(range(len(all_mean_losses[i])), all_mean_losses[i], label=labels[i])
    plt.xlabel('Number of Inliers')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss vs Number of Inliers')
    plt.legend()
    plt.show()

    # Plot iteration-wise RANSAC convergence
    plt.figure(figsize=(12, 6))
    for i in range(len(all_iter_2_ransac_convergence)):
        plt.plot(range(len(all_iter_2_ransac_convergence[i])), all_iter_2_ransac_convergence[i], label=labels[i])
    plt.xlabel('Iteration')
    plt.ylabel('RANSAC Convergence')
    plt.title('Iter 2 RANSAC Convergence')
    plt.legend()
    plt.show()

    # Plot all number of inliers
    plt.figure(figsize=(12, 6))
    for i in range(len(all_num_of_inliers)):
        plt.plot(range(len(all_num_of_inliers[i])), all_num_of_inliers[i], label=labels[i])
    plt.xlabel('Configuration Index')
    plt.ylabel('Number of Inliers')
    plt.title('Number of Inliers for Different Configurations')
    plt.legend()
    plt.show()
# for i in range(1,2):
    # # checkData()
    # # for i in range(6,12):
    # # for i in range(6,7):
    #
    #     scaling_factor = i + 9
    #     args_shape = configArgsPCT()
    #     args_shape.batch_size = 1024
    #     # args_shape.num_mlp_layers = 4
    #     args_shape.num_mlp_layers = 3
    #     args_shape.num_neurons_per_layer = 32
    #     args_shape.sampled_points = 40
    #     args_shape.use_second_deg = 1
    #     args_shape.lpe_normalize = 1
    #     # model_name = '4MLP32Nlpe6ContrNEWstdfun2Weight05_05LR001std01'
    #     # model_name = '4MLP32Nlpe6ContrNEWstdfun2Weight05_02LR001std01NN5'
    #     model_name = '3MLP32Ncls'
    #     args_shape.exp = model_name
    #     args_shape.lpe_dim = 6
    #     args_shape.output_dim = 4
    #     # visualizeShapesWithEmbeddings(model_name=model_name, args_shape=args_shape, scaling_factor=10)
    #     # worst_losses, losses, num_of_inliers, iter_2_ransac_convergence,shape_size_list, dist_from_orig, shortest_dist_list, avg_dist_list  \
    #     #     = test_coress_dis(model_name=model_name, args_shape=args_shape,max_non_unique_correspondences=5, scaling_factor=scaling_factor)
    #
    #     # cProfile.run('test_classification(model_name=model_name, args_shape=args_shape,max_non_unique_correspondences=5, scaling_factor=scaling_factor)', 'profile_output')
    #     #
    #     # # To print the profile stats
    #     # p = pstats.Stats('profile_output')
    #     # p.sort_stats('cumulative').print_stats(20)
    #
    #     worst_losses, losses, final_thresh_list, num_of_inliers, point_distance_list, worst_point_losses, iter_2_ransac_convergence \
    #         = test_classification(model_name=model_name, args_shape=args_shape, max_non_unique_correspondences=5,
    #                           scaling_factor=scaling_factor, point_choice=0,
    #                           num_of_ransac_iter=1000, random_pairing=0)
    #     # view_stabiity(model_name=model_name, args_shape=args_shape, scaling_factor=scaling_factor)
    #     # worst_losses, losses, num_of_inliers, iter_2_ransac_convergence,shape_size_list, dist_from_orig, shortest_dist_list, avg_dist_list  \
    #     #     = findRotTrans(model_name=model_name, args_shape=args_shape,max_non_unique_correspondences=3, scaling_factor=scaling_factor)
    #     plot_losses(losses, num_of_inliers, filename=f'{scaling_factor}_'+"contrast_lpe6")
    #     # plot_losses(point_distance_list, num_of_inliers, filename=f'{scaling_factor}_'+"point_contrast_lpe6")
    #     # plot_distances(shape_size_list, shortest_dist_list, avg_dist_list, dist_from_orig, filename=f'{scaling_factor}_'+"dist_plot_lpe6")
    #     # plot_and_save_with_stats(iter_2_ransac_convergence, name=f'{scaling_factor}_'+"ransac_iter")
    #     #
    #     plotWorst(worst_losses, model_name=f'{scaling_factor}_'+"contrast_lpe6")
    #     plotWorst(worst_point_losses, model_name=f'{scaling_factor}_'+"point_contrast_lpe6")
    #
    #
