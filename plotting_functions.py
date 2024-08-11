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
import cProfile
import pstats
from scipy.spatial import cKDTree
from benchmark_modelnet import dcm2euler

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
    total_classes = len(point_clouds_1)
    # Filter out empty sub point clouds
    colors_pcl_1 = [clr for clr, pc in enumerate(point_clouds_1) if len(pc) > 0]
    point_clouds_1 = [pc for pc in point_clouds_1 if len(pc) > 0]

    colors_pcl_2 = [clr for clr, pc in enumerate(point_clouds_2) if len(pc) > 0]
    point_clouds_2 = [np.array(pc) for pc in point_clouds_2 if len(pc) > 0]

    if rotation is not None:
        center = np.mean(np.vstack(point_clouds_1), axis=0)
        point_clouds_1 = [np.matmul((pc - center), rotation.T) for pc in point_clouds_1]
        axis = np.argmin(np.max(np.vstack(point_clouds_1), axis=0))
        for pc in point_clouds_1:
            pc[:, axis] += 1.5

    # Define a broad range of colors
    # num_classes = len(set(colors_pcl_1) | set(colors_pcl_2))
    # cmap = plt.get_cmap('tab20', len(point_clouds_1))
    cmap = plt.get_cmap('plasma')
    jumps = (int)(cmap.N // (total_classes-1))
    colors = [cmap(i*jumps) for i in range(total_classes)]

    # Plot each point cloud from point_clouds_1 and point_clouds_2 with corresponding color
    for i, point_cloud1 in enumerate(point_clouds_1):
        fig.add_trace(go.Scatter3d(
            x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
            mode='markers', marker=dict(size=2, color=f'rgb({colors[colors_pcl_1[i]][0]*255},{colors[colors_pcl_1[i]][1]*255},{colors[colors_pcl_1[i]][2]*255})', opacity=0.5),
            name=f'PCL1: Class {colors_pcl_1[i]+1}, {len(point_cloud1)} points'
        ))
    for i, point_cloud2 in enumerate(point_clouds_2):
        fig.add_trace(go.Scatter3d(
            x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
            mode='markers', marker=dict(size=2, color=f'rgb({colors[colors_pcl_2[i]][0]*255},{colors[colors_pcl_2[i]][1]*255},{colors[colors_pcl_2[i]][2]*255})', opacity=0.5),
            name=f'PCL2: Class {colors_pcl_2[i]+1}, {len(point_cloud2)} points'
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
        pcl1 = [np.matmul((pcl - center), rotation.T) for pcl in pcl1]
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

def plot_4_point_clouds(point_cloud1, point_cloud2, point_cloud3, point_cloud4, rotation=None, translation=None, title=""):
  """
  Plot four point clouds in an interactive 3D plot with Plotly.

  Args:
      point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
      point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
      point_cloud3 (np.ndarray): Third point cloud of shape (41, 3)
      point_cloud4 (np.ndarray): Fourth point cloud of shape (41, 3)
  """
  if isinstance(point_cloud1, torch.Tensor):
      point_cloud1 = point_cloud1.cpu().detach().numpy()
  if isinstance(point_cloud2, torch.Tensor):
      point_cloud2 = point_cloud2.cpu().detach().numpy()
  if isinstance(point_cloud3, torch.Tensor):
      point_cloud3 = point_cloud3.cpu().detach().numpy()
  if isinstance(point_cloud4, torch.Tensor):
      point_cloud4 = point_cloud4.cpu().detach().numpy()
  fig = go.Figure()
  if rotation is not None:
      if isinstance(rotation, torch.Tensor):
          rotation = rotation.cpu().detach().numpy()
      point_cloud1 = np.matmul((point_cloud1), rotation.T) + translation
      point_cloud3 = np.matmul((point_cloud3), rotation.T) + translation
      axis = np.argmin(np.max(point_cloud1, axis=0))

      # point_cloud1[:, axis] = point_cloud1[:, axis] + 1.5
      # point_cloud3[:, axis] = point_cloud3[:, axis] + 1.5

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

def save_4_point_clouds(point_cloud1, point_cloud2, point_cloud3, point_cloud4, filename="plot.html", rotation=None, translation=None, title="", dist_thresh=5):
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
  colors = ['blue', 'orange', 'green', 'red']

  nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(point_cloud2);
  closest_neighbor_dist = nbrs.kneighbors(point_cloud2)[0][:, 1];
  mean_closest_neighbor_dist = np.mean(closest_neighbor_dist)

  if rotation is not None:
      point_cloud1 = np.matmul((point_cloud1), rotation.T) + translation.squeeze()
      point_cloud3 = np.matmul((point_cloud3), rotation.T) + translation.squeeze()
      axis = np.argmin(np.max(point_cloud1, axis=0))

      # point_cloud1[:, axis] = point_cloud1[:, axis] + 1.5
      # point_cloud3[:, axis] = point_cloud3[:, axis] + 1.5

  # Add traces for each point cloud with corresponding color
  for i, point_cloud in enumerate(
      [point_cloud1, point_cloud2, point_cloud3, point_cloud4]):
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers', marker=dict(size=2, color=colors[i]),
        name=f'Point Cloud {i+1}'
    ))

  amount_of_corr = len(point_cloud3)
  count = 0
  sum = 0
  for i in range(len(point_cloud3)):
      dist = np.linalg.norm(point_cloud3[i]-point_cloud4[i])
      if dist <= (dist_thresh * mean_closest_neighbor_dist):
          count+=1
          sum += dist
          fig.add_trace(go.Scatter3d(
              x=[point_cloud3[i, 0], point_cloud4[i, 0]],
              y=[point_cloud3[i, 1], point_cloud4[i, 1]],
              z=[point_cloud3[i, 2], point_cloud4[i, 2]],
              mode='lines', line=dict(color='black', width=2),
              showlegend=False
          ))

  avg_dist = (sum / count) if sum>0 else 0
  fig.update_layout(
      title=title+f'#corr={amount_of_corr}; #good={count}; dis_threshold={dist_thresh * mean_closest_neighbor_dist:.2f}; avg_dist={avg_dist:.2f}',
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
        margin=dict(r=20, l=20, b=20, t=60),  # Adjust top margin for title visibility
        title=dict(
            text=title,
            y=0.9,  # Position title within the top margin
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )

    # Save the figure as a png image
    fig.write_html(filename)

def plot_point_clouds(point_cloud1, point_cloud2, title=""):
    if isinstance(point_cloud1, torch.Tensor):
        point_cloud1 = point_cloud1.cpu().detach().numpy()
    if isinstance(point_cloud2, torch.Tensor):
        point_cloud2 = point_cloud2.cpu().detach().numpy()
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
        margin=dict(r=20, l=10, b=10, t=50),
        title=title
    )
    fig.show()


def plotWorst(worst_losses, model_name="", dir=r"./"):
    count = 0
    os.makedirs(dir, exist_ok=True)
    for (loss,worst_loss_variables) in worst_losses:
        noisy_pointcloud_1 = worst_loss_variables['noisy_pointcloud_1']
        noisy_pointcloud_2 = worst_loss_variables['noisy_pointcloud_2']
        chosen_points_1 = worst_loss_variables['chosen_points_1']
        chosen_points_2 = worst_loss_variables['chosen_points_2']
        rotation_matrix = worst_loss_variables['rotation_matrix']
        translation = worst_loss_variables['translation']
        best_rotation = worst_loss_variables['best_rotation']
        best_translation = worst_loss_variables['best_translation']
        All_pairs_1 = worst_loss_variables['All_pairs_1']
        All_pairs_2 = worst_loss_variables['All_pairs_2']
        pcl_id = worst_loss_variables['pcl_id']
        failed_ransac = worst_loss_variables['failed_ransac']
        failed_ransac_str = "failed_ransac_" if failed_ransac else ""
        if isinstance(noisy_pointcloud_1, torch.Tensor):
            noisy_pointcloud_1 = noisy_pointcloud_1.detach().cpu().numpy()
        if isinstance(noisy_pointcloud_2, torch.Tensor):
            noisy_pointcloud_2 = noisy_pointcloud_2.detach().cpu().numpy()
        if isinstance(chosen_points_1, torch.Tensor):
            chosen_points_1 = chosen_points_1.detach().cpu().numpy()
        if isinstance(chosen_points_2, torch.Tensor):
            chosen_points_2 = chosen_points_2.detach().cpu().numpy()
        if isinstance(rotation_matrix, torch.Tensor):
            rotation_matrix = rotation_matrix.detach().cpu().numpy()
        if isinstance(translation, torch.Tensor):
            translation = translation.detach().cpu().numpy()
        if isinstance(best_rotation, torch.Tensor):
            best_rotation = best_rotation.detach().cpu().numpy()
        if isinstance(best_translation, torch.Tensor):
            best_translation = best_translation.detach().cpu().numpy()
        if isinstance(All_pairs_1, torch.Tensor):
            All_pairs_1 = All_pairs_1.detach().cpu().numpy()
        if isinstance(All_pairs_2, torch.Tensor):
            All_pairs_2 = All_pairs_2.detach().cpu().numpy()
        if isinstance(pcl_id, torch.Tensor):
            pcl_id = pcl_id.detach().cpu().numpy()
        save_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, title="", filename=os.path.join(dir, model_name+f"_{failed_ransac_str}pcl_{pcl_id}_{loss:.3f}_orig_{count}_loss.html"))
        save_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, chosen_points_1, chosen_points_2, filename=os.path.join(dir, model_name+f"_{failed_ransac_str}pcl_{pcl_id}_{loss:.3f}_correspondence_{count}_loss.html"), rotation=rotation_matrix, translation=translation, dist_thresh=100)
        save_4_point_clouds(noisy_pointcloud_1, noisy_pointcloud_2, All_pairs_1, All_pairs_2, filename=os.path.join(dir, model_name+f"_{failed_ransac_str}pcl_{pcl_id}_{loss:.3f}_allpairs_{count}_loss.html"), rotation=rotation_matrix, translation=translation)

        transformed_points1 = np.matmul((noisy_pointcloud_1), best_rotation.T) + best_translation
        r_pred_euler_deg = dcm2euler(np.array([best_rotation]), seq='xyz')
        save_point_clouds(transformed_points1, noisy_pointcloud_2, title=f'rot: {r_pred_euler_deg}; trans: {best_translation}', filename=os.path.join(dir, model_name+f"_{failed_ransac_str}pcl_{pcl_id}_{loss:.3f}_{count}_loss.html"))
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
def plot_losses(losses, inliers, filename="loss_plot.png", dir=r"./"):
    """
    Plots the given list of losses and corresponding number of inliers,
    and saves the plot as an image
    """
    os.makedirs(dir, exist_ok=True)
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
    plt.savefig(os.path.join(dir, filename))

    # You can optionally remove the following line if you don't want to display the plot as well
    # plt.show()



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
def plot_point_cloud_with_colors_by_dist_2_pcls(point_cloud1, point_cloud2, embedding1, embedding2, chosen_indices):
    # Generate random index to choose a point from point_cloud1
    random_point = point_cloud2[chosen_indices[1]]
    random_embedding = embedding1[chosen_indices[0]]

    # Calculate distances from the random embedding to all embeddings in embedding2
    distances = np.linalg.norm(embedding2 - random_embedding, axis=1)

    # Find indices of the 20 closest points
    closest_indices = np.argsort(distances)[:20]

    # Define color scale based on distances
    max_distance = distances.max()
    min_distance = distances.min()
    colors = [(d - min_distance) / (max_distance - min_distance) for d in distances]
    colors = np.array(colors)
    zero_indices = (colors == 0)
    non_zero_indices = (colors != 0)
    colors[non_zero_indices] = 1 / np.array(colors[non_zero_indices])
    colors[zero_indices] = np.inf
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


def plot_metrics(dictionary, dir=r"./",run_name=""):
    """
    Function to plot each array in a dictionary as a separate plot.

    Parameters:
    - dictionary: Dictionary where each value is a 1D numpy array.
    """
    os.makedirs(dir, exist_ok=True)
    # Iterate through each key-value pair in the dictionary
    for key, array in dictionary.items():
        # Calculate mean and median of the array
        mean_val = np.mean(array)
        median_val = np.median(array)

        # Plot the array
        plt.figure()
        plt.plot(array)

        # Set title with key, mean, and median
        plt.title(f"Array: {key}\nMean: {mean_val:.2f}, Median: {median_val:.2f}")

        # Set labels
        plt.xlabel('Index')
        plt.ylabel('Value')

        # Show plot
        plt.grid(True)
        plt.savefig(os.path.join(dir, run_name + f'{key}_Mean_{mean_val:.2f}_Median_{median_val:.2f}_plot.png'))
        # plt.show()