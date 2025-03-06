import numpy as np
import torch
from plotting_functions import *
from ransac import *
from threedmatch import *
from indoor import *
import platform

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


def visualize_classified_point_cloud_grad(pcl, model_name, args_shape, scaling_factor="1",
                                     zoom_point_indices=None, zoom_k=21, fig_size=(12, 10)):
    """
    Visualize point cloud with classification-based coloring and zoomed patches for each class
    Creates separate plots for the main point cloud and each zoomed view

    Args:
        pcl: Input point cloud (numpy array of shape Nx3)
        model_name: Name of the shape classifier model
        args_shape: Arguments for the shape classifier
        scaling_factor: Scaling factor for normalization
        zoom_point_indices: Dict of indices of points to zoom in on for each class (if None, selects highest confidence points)
        zoom_k: Number of nearest neighbors to show in zoomed view
        fig_size: Figure size (width, height) in inches
    """
    # Use original point cloud
    noisy_pcl = pcl

    # Get classification outputs
    colors, scaling_fac = classifyPoints(model_name=model_name,
                                         pcl_src=noisy_pcl,
                                         pcl_interest=noisy_pcl,
                                         args_shape=args_shape,
                                         scaling_factor=scaling_factor)

    # Get raw classification scores including the edge class (index 4)
    all_class_scores = colors.detach().cpu().numpy().squeeze()
    class_scores = all_class_scores[:, :4]  # First 4 classes (non-edge)
    edge_scores = all_class_scores[:, 4]  # Edge class scores

    # Get the predicted class for each point from the first 4 classes
    predicted_class = np.argmax(class_scores, axis=1)

    # Create a mask for non-edge points (where edge score is not the highest)
    non_edge_mask = np.zeros(len(predicted_class), dtype=bool)

    # For each point, check if any of the first 4 class scores is higher than the edge score
    for i in range(len(predicted_class)):
        max_non_edge_score = np.max(class_scores[i])
        if max_non_edge_score > edge_scores[i]:
            non_edge_mask[i] = True

    # Normalize scores within each class for coloring
    normalized_scores = np.zeros_like(class_scores)
    for i in range(4):
        class_mask = predicted_class == i
        if np.any(class_mask):
            class_values = class_scores[class_mask, i]
            min_val, max_val = class_values.min(), class_values.max()
            if max_val > min_val:
                normalized_scores[class_mask, i] = (class_scores[class_mask, i] - min_val) / (max_val - min_val)
            else:
                normalized_scores[class_mask, i] = 0.5  # Default to mid-scale if all values are the same

    # Create color maps for each class
    class_colors = ['blue', 'green', 'red', 'grey']
    class_colors = ['grey', 'red', 'green', 'blue']
    color_values = []

    for i in range(len(noisy_pcl)):
        cls = predicted_class[i]
        intensity = normalized_scores[i, cls]
        intensity =0
        # Get base color
        base_color = class_colors[cls]

        # Convert to RGB and adjust intensity (darker = lower confidence, brighter = higher confidence)
        if base_color == 'blue':
            color = f'rgba(0, 0, {int(50 + 205 * intensity)}, 0.8)'
        elif base_color == 'green':
            color = f'rgba(0, {int(50 + 205 * intensity)}, 0, 0.8)'
        elif base_color == 'red':
            color = f'rgba({int(50 + 205 * intensity)}, 0, 0, 0.8)'
        else:  # grey
            grey_val = int(50 + 150 * intensity)
            color = f'rgba({grey_val}, {grey_val}, {grey_val}, 0.8)'

        color_values.append(color)

    # Select points to zoom in for each class
    class_names = ['Plane', 'Peak/Pit', 'Valley/Ridge', 'Saddle']
    if zoom_point_indices is None:
        zoom_point_indices = {}
        for class_idx in range(4):
            # Combine class mask with non-edge mask to get only non-edge points of this class
            class_mask = (predicted_class == class_idx) & non_edge_mask
            if np.any(class_mask):
                # Get the point with highest classification confidence for this class
                class_points_indices = np.where(class_mask)[0]
                class_confidence = class_scores[class_points_indices, class_idx]
                highest_conf_idx = class_points_indices[np.argmax(class_confidence)]
                zoom_point_indices[class_idx] = highest_conf_idx
            else:
                # Fallback if no non-edge points are available for this class
                print(f"Warning: No non-edge points found for class {class_names[class_idx]}.")
                # Try to find any point of this class, even if it's an edge
                original_class_mask = predicted_class == class_idx
                if np.any(original_class_mask):
                    class_confidence = class_scores[original_class_mask, class_idx]
                    highest_conf_idx = np.where(original_class_mask)[0][np.argmax(class_confidence)]
                    zoom_point_indices[class_idx] = highest_conf_idx

    # Create a list to store all figures
    figures = []

    # Create main point cloud figure
    main_fig = go.Figure()

    # Add main point cloud
    main_fig.add_trace(
        go.Scatter3d(
            x=noisy_pcl[:, 0],
            y=noisy_pcl[:, 1],
            z=noisy_pcl[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
            ),
            showlegend=False
        )
    )

    # Add zoom points to main plot
    for class_idx, point_idx in zoom_point_indices.items():
        main_fig.add_trace(
            go.Scatter3d(
                x=[noisy_pcl[point_idx, 0]],
                y=[noisy_pcl[point_idx, 1]],
                z=[noisy_pcl[point_idx, 2]],
                mode='markers',
                marker=dict(
                    size=6,
                    color=class_colors[class_idx],
                    symbol='circle',
                    line=dict(
                        color='black',
                        width=1
                    )
                ),
                showlegend=False
            )
        )

    # Update main figure layout
    main_fig.update_layout(
        title="Surface Classification",
        width=fig_size[0] * 100,
        height=fig_size[1] * 100,
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=False
    )

    figures.append(main_fig)

    # # Create separate zoom figures for each class
    # for class_idx, point_idx in zoom_point_indices.items():
    #     zoom_fig = go.Figure()
    #
    #     # Get nearest neighbors for zoom
    #     zoom_patch_indices = get_k_nearest_neighbors_indices(noisy_pcl, noisy_pcl[point_idx], k=zoom_k)
    #     zoom_patch = noisy_pcl[zoom_patch_indices]
    #     zoom_colors = [color_values[i] for i in zoom_patch_indices]
    #
    #     # Add zoomed point cloud
    #     zoom_fig.add_trace(
    #         go.Scatter3d(
    #             x=zoom_patch[:, 0],
    #             y=zoom_patch[:, 1],
    #             z=zoom_patch[:, 2],
    #             mode='markers',
    #             marker=dict(
    #                 size=4,
    #                 color=zoom_colors,
    #             ),
    #             showlegend=False
    #         )
    #     )
    #
    #     # Highlight center point in zoom
    #     zoom_fig.add_trace(
    #         go.Scatter3d(
    #             x=[noisy_pcl[point_idx, 0]],
    #             y=[noisy_pcl[point_idx, 1]],
    #             z=[noisy_pcl[point_idx, 2]],
    #             mode='markers',
    #             marker=dict(
    #                 size=8,
    #                 color=class_colors[class_idx],
    #                 symbol='circle',
    #                 line=dict(
    #                     color='black',
    #                     width=1
    #                 )
    #             ),
    #             showlegend=False
    #         )
    #     )
    #
    #     # Update zoom figure layout
    #     zoom_fig.update_layout(
    #         title=f"{class_names[class_idx]} Zoom",
    #         width=fig_size[0] * 70,
    #         height=fig_size[1] * 70,
    #         scene=dict(
    #             aspectmode='data',
    #             xaxis_title='X',
    #             yaxis_title='Y',
    #             zaxis_title='Z'
    #         ),
    #         showlegend=False
    #     )
    #
    #     figures.append(zoom_fig)

    return figures


def visualize_classified_point_cloud(pcl, model_name, args_shape, scaling_factor="1",
                                     zoom_point_indices=None, zoom_k=21, fig_size=(15, 12)):
    """
    Visualize classified point cloud with enhanced colors and improved visual clarity.

    Parameters:
    -----------
    pcl : numpy.ndarray
        Point cloud data with shape (n_points, 3)
    model_name : str
        Name of the model used for classification
    args_shape : dict
        Shape arguments for the model
    scaling_factor : str, optional
        Scaling factor for the point cloud, default is "1"
    zoom_point_indices : list, optional
        Indices of points to zoom in on
    zoom_k : int, optional
        Number of nearest neighbors to include in zoom view
    fig_size : tuple, optional
        Figure size (width, height) in inches, default is (15, 12)

    Returns:
    --------
    list
        List of plotly figures
    """
    # Use original point cloud
    noisy_pcl = pcl

    # Get classification outputs
    colors, scaling_fac = classifyPoints(model_name=model_name,
                                         pcl_src=noisy_pcl,
                                         pcl_interest=noisy_pcl,
                                         args_shape=args_shape,
                                         scaling_factor=scaling_factor)

    # Get raw classification scores including the edge class (index 4)
    all_class_scores = colors.detach().cpu().numpy().squeeze()
    class_scores = all_class_scores[:, :4]  # First 4 classes (non-edge)
    edge_scores = all_class_scores[:, 4]  # Edge class scores

    # Get the predicted class for each point from the first 4 classes
    predicted_class = np.argmax(class_scores, axis=1)

    # Create a mask for non-edge points (where edge score is not the highest)
    non_edge_mask = np.zeros(len(predicted_class), dtype=bool)

    # For each point, check if any of the first 4 class scores is higher than the edge score
    for i in range(len(predicted_class)):
        max_non_edge_score = np.max(class_scores[i])
        if max_non_edge_score > edge_scores[i]:
            non_edge_mask[i] = True

    # Define maximally discriminative, perceptually optimized color palette
    # Selected using advanced color theory principles for maximum visual differentiation
    # These colors maintain distinctiveness under various lighting conditions and for colorblind viewers
    fixed_class_colors = [
        '#FF0000',  # Class 0: Pure Red (RGB: 255,0,0)
        '#00B7FF',  # Class 1: Azure Blue (RGB: 0,183,255)
        '#FFFF00',  # Class 2: Pure Yellow (RGB: 255,255,0)
        '#7F00FF'  # Class 3: Violet (RGB: 127,0,255)
    ]

    # Class names for legend (customize these based on your application)
    class_names = [
        "Surface Type A",
        "Surface Type B",
        "Surface Type C",
        "Surface Type D"
    ]

    # Assign colors based on predicted class
    color_values = []
    for i in range(len(noisy_pcl)):
        cls = predicted_class[i]
        color = fixed_class_colors[cls]
        color_values.append(color)

    # Create a list to store all figures
    figures = []

    # Create main point cloud figure
    main_fig = go.Figure()

    # Add traces for each class to enable proper legend
    for class_idx in range(4):
        mask = predicted_class == class_idx
        if np.any(mask):  # Only add class if points exist
            main_fig.add_trace(
                go.Scatter3d(
                    x=noisy_pcl[mask, 0],
                    y=noisy_pcl[mask, 1],
                    z=noisy_pcl[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=fixed_class_colors[class_idx],
                        opacity=0.9,
                        symbol='circle',
                        line=dict(
                            width=0.5,
                            color='rgba(0,0,0,0.3)'
                        ),
                    ),
                    name=class_names[class_idx],
                )
            )

    # Update main figure layout with enhanced visual properties
    main_fig.update_layout(
        title=dict(
            text="Point Cloud Surface Classification",
            font=dict(size=22, color='#333333')
        ),
        width=fig_size[0] * 100,
        height=fig_size[1] * 100,
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                title=dict(text='X', font=dict(size=16)),
                gridcolor='#DDDDDD',
                backgroundcolor='rgba(255,255,255,0.95)',
                showbackground=True
            ),
            yaxis=dict(
                title=dict(text='Y', font=dict(size=16)),
                gridcolor='#DDDDDD',
                backgroundcolor='rgba(255,255,255,0.95)',
                showbackground=True
            ),
            zaxis=dict(
                title=dict(text='Z', font=dict(size=16)),
                gridcolor='#DDDDDD',
                backgroundcolor='rgba(255,255,255,0.95)',
                showbackground=True
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='#CCCCCC',
            borderwidth=1
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Add zoom view if indices are provided
    if zoom_point_indices is not None and len(zoom_point_indices) > 0:
        # For each point to zoom in on, create a separate figure
        for idx in zoom_point_indices:
            # Find k nearest neighbors
            point = noisy_pcl[idx]
            distances = np.sum((noisy_pcl - point) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[:zoom_k]

            zoom_fig = go.Figure()

            # Add center point with distinctive appearance
            zoom_fig.add_trace(
                go.Scatter3d(
                    x=[noisy_pcl[idx, 0]],
                    y=[noisy_pcl[idx, 1]],
                    z=[noisy_pcl[idx, 2]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond',
                        line=dict(width=1, color='black')
                    ),
                    name='Focus Point'
                )
            )

            # Add neighboring points by class
            for class_idx in range(4):
                mask = (predicted_class[nearest_indices] == class_idx) & (nearest_indices != idx)
                if np.any(mask):
                    zoom_fig.add_trace(
                        go.Scatter3d(
                            x=noisy_pcl[nearest_indices[mask], 0],
                            y=noisy_pcl[nearest_indices[mask], 1],
                            z=noisy_pcl[nearest_indices[mask], 2],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=fixed_class_colors[class_idx],
                                opacity=0.9,
                            ),
                            name=class_names[class_idx]
                        )
                    )

            # Update zoom figure layout
            zoom_fig.update_layout(
                title=f"Detailed View of Point {idx}",
                width=fig_size[0] * 60,
                height=fig_size[1] * 60,
                scene=dict(
                    aspectmode='data',
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                showlegend=True
            )

            figures.append(zoom_fig)

    # Add main figure at the beginning of the list
    figures.insert(0, main_fig)

    return figures
def get_k_nearest_neighbors_indices(pcl, query_point, k=21):
    """
    Get indices of k nearest neighbors for a query point

    Args:
        pcl: Point cloud (Nx3 numpy array)
        query_point: Query point (3D vector)
        k: Number of neighbors to return

    Returns:
        Indices of k nearest points including the query point
    """
    # Calculate distances
    diff = pcl - query_point
    distances = np.sum(diff * diff, axis=1)

    # Get indices of k nearest points
    indices = np.argsort(distances)[:k]

    return indices


# Function to create a figure with colorbar legends for a paper
def create_paper_figure(pcl, model_name, args_shape, scaling_factor="1", zoom_point_indices=None):
    """
    Create a publication-quality figure with proper colorbars for each class
    and zooms for each class's highest confidence point

    Returns:
        matplotlib figure for paper publication
    """
    # Process point cloud similar to the interactive version
    noisy_pcl = pcl

    # Get classification outputs
    colors, scaling_fac = classifyPoints(model_name=model_name,
                                         pcl_src=noisy_pcl,
                                         pcl_interest=noisy_pcl,
                                         args_shape=args_shape,
                                         scaling_factor=scaling_factor)

    # Get raw classification scores (first 4 classes)
    class_scores = colors.detach().cpu().numpy().squeeze()[:, :4]

    # Get the predicted class for each point
    predicted_class = np.argmax(class_scores, axis=1)

    # Choose zoom points if not provided - one for each class with highest confidence
    class_names = ['Plane', 'Peak/Pit', 'Valley/Ridge', 'Saddle']
    if zoom_point_indices is None:
        zoom_point_indices = {}
        for class_idx in range(4):
            class_mask = predicted_class == class_idx
            if np.any(class_mask):
                # Get the point with highest classification confidence for this class
                class_confidence = class_scores[class_mask, class_idx]
                highest_conf_idx = np.where(class_mask)[0][np.argmax(class_confidence)]
                zoom_point_indices[class_idx] = highest_conf_idx

    # Create figure with a grid layout
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)

    # Define grid based on number of classes
    num_classes = len(zoom_point_indices)
    if num_classes <= 2:
        gs = fig.add_gridspec(2, 1 + num_classes)
        main_slice = slice(0, 2)
        zoom_row_start = 0
    else:
        gs = fig.add_gridspec(3, 3)
        main_slice = slice(0, 3)
        zoom_row_start = 1

    # Main 3D plot
    ax_main = fig.add_subplot(gs[main_slice, 0:2], projection='3d')

    # Define class colors with proper colormaps
    colormaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]

    # Plot each class separately
    for cls_idx in range(4):
        mask = predicted_class == cls_idx
        if np.any(mask):
            cls_points = noisy_pcl[mask]
            cls_values = class_scores[mask, cls_idx]

            # Normalize values
            if cls_values.max() > cls_values.min():
                norm_values = (cls_values - cls_values.min()) / (cls_values.max() - cls_values.min())
            else:
                norm_values = np.ones_like(cls_values) * 0.5

            colors = colormaps[cls_idx](norm_values)

            ax_main.scatter(
                cls_points[:, 0], cls_points[:, 1], cls_points[:, 2],
                c=colors, s=2, alpha=0.7
            )

    # Mark zoom points in main plot
    for cls_idx, point_idx in zoom_point_indices.items():
        ax_main.scatter(
            noisy_pcl[point_idx, 0],
            noisy_pcl[point_idx, 1],
            noisy_pcl[point_idx, 2],
            color=colormaps[cls_idx](0.9), s=30, edgecolor='black'
        )

    ax_main.set_title('Surface Classification')
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')
    ax_main.set_zlabel('Z')

    # Zoom plots - one for each class
    zoom_k = 21
    zoom_col = 2
    zoom_row = zoom_row_start

    for cls_idx, point_idx in zoom_point_indices.items():
        # Get zoom patch
        zoom_patch_indices = get_k_nearest_neighbors_indices(noisy_pcl, noisy_pcl[point_idx], k=zoom_k)
        zoom_patch = noisy_pcl[zoom_patch_indices]
        zoom_classes = predicted_class[zoom_patch_indices]

        # Create subplot
        ax_zoom = fig.add_subplot(gs[zoom_row, zoom_col], projection='3d')

        # Plot zoom points with class colors
        for subcls_idx in range(4):
            zoom_mask = zoom_classes == subcls_idx
            if np.any(zoom_mask):
                z_points = zoom_patch[zoom_mask]
                z_values = class_scores[zoom_patch_indices[zoom_mask], subcls_idx]

                # Normalize values
                if z_values.max() > z_values.min():
                    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())
                else:
                    z_norm = np.ones_like(z_values) * 0.5

                z_colors = colormaps[subcls_idx](z_norm)

                ax_zoom.scatter(
                    z_points[:, 0], z_points[:, 1], z_points[:, 2],
                    c=z_colors, s=10, alpha=0.8
                )

        # Mark zoom center point
        ax_zoom.scatter(
            noisy_pcl[point_idx, 0],
            noisy_pcl[point_idx, 1],
            noisy_pcl[point_idx, 2],
            color=colormaps[cls_idx](0.9), s=40, edgecolor='black'
        )

        ax_zoom.set_title(f'{class_names[cls_idx]} Zoom')

        # Set zoom view limits centered at the zoom point
        zoom_radius = np.max(np.linalg.norm(zoom_patch - noisy_pcl[point_idx], axis=1)) * 1.2
        ax_zoom.set_xlim(noisy_pcl[point_idx, 0] - zoom_radius, noisy_pcl[point_idx, 0] + zoom_radius)
        ax_zoom.set_ylim(noisy_pcl[point_idx, 1] - zoom_radius, noisy_pcl[point_idx, 1] + zoom_radius)
        ax_zoom.set_zlim(noisy_pcl[point_idx, 2] - zoom_radius, noisy_pcl[point_idx, 2] + zoom_radius)

        # Move to next position
        zoom_row += 1
        if zoom_row >= gs.get_geometry()[0]:
            zoom_row = zoom_row_start
            zoom_col += 1

    # Color scale legends
    gs_cb = gs[-1, 0:2].subgridspec(1, 4)
    for i, (cmap, class_name) in enumerate(zip(colormaps, class_names)):
        cax = fig.add_subplot(gs_cb[0, i])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb = mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, orientation='horizontal',
            label=f'{class_name} Confidence'
        )

    return fig


def claude_plotting(model_name=None, args_shape=None, scaling_factor=None, rgb=False, add_noise=True):
    # pcls, label = load_data()
    pcl_person = np.load("persons_pcls.npy")
    pcl_guitar = np.load("guitar_pcls.npy")
    # pcl_airplane = np.load("airplane_pcls.npy")
    # pcl_bath = np.load("bath_pcls.npy")
    # pcl = pcls[162][:]
    shapes = [47, 86, 162,174,176,179]
    shapes = [86, 162,174,179]
    shapes = [86,174,179]
    for idx in shapes:
        pcl = np.load(f'{idx}.npy')
        figures = visualize_classified_point_cloud(pcl, model_name=model_name, args_shape=args_shape)
        for fig in figures:
            fig.show()

    # humans =[3,12,17,19]
    humans =[12]
    for idx in humans:
        pcl = pcl_person[idx, :, :]
        figures = visualize_classified_point_cloud(pcl, model_name=model_name, args_shape=args_shape)
        for fig in figures:
            fig.show()
    # guitars =[5,11]
    # for idx in guitars:
    #     pcl = pcl_guitar[idx, :, :]
    #     figures = visualize_classified_point_cloud(pcl, model_name=model_name, args_shape=args_shape)
    #     for fig in figures:
    #         fig.show()

def rotate_point_cloud(pcl, rotation_angles=(0, 0, 0)):
    """
    Rotate a 3D point cloud using Euler angles.

    Args:
        pcl: Input point cloud (numpy array of shape Nx3)
        rotation_angles: Tuple of (rx, ry, rz) rotation angles in degrees
                         to rotate around the x, y, and z axes respectively

    Returns:
        rotated_pcl: Rotated point cloud (numpy array of shape Nx3)
    """
    # Ensure the point cloud is in float format
    pcl = np.asarray(pcl, dtype=np.float32)
    # Convert angles from degrees to radians
    rx, ry, rz = np.radians(rotation_angles)

    # Rotation matrix around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    # Rotation matrix around y-axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Rotation matrix around z-axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix (applying rotations in order Z, Y, X)
    # This is the extrinsic rotation sequence, which is most commonly used
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Apply rotation to the point cloud (each point is a row in the array)
    rotated_pcl = np.dot(pcl, R.T)

    # Ensure the returned point cloud is of float type
    rotated_pcl = rotated_pcl.astype(np.float32)

    return rotated_pcl


# Example usage:
# rotated_cloud = rotate_point_cloud(point_cloud, (30, 45, 10))
# This rotates the point cloud 30° around x-axis, 45° around y-axis, and 10° around z-axis

def visualize_classified_point_cloud_static(pcl, model_name, args_shape, scaling_factor="1",
                                            zoom_point_indices=None, zoom_k=21, fig_size=(12, 10),
                                            save_path=None):
    """
    Create 5 static plots: one main plot of the colored point cloud,
    and 4 separate plots for each zoomed-in region

    Args:
        pcl: Input point cloud (numpy array of shape Nx3)
        model_name: Name of the shape classifier model
        args_shape: Arguments for the shape classifier
        scaling_factor: Scaling factor for normalization
        zoom_point_indices: Dict of indices of points to zoom in on for each class (if None, selects highest confidence points)
        zoom_k: Number of nearest neighbors to show in zoomed view
        fig_size: Figure size (width, height) in inches
        save_path: Path to save figures (if None, figures are displayed but not saved)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl

    # Use the noisy point cloud directly
    # noisy_pcl = rotate_point_cloud(pcl, (30, 45, 10))
    noisy_pcl = rotate_point_cloud(pcl, (0, 0, 0))
    # Get classification outputs
    colors, scaling_fac = classifyPoints(model_name=model_name,
                                         pcl_src=noisy_pcl,
                                         pcl_interest=noisy_pcl,
                                         args_shape=args_shape,
                                         scaling_factor=scaling_factor)

    # Get raw classification scores (first 4 classes - adjust if needed)
    class_scores = colors.detach().cpu().numpy().squeeze()[:, :4]

    # Get the predicted class for each point
    predicted_class = np.argmax(class_scores, axis=1)

    # Normalize scores within each class for coloring
    normalized_scores = np.zeros_like(class_scores)
    for i in range(4):
        class_mask = predicted_class == i
        if np.any(class_mask):
            class_values = class_scores[class_mask, i]
            min_val, max_val = class_values.min(), class_values.max()
            if max_val > min_val:
                normalized_scores[class_mask, i] = (class_scores[class_mask, i] - min_val) / (max_val - min_val)
            else:
                normalized_scores[class_mask, i] = 0.5  # Default to mid-scale if all values are the same

    # Define class colors and names
    class_names = ['Plane', 'Peak/Pit', 'Valley/Ridge', 'Saddle']
    colormaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Reds, plt.cm.Greys]

    # Select points to zoom in for each class
    if zoom_point_indices is None:
        zoom_point_indices = {}
        for class_idx in range(4):
            class_mask = predicted_class == class_idx
            if np.any(class_mask):
                # Get the point with highest classification confidence for this class
                class_confidence = class_scores[class_mask, class_idx]
                highest_conf_idx = np.where(class_mask)[0][np.argmax(class_confidence)]
                zoom_point_indices[class_idx] = highest_conf_idx

    # Create the main figure
    fig_main = plt.figure(figsize=fig_size)
    ax_main = fig_main.add_subplot(111, projection='3d')


    # Plot each class separately in the main plot
    for cls_idx in range(4):
        mask = predicted_class == cls_idx
        if np.any(mask):
            cls_points = noisy_pcl[mask]
            cls_values = normalized_scores[mask, cls_idx]

            # Use the appropriate colormap
            colors = colormaps[cls_idx](cls_values)

            ax_main.scatter(
                cls_points[:, 0], cls_points[:, 1], cls_points[:, 2],
                c=colors, s=10, alpha=0.7, label=class_names[cls_idx]
            )

    # Mark zoom points in main plot
    for cls_idx, point_idx in zoom_point_indices.items():
        # Glow effect (larger, semi-transparent point)
        ax_main.scatter(
            noisy_pcl[point_idx, 0],
            noisy_pcl[point_idx, 1],
            noisy_pcl[point_idx, 2],
            color=colormaps[cls_idx](0.9),
            s=300, alpha=0.3,  # Larger, transparent point for glow
            marker='o'
        )

        # Actual zoom point (sharp and visible)
        ax_main.scatter(
            noisy_pcl[point_idx, 0],
            noisy_pcl[point_idx, 1],
            noisy_pcl[point_idx, 2],
            color=colormaps[cls_idx](0.9),
            s=80, edgecolor='black', linewidth=1.5,
            label=f'{class_names[cls_idx]} Zoom Point'
        )

    ax_main.set_xlim([-1, 1])
    ax_main.set_ylim([-1, 1])
    ax_main.set_zlim([-1, 1])
    # ax_main.grid(False)  # Remove grid from the main plot
    # ax_main.set_xticks([])
    # ax_main.set_yticks([])
    # ax_main.set_zticks([])
    # ax_main.set_xticklabels([])
    # ax_main.set_yticklabels([])
    # ax_main.set_zticklabels([])
    # ax_main.axis('off')
    ax_main.set_title('Surface Classification')
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')
    ax_main.set_zlabel('Z')
    ax_main.legend()
    # ax_main.view_init(elev=0, azim=0)

    # Save or show the main figure
    if save_path:
        fig_main.savefig(f"{save_path}/main_plot.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    # Create separate zoom plots
    for cls_idx, point_idx in zoom_point_indices.items():
        # Get zoom patch
        zoom_patch_indices = get_k_nearest_neighbors_indices(noisy_pcl, noisy_pcl[point_idx], k=zoom_k)
        zoom_patch = noisy_pcl[zoom_patch_indices]
        zoom_classes = predicted_class[zoom_patch_indices]

        # Create new figure for this zoom
        fig_zoom = plt.figure(figsize=(10, 8))
        ax_zoom = fig_zoom.add_subplot(111, projection='3d')
        ax_zoom.grid(False)

        # Plot all zoom points with their respective class colors
        for subcls_idx in range(4):
            zoom_mask = zoom_classes == subcls_idx
            if np.any(zoom_mask):
                z_points = zoom_patch[zoom_mask]
                z_values = normalized_scores[zoom_patch_indices[zoom_mask], subcls_idx]

                z_colors = colormaps[subcls_idx](z_values)

                ax_zoom.scatter(
                    z_points[:, 0], z_points[:, 1], z_points[:, 2],
                    c=z_colors, s=20, alpha=0.8
                )

        # Mark zoom center point
        ax_zoom.scatter(
            noisy_pcl[point_idx, 0],
            noisy_pcl[point_idx, 1],
            noisy_pcl[point_idx, 2],
            color=colormaps[cls_idx](0.9), s=80, edgecolor='black'
        )

        ax_zoom.set_title(f'{class_names[cls_idx]} Zoom')
        ax_zoom.set_xlabel('X')
        ax_zoom.set_ylabel('Y')
        ax_zoom.set_zlabel('Z')
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax_zoom.set_zticks([])
        ax_zoom.set_xticklabels([])
        ax_zoom.set_yticklabels([])
        ax_zoom.set_zticklabels([])
        ax_zoom.axis('off')


        # Save or show the zoom figure
        if save_path:
            # Replace slashes with underscores for safe filenames
            safe_class_name = class_names[cls_idx].replace('/', '_')
            fig_zoom.savefig(f"{save_path}/{safe_class_name}_zoom.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # Create a figure with just the color legends
    fig_legend = plt.figure(figsize=(12, 3))
    axes = []

    for i, (cmap, class_name) in enumerate(zip(colormaps, class_names)):
        ax = fig_legend.add_subplot(1, 4, i + 1)
        axes.append(ax)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb = mpl.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation='horizontal',
            label=f'{class_name} Confidence'
        )

    fig_legend.tight_layout()

    # Save or show the legend figure
    if save_path:
        fig_legend.savefig(f"{save_path}/color_legend.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig_main


def claude_plotting_static(model_name=None, args_shape=None, scaling_factor=None, save_path="./plots"):
    """
    Modified plotting function to create static plots

    Args:
        model_name: Name of the shape classifier model
        args_shape: Arguments for the shape classifier
        scaling_factor: Scaling factor for normalization
        save_path: Directory to save plots (will be created if it doesn't exist)
    """
    import os

    # Create save directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    pcls, label = load_data()
    pcl = pcls[162][:]

    # Create static plots
    fig = visualize_classified_point_cloud_static(
        pcl,
        model_name=model_name,
        args_shape=args_shape,
        scaling_factor=scaling_factor,
        save_path=save_path
    )

    print(f"Plots saved to {save_path}")
def load_data(partition='test', divide_data=1):
    DATA_DIR = r'C:\\Users\\Owner\\PycharmProjects\\curvTrans\\bbsWithShapes\\data'
    if platform.system() != "Windows":
        DATA_DIR = r'/content/curvTrans/bbsWithShapes/data'
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
def checkSyntheticData():
    args = configArgsPCT()
    args.std_dev = 0.05
    args.rotate_data = 1
    args.std_dev = 0
    args.batch_size = 40000
    max_list_x = []
    min_list_x = []
    max_list_y = []
    min_list_y = []
    max_list_z = []
    min_list_z = []
    std_list_x = []
    std_list_y = []
    std_list_z = []
    diameter_list_x = []
    diameter_list_y = []
    diameter_list_z = []
    full_diameter_from_center_mean_list = []
    full_diameter_from_center_median_list = []
    full_diameter_from_center_max_list = []
    full_diameter_from_center_min_list = []
    full_diameter_from_center_09_list = []
    density_list = []
    avg_dist_list = []
    # train_dataset = BasicPointCloudDataset(file_path="train_surfaces_40_stronger_boundaries.h5", args=args)
    train_dataset = BasicPointCloudDataset(file_path="train_surfaces_with_corners_very_mild_curve.h5", args=args)
    # train_dataset = BasicPointCloudDataset(file_path="train_surfaces_with_corners_very_mild_1_5__2.h5", args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    # test_dataset = BasicPointCloudDataset(file_path="test_surfaces_40_stronger_boundaries.h5", args=args)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # with tqdm(test_dataloader) as tqdm_bar:
    count = 0
    with tqdm(train_dataloader) as tqdm_bar:
        for batch in tqdm_bar:
            # if count == 1:
            #     break
            count +=1
            pcl = batch['point_cloud']
            # plot_point_clouds(pcl[2], pcl[2],
            #                   f'H: {batch["info"]["H"][2].item():.2f}, K: {batch["info"]["K"][2].item():.2f}, Class: {int(batch["info"]["class"][2].item())} ')
            # pcl = pcl / 37
            # Calculate the mean along the point dimension (axis 1)
            mean = pcl.mean(axis=1, keepdim=True)

            # Center the data by subtracting the mean
            centered_pointcloud = pcl - mean
            std = centered_pointcloud.std(axis=1, keepdim=True)
            std_list_x.append(torch.mean(std, dim=0)[0][0].item())
            std_list_y.append(torch.mean(std, dim=0)[0][1].item())
            std_list_z.append(torch.mean(std, dim=0)[0][2].item())

            diameter_list_x.append((torch.mean((torch.max(pcl[:,:,0], dim=1).values)-(torch.min(pcl[:,:,0], dim=1).values))).item())
            diameter_list_y.append((torch.mean((torch.max(pcl[:,:,1], dim=1).values)-(torch.min(pcl[:,:,1], dim=1).values))).item())
            diameter_list_z .append((torch.mean((torch.max(pcl[:,:,2], dim=1).values)-(torch.min(pcl[:,:,2], dim=1).values))).item())

            min_coords = torch.min(pcl, dim=1)[0]
            max_coords = torch.max(pcl, dim=1)[0]

            # Compute the volume of the bounding box for each point cloud
            bounding_box_volumes = torch.prod(max_coords - min_coords, dim=1)

            # Number of points in each point cloud (assuming all point clouds have the same number of points)
            num_points = pcl.shape[1]

            # Calculate the density for each point cloud
            densities = num_points / bounding_box_volumes
            avg_density = torch.mean(densities)
            density_list.append(avg_density)

            pairwise_distances = torch.cdist(pcl, pcl, p=2)
            sum_distances = torch.sum(pairwise_distances, dim=(1, 2))
            num_pairs = num_points * (num_points - 1)
            avg_distances = sum_distances / num_pairs
            avg_dist_list.append(torch.mean(avg_distances).item())

            diam = (((torch.max(pairwise_distances[:, 0, :], dim=1))[0]))
            print(torch.mean(diam))
            print(torch.std(diam))
            print(f'KKKKK')
            full_diameter_from_center_mean_list.append((torch.mean(diam)).item())
            full_diameter_from_center_max_list.append((torch.max(diam)).item())
            full_diameter_from_center_min_list.append((torch.min(diam)).item())
            full_diameter_from_center_09_list.append((torch.quantile(diam, 0.9)).item())
            full_diameter_from_center_median_list.append((torch.median(diam)).item())
            pcl = pcl.reshape(pcl.shape[0]*pcl.shape[1], -1)

            max_list_x.append(torch.max(pcl[:,0]).item())
            min_list_x.append(torch.min(pcl[:,0]).item())
            max_list_y.append(torch.max(pcl[:,1]).item())
            min_list_y.append(torch.min(pcl[:,1]).item())
            max_list_z.append(torch.max(pcl[:,2]).item())
            min_list_z.append(torch.min(pcl[:,2]).item())
            print('yay')
    print(f'-----------STD--------------')
    print(f'x std: {np.mean(std_list_x)}')
    print(f'y std: {np.mean(std_list_y)}')
    print(f'z std: {np.mean(std_list_z)}')
    print(f'-----------Density--------------')
    print(f'Density: {np.mean(density_list)}')
    print(f'-----------distance--------------')
    print(f'Distance: {np.mean(avg_dist_list)}')
    print(f'-----------DIAMETER--------------')
    print(f'full diameter from center MEAN: {np.mean(full_diameter_from_center_mean_list)}')
    print(f'full diameter from center MEDIAN: {np.mean(full_diameter_from_center_median_list)}')
    print(f'full diameter from center MAX: {np.mean(full_diameter_from_center_max_list)}')
    print(f'full diameter from center MIN: {np.mean(full_diameter_from_center_min_list)}')
    print(f'full diameter from center 90: {np.mean(full_diameter_from_center_09_list)}')
    print(f'x diameter: {np.mean(diameter_list_x)}')
    print(f'y diameter: {np.mean(diameter_list_y)}')
    print(f'z diameter: {np.mean(diameter_list_z)}')



def visualizeShapesWithEmbeddings(model_name=None, args_shape=None, scaling_factor=None, rgb=False, add_noise=True):
    pcls, label = load_data()
    # shapes = [86, 174, 51]
    shapes = [47, 86, 162, 174, 176, 179]
    # shapes = [86]
    # shapes = [10, 17, 24, 47]
    # shapes = range(10)
    for k in shapes:
        pointcloud = pcls[k][:]

        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud
        if add_noise:
            noise = np.clip(np.random.normal(0.0, scale=0.01, size=(pointcloud.shape)),
                            a_min=-0.05, a_max=0.05)
            noise = noise / 4
            noisy_pointcloud += noise
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors , scaling_fac = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
                       args_shape=args_shape, scaling_factor=scaling_factor)

        colors = colors.detach().cpu().numpy().squeeze()
        colors = colors[:,:4]
        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        if rgb:
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

def visualizeShapesWithEmbeddings3dMatch(model_name=None, args_shape=None, scaling_factor=None, rgb=False):
    train_set = IndoorDataset(data_augmentation=False)
    # sample = train_set.__getitem__(10)
    for k in range(5):
        pointcloud = train_set.__getitem__(k)[0]
        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors , scaling_fac = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
                       args_shape=args_shape, scaling_factor=scaling_factor)

        colors = colors.detach().cpu().numpy().squeeze()
        colors = colors[:,:4]
        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        if rgb:
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
def visualizeRGB(model_name,pointcloud,args_shape,scaling_factor):
    colors , scaling_fac = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
                            args_shape=args_shape, scaling_factor=scaling_factor)

    colors = colors.detach().cpu().numpy().squeeze()
    colors = colors[:, :4]
    layout = go.Layout(
        title=f"Point Cloud with Embedding-based Colors",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
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
def visualizePclClassification(pointcloud, colors):
    colors = colors[:, :5]
    layout = go.Layout(
        title=f"Point Cloud with Embedding-based Colors",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Plot the maximum value embedding with specified colors
    max_embedding_index = np.argmax(colors, axis=1)
    max_embedding_colors = np.array(['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'purple', 'orange'])[
        max_embedding_index]

    data_max_embedding = []
    colors_shape = ['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'purple', 'orange']
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle', '15', '45', '90', 'corner']
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
def visualizeShapesWithEmbeddingsCorners(model_name=None, args_shape=None, scaling_factor=None, rgb=False, add_noise=True):
    pcls, label = load_data()
    shapes = [47,86]
    # shapes = [47,86, 174, 51]
    shapes = [47, 86, 162, 174, 176, 179]
    # shapes = [86]
    # shapes = [10, 17, 24, 47]
    # shapes = range(10)
    for k in shapes:
        pointcloud = pcls[k][:]
        # pointcloud = np.load("pcl1.npy")
        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud
        if add_noise:
            noise = np.clip(np.random.normal(0.0, scale=0.01, size=(pointcloud.shape)),
                            a_min=-0.05, a_max=0.05)
            noise = noise / 4
            noisy_pointcloud += noise
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors , scaling_fac = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
                       args_shape=args_shape, scaling_factor=scaling_factor)

        colors = colors.detach().cpu().numpy().squeeze()
        colors = colors[:,:5]
        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        if rgb:
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
        max_embedding_colors = np.array(['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'purple', 'orange'])[
            max_embedding_index]

        data_max_embedding = []
        colors_shape = ['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'purple', 'orange']
        names = ['plane', 'peak/pit', 'valley/ridge', 'saddle', '15', '45', '90', 'corner']
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


import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm


def vis2(model_name=None, args_shape=None, scaling_factor=None, rgb=False,
                                         add_noise=True):
    shapes = [47, 86, 162, 174, 176, 179]
    shapes = [86,174]

    colormaps = [cm.Reds, cm.Blues, cm.Greens, cm.Purples, cm.Oranges, cm.cool, cm.winter, cm.summer]
    colormaps = [cm.Greys, cm.Blues, cm.Greens, cm.Reds, cm.Oranges]
    colormaps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis]
    colormaps = [cm.plasma, cm.plasma, cm.plasma, cm.plasma, cm.plasma]
    # colormaps = [cm.cool, cm.winter, cm.summer, cm.autumn, cm.Blues, cm.Greens, cm.Purples, cm.Oranges]

    # for k in range(len(pcls)):
    for class_num in [0,1,2,3,4]:
        for k in shapes:
            pointcloud = np.load(f"{k}.npy")
            noisy_pointcloud = pointcloud

            pointcloud = noisy_pointcloud.astype(np.float32)
            colors, scaling_fac = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
                                                 args_shape=args_shape, scaling_factor=scaling_factor)

            colors = colors.detach().cpu().numpy().squeeze()
            colors = colors[:, :5]

            # class_num = 2

            max_embedding_index = np.argmax(colors, axis=1)
            max_embedding_index = class_num * np.ones_like(max_embedding_index)
            max_embedding_weights = np.max(colors, axis=1)  # Get the confidence scores
            max_embedding_weights = colors[:,class_num]  # Get the confidence scores
            # Normalize weights to [0, 1] range
            max_embedding_weights = (max_embedding_weights - np.min(max_embedding_weights)) / (
                        np.max(max_embedding_weights) - np.min(max_embedding_weights) + 1e-6)
            # max_embedding_weights = 0.3333 + max_embedding_weights / 1.5
            # Map weights to colors using class-specific colormaps
            mapped_colors = [colormaps[i](max_embedding_weights[j])[:3] for j, i in enumerate(max_embedding_index)]
            mapped_colors = np.array(mapped_colors)

            layout = go.Layout(
                title=f"Point Cloud with Embedding-based Colors {k}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )

            data_max_embedding = []
            names = ['plane', 'peak/pit', 'valley/ridge', 'saddle', 'corner']

            for i in range(len(colormaps)):
                indices = np.where(max_embedding_index == i)[0]
                if len(indices) > 0:
                    data_max_embedding.append(
                        go.Scatter3d(
                            x=pointcloud[indices, 0],
                            y=pointcloud[indices, 1],
                            z=pointcloud[indices, 2],
                            mode='markers',
                            marker=dict(
                                size=2.5,
                                opacity=1,
                                color=[f'rgb({r * 255},{g * 255},{b * 255})' for r, g, b in mapped_colors[indices]]
                            ),
                            name=f'Max Value Embedding - {names[i]}'
                        )
                    )

            fig_max_embedding = go.Figure(data=data_max_embedding, layout=layout)
            fig_max_embedding.show()


def vis2_static(model_name=None, args_shape=None, scaling_factor=None, rgb=False, add_noise=True):
    """
    Creates static non-interactive 3D point cloud visualizations using matplotlib.
    The visualization shows only the points from a side view perspective and
    adjusts the image boundaries to match the point cloud dimensions.

    Parameters:
    -----------
    model_name : str
        Name of the model to use for classification
    args_shape : tuple
        Shape parameters for the classification
    scaling_factor : float
        Scaling factor for the point cloud
    rgb : bool
        Whether to use RGB color representation
    add_noise : bool
        Whether to add noise to the point cloud
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import os

    # Define shapes to visualize
    shapes = [86]

    # Define colormaps for different classes
    colormaps = [cm.plasma, cm.viridis, cm.inferno, cm.cividis, cm.magma]
    colormaps = [cm.plasma, cm.viridis, cm.cividis, cm.inferno, cm.magma]

    # Define class names for labels
    names = ['plane', 'peak/pit', 'valley/ridge', 'saddle', 'corner']

    # Configure matplotlib for clean visualization
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'

    # Create output directory
    output_dir = "point_cloud_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through classes and shapes
    for class_num in [0, 1, 2, 3, 4]:
        for k in shapes:
            # Load point cloud data
            pointcloud = np.load(f"{k}.npy")
            pointcloud = pointcloud.astype(np.float32)

            # Get classification results
            colors, scaling_fac = classifyPoints(model_name=model_name,
                                                 pcl_src=pointcloud,
                                                 pcl_interest=pointcloud,
                                                 args_shape=args_shape,
                                                 scaling_factor=scaling_factor)

            # Process classification results
            colors = colors.detach().cpu().numpy().squeeze()
            colors = colors[:, :5]

            # Set all points to the current class for visualization
            max_embedding_index = class_num * np.ones(len(pointcloud), dtype=int)

            # Get confidence scores for the current class
            max_embedding_weights = colors[:, class_num]

            # Normalize weights to [0, 1] range
            min_weight = np.min(max_embedding_weights)
            max_weight = np.max(max_embedding_weights)
            max_embedding_weights = (max_embedding_weights - min_weight) / (max_weight - min_weight + 1e-6)

            # Calculate point cloud bounds for precise framing
            y_min, y_max = np.min(pointcloud[:, 1]), np.max(pointcloud[:, 1])
            z_min, z_max = np.min(pointcloud[:, 2]), np.max(pointcloud[:, 2])

            # Create a figure with tight bounds
            fig = plt.figure(dpi=300, frameon=False)
            ax = fig.add_subplot(111, projection='3d')

            # Map weights to colors
            mapped_colors = colormaps[class_num](max_embedding_weights)

            # Plot only the points
            scatter = ax.scatter(
                pointcloud[:, 0],
                pointcloud[:, 1],
                pointcloud[:, 2],
                c=mapped_colors,
                s=2.5,  # marker size
                alpha=1.0  # opacity
            )

            # Remove all visual elements except points
            ax.set_axis_off()
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')

            # Set side view (YZ plane)
            ax.view_init(elev=0, azim=90)

            # Set axis limits precisely to point cloud boundaries
            ax.set_xlim(np.min(pointcloud[:, 0]), np.max(pointcloud[:, 0]))
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            # Disable auto scaling
            ax.autoscale(False)

            # Remove margins to show only points
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            # Use tight layout to remove whitespace
            fig.tight_layout(pad=0)

            # Generate output filename
            output_file = f"{output_dir}/point_cloud_class{class_num}_shape{k}.png"

            # Save with exact bounding box
            plt.savefig(output_file,
                        bbox_inches='tight',
                        pad_inches=0,
                        dpi=300,
                        transparent=True)

            # Close figure to free memory
            plt.close(fig)

            # Additional step: Further crop image to exact point cloud boundaries
            try:
                from PIL import Image
                import numpy as np

                # Open the saved image
                img = Image.open(output_file)
                img_array = np.array(img)

                # Find non-transparent pixels (alpha > 0)
                if img_array.shape[2] == 4:  # RGBA
                    non_empty = np.where(img_array[:, :, 3] > 0)
                    if len(non_empty[0]) > 0 and len(non_empty[1]) > 0:
                        # Get bounds of content
                        min_y, max_y = np.min(non_empty[0]), np.max(non_empty[0])
                        min_x, max_x = np.min(non_empty[1]), np.max(non_empty[1])

                        # Crop to content bounds
                        img_cropped = img.crop((min_x, min_y, max_x + 1, max_y + 1))
                        img_cropped.save(output_file)
                        print(f"Cropped image saved: {output_file}")
                    else:
                        print(f"No visible content to crop in {output_file}")
                else:
                    print(f"Image does not have alpha channel, skipping cropping for {output_file}")
            except ImportError:
                print("PIL not available, skipping final cropping step")
            except Exception as e:
                print(f"Error during image cropping: {e}")

    print(f"Static point cloud visualizations saved to '{output_dir}' directory")

def visualizeShapesWithEmbeddings3dMatchCorners(model_name=None, args_shape=None, scaling_factor=None, rgb=False):
    train_set = IndoorDataset(data_augmentation=False)
    # sample = train_set.__getitem__(10)
    for k in range(5):
        pointcloud = train_set.__getitem__(k)[0]
        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors , scaling_fac = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
                       args_shape=args_shape, scaling_factor=scaling_factor)

        colors = colors.detach().cpu().numpy().squeeze()
        colors = colors[:,:5]
        layout = go.Layout(
            title=f"Point Cloud with Embedding-based Colors {k}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        if rgb:
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
        max_embedding_colors = np.array(['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'purple', 'orange'])[
            max_embedding_index]

        data_max_embedding = []
        colors_shape = ['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'purple', 'orange']
        names = ['plane', 'peak/pit', 'valley/ridge', 'saddle', '15', '45', '90', 'corner']
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

def view_stabiity(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], add_noise=True, create_pcls_func=None, given_pcls=None):
    pcls, label = load_data()
    finished = False
    shapes = [0,51,54, 86, 174]
    # shapes = [86, 174]
    # shapes = [86]
    # shapes = [51]
    for k in shapes:
        if finished==True:
            break
        if given_pcls is None:
            pointcloud = pcls[k][:]
            # save_receptive_field(pointcloud, pointcloud[10, :], [1,5,10,15,20,25,30], f"akak.html")
            # continue
            rotated_pcl, rotation_matrix, _ = random_rotation_translation(pointcloud)
            chosen_point = [10,10]
            # chosen_point = [50,50]
            if create_pcls_func is not None:
                pcl1, pcl2, pcl1_indices, pcl2_indices, overlapping_indices = create_pcls_func(pointcloud)
                chosen_overlapping_point = np.random.choice(overlapping_indices)
                index_pcl_1 = np.where(pcl1_indices == chosen_overlapping_point)[0][0]
                index_pcl_2 = np.where(pcl2_indices == chosen_overlapping_point)[0][0]
                chosen_point = [index_pcl_1, index_pcl_2]
                rotated_pcl, rotation_matrix, translation = random_rotation_translation(pcl2)
            if add_noise:
                noise_1 = np.clip(np.random.normal(0.0, scale=0.01, size=(pointcloud.shape)),
                                a_min=-0.05, a_max=0.05)
                noise_2 = np.clip(np.random.normal(0.0, scale=0.01, size=(pointcloud.shape)),
                                a_min=-0.05, a_max=0.05)
                # Because noise is added for 700 sample which is roughly a quarter of the original so less noise should be added
                noise_1 = noise_1 / 4
                noise_2 = noise_2 / 4
            else:
                noise_1 = np.zeros_like(pointcloud)
                noise_2 = np.zeros_like(pointcloud)
            noisy_pointcloud_1 = pointcloud + noise_1
            noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
            noisy_pointcloud_2 = rotated_pcl + noise_2
            noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        else:
            noisy_pointcloud_1 = given_pcls[0]
            noisy_pointcloud_2 = given_pcls[1]
            chosen_point = given_pcls[2].detach().cpu().numpy()
            finished=True


        emb_1 , scaling_fac = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_2 , scaling_fac = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy().squeeze()
        emb_2 = emb_2.detach().cpu().numpy().squeeze()

        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, chosen_point)
        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
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
                # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, global_emb_1, global_emb_2)
                emb_1 = np.hstack((emb_1, global_emb_1))
                emb_2 = np.hstack((emb_2, global_emb_2))
                # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, chosen_point)

        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, chosen_point)
import itertools
def check3dStability(cls_args=None, scaling_factor=None, scales=1, receptive_field=[1, 2]):
    pointcloud = np.load("pcl1.npy")
    receptive_fields = [[1]+list(combo) for r in range(2, 6) for combo in itertools.combinations([3,5,7,9,11,13,15], r)]
    # receptive_fields = [[1, 3, 5, 9, 13],[1, 7, 13],[1, 5, 13, 15],[1, 3, 7, 11],[1, 3, 7, 11, 13],[1, 3, 9, 11, 15],[1, 5, 7, 9, 11],[1, 5, 7, 13, 15]]
    dic_yay ={}
    for r_field in receptive_fields:
        indices= sorted(np.random.choice(range(len(pointcloud)), size=20, replace=False))
        count = 0
        for ind in indices:
            rotated_pcl, rotation_matrix, _ = random_rotation_translation(pointcloud)
            noise_1 = np.clip(np.random.normal(0.0, scale=0.01, size=(pointcloud.shape)),
                              a_min=-0.05, a_max=0.05)
            noise_2 = np.clip(np.random.normal(0.0, scale=0.01, size=(pointcloud.shape)),
                              a_min=-0.05, a_max=0.05)
            # Because noise is added for 700 sample which is roughly a quarter of the original so less noise should be added
            noise_1 = noise_1 / 4
            noise_2 = noise_2 / 4
            noisy_pointcloud_1 = pointcloud + noise_1
            noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
            noisy_pointcloud_2 = rotated_pcl + noise_2
            noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

            emb_1 , scaling_factor_1 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1[ind,:].reshape(1,3), args_shape=cls_args, scaling_factor=scaling_factor)
            emb_2 , scaling_factor_2 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2, args_shape=cls_args, scaling_factor=scaling_factor_1)

            emb_1 = emb_1.detach().cpu().numpy().squeeze()
            emb_2 = emb_2.detach().cpu().numpy().squeeze()

            for scale in r_field[1:]:
                subsampled_1 = farthest_point_sampling_o3d(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1) // scale))
                subsampled_2 = farthest_point_sampling_o3d(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2) // scale))

                global_emb_1 , scaling_factor_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_1,
                                              pcl_interest=noisy_pointcloud_1[ind,:].reshape(1,3), args_shape=cls_args,
                                              scaling_factor=scaling_factor)

                global_emb_2 , scaling_factor_2 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_2,
                                              pcl_interest=noisy_pointcloud_2, args_shape=cls_args,
                                              scaling_factor=scaling_factor_1)

                global_emb_1 = global_emb_1.detach().cpu().numpy().squeeze()
                global_emb_2 = global_emb_2.detach().cpu().numpy().squeeze()
                # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, global_emb_1, global_emb_2)
                emb_1 = np.hstack((emb_1, global_emb_1))
                emb_2 = np.hstack((emb_2, global_emb_2))
                # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, chosen_point)

            # Calculate distances from the random embedding to all embeddings in embedding2
            distances = np.linalg.norm(emb_2 - emb_1, axis=1)

            # Find indices of the 20 closest points
            closest_indices = np.argsort(distances)[:5]
            if ind in closest_indices:
                count +=1

        dic_yay[str(r_field)]=count
        print(f'{str(r_field)}: {count}')
    sorted_dict = {k: v for k, v in sorted(dic_yay.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_dict)

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


from sklearn.neighbors import NearestNeighbors
import numpy as np
def find_closest_points_best_buddy_beta(
    embeddings1,
    embeddings2,
    num_of_pairs=40,
    n_neighbors=3,
    avoid_planes=False,
    avoid_diff_classification=False
):
    # Extract classifications
    classification_1 = np.argmax(embeddings1[:, :5], axis=1)
    classification_2 = np.argmax(embeddings2[:, :5], axis=1)

    # Track original indices
    original_indices_1 = np.arange(embeddings1.shape[0])
    original_indices_2 = np.arange(embeddings2.shape[0])

    # Optionally filter out plane points
    if avoid_planes:
        mask_plane_1 = classification_1 != 0
        mask_plane_2 = classification_2 != 0
        embeddings1 = embeddings1[mask_plane_1]
        embeddings2 = embeddings2[mask_plane_2]
        classification_1 = classification_1[mask_plane_1]
        classification_2 = classification_2[mask_plane_2]
        original_indices_1 = original_indices_1[mask_plane_1]
        original_indices_2 = original_indices_2[mask_plane_2]

    # Initialize NearestNeighbors instance for embeddings1 and embeddings2
    nbrs1 = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings2)
    nbrs2 = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings1)

    # Find the indices and distances of the closest points
    distances1, indices1 = nbrs1.kneighbors(embeddings1)
    distances2, indices2 = nbrs2.kneighbors(embeddings2)

    # Optionally filter out point pairings with different classifications
    if avoid_diff_classification:
        classification_mask_1 = classification_1.reshape(-1,1) != classification_2[indices1]
        distances1[classification_mask_1] = np.inf
        classification_mask_2 = classification_2.reshape(-1,1) != classification_1[indices2]
        distances2[classification_mask_2] = np.inf

    best_buddies = []

    for index_1, corr_indices in enumerate(indices1):
        # Check if the point is a best buddy
        for nn_num, index_2 in enumerate(corr_indices):
            cur_dist = distances1[index_1, nn_num]
            if cur_dist == np.inf:
                continue
            if index_1 in indices2[index_2]:
                best_buddies.append((index_1, nn_num, index_2, cur_dist))

    best_buddies = np.array(best_buddies)

    # Get indices of k smallest d values
    lowest_indices = np.argsort(best_buddies[:, -1])[:num_of_pairs]
    # Sort by distances and select the top num_of_pairs
    best_buddies = [best_buddies[i] for i in lowest_indices]

    # Map back to original indices
    emb1_indices = np.array([original_indices_1[int(x[0])] for x in best_buddies])
    emb2_indices = np.array([original_indices_2[int(x[2])] for x in best_buddies])

    return emb1_indices, emb2_indices
def find_closest_points_beta(
    embeddings1,
    embeddings2,
    num_of_pairs=40,
    max_non_unique_correspondences=3,
    n_neighbors=3,
    avoid_duplicates=True,
    avoid_diff_classification=True,
    avoid_planes=True
):
    # Extract classifications
    classification_1 = np.argmax(embeddings1[:, :5], axis=1)
    classification_2 = np.argmax(embeddings2[:, :5], axis=1)

    # Track original indices
    original_indices_1 = np.arange(embeddings1.shape[0])
    original_indices_2 = np.arange(embeddings2.shape[0])

    # Optionally filter out plane points
    if avoid_planes:
        mask_plane_1 = classification_1 != 0
        mask_plane_2 = classification_2 != 0
        embeddings1 = embeddings1[mask_plane_1]
        embeddings2 = embeddings2[mask_plane_2]
        classification_1 = classification_1[mask_plane_1]
        classification_2 = classification_2[mask_plane_2]
        original_indices_1 = original_indices_1[mask_plane_1]
        original_indices_2 = original_indices_2[mask_plane_2]

    # Nearest Neighbors search
    size_1 = embeddings1.shape[0]
    size_2 = embeddings2.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings2)
    distances, indices = nbrs.kneighbors(embeddings1)
    nearest_emb2_indices = indices[:, 0]

    # Optionally filter out duplicates
    if avoid_duplicates:
        appearance_1_nn = np.bincount(nearest_emb2_indices, minlength=size_2)
        mask_dup = (appearance_1_nn >= max_non_unique_correspondences)
        distances[:, 0][mask_dup[nearest_emb2_indices]] = np.inf

    # Filter out point pairings with different classifications after nearest neighbors
    if avoid_diff_classification:
        # Compare classifications of paired points
        classification_mask = classification_1.reshape(-1,1) != classification_2[indices]
        # Mark distances for invalid pairs as infinity
        distances[classification_mask] = np.inf

    # Identify valid indices based on distances
    smallest_distances_indices = np.argsort(distances.flatten())
    first_inf_index = np.where(distances.flatten()[smallest_distances_indices] == np.inf)[0][
        0] if np.inf in distances else len(distances.flatten())
    num_of_pairs_2_take = min(num_of_pairs, first_inf_index)
    smallest_distances_indices = smallest_distances_indices[:num_of_pairs_2_take]

    # Map back to original indices
    filtered_emb1_indices = smallest_distances_indices.squeeze() // n_neighbors
    filtered_emb2_indices = (indices.flatten()[smallest_distances_indices].squeeze())
    original_emb1_indices = original_indices_1[filtered_emb1_indices]
    original_emb2_indices = original_indices_2[filtered_emb2_indices]

    return original_emb1_indices, original_emb2_indices


def find_closest_points(embeddings1, embeddings2, num_of_pairs=40, max_non_unique_correspondences=3, n_neighbors=1):
    classification_1 = np.argmax(embeddings1[:,:5], axis=1)
    classification_2 = np.argmax(embeddings2[:,:5], axis=1)
    size_1 = embeddings1.shape[0]
    size_2 = embeddings2.shape[0]
    # Initialize NearestNeighbors instance
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings2)

    # Find the indices and distances of the closest points in embeddings2 for each point in embeddings1
    distances, indices = nbrs.kneighbors(embeddings1)
    appearance_1_nn = np.bincount(indices[:, 0], minlength=len(embeddings2))

    a1 = np.count_nonzero(distances == np.inf)
    # remove points which are NN of multiple points
    mask_dup = (appearance_1_nn >= max_non_unique_correspondences)
    distances[:, 0][mask_dup[indices[:, 0]]] = np.inf

    a2 = np.count_nonzero(distances == np.inf)
    # remove point pairings which have different classificatio
    mask_cls = classification_1 != classification_2[indices[:, 0]]
    distances[:, 0][mask_cls] = np.inf

    a3 = np.count_nonzero(distances==np.inf)
    # remove plane points
    mask_plane = (classification_1 == 0)
    distances[:, 0][mask_plane] = np.inf

    a4 = np.count_nonzero(distances == np.inf)
    smallest_distances_indices = np.argsort(distances.flatten())
    first_inf_index = np.where(distances.flatten()[smallest_distances_indices] == np.inf)[0][0]
    num_of_pairs_2_take = min(num_of_pairs, first_inf_index)
    smallest_distances_indices= smallest_distances_indices[:num_of_pairs_2_take]
    emb1_indices = smallest_distances_indices.squeeze() % size_1
    emb2_indices = (indices.flatten()[smallest_distances_indices].squeeze()) % size_2
    return emb1_indices, emb2_indices
def find_closest_points_with_dup(embeddings1, embeddings2, num_of_pairs=40, max_non_unique_correspondences=3, n_neighbors=1):
    classification_1 = np.argmax(embeddings1[:, :5], axis=1)
    classification_2 = np.argmax(embeddings2[:, :5], axis=1)
    size_1 = embeddings1.shape[0]
    size_2 = embeddings2.shape[0]
    # Initialize NearestNeighbors instance
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2)

    # Find the indices and distances of the closest points in embeddings2 for each point in embeddings1
    distances, indices = nbrs.kneighbors(embeddings1)
    a1 = np.count_nonzero(distances == np.inf)
    # remove point pairings which have different classificatio
    mask_cls = classification_1 != classification_2[indices[:, 0]]
    distances[:, 0][mask_cls] = np.inf
    a2 = np.count_nonzero(distances==np.inf)
    # remove plane points
    mask_plane = (classification_1 == 0)
    distances[:, 0][mask_plane[indices[:, 0]]] = np.inf
    a3 = np.count_nonzero(distances == np.inf)
    smallest_distances_indices = np.argsort(distances.flatten())
    first_inf_index = np.where(distances.flatten()[smallest_distances_indices] == np.inf)[0][0]
    # first_inf_index = np.inf


    num_of_pairs_2_take = min(num_of_pairs, first_inf_index)
    smallest_distances_indices= smallest_distances_indices[:num_of_pairs_2_take]
    emb1_indices = smallest_distances_indices.squeeze() % size_1
    emb2_indices = (indices.flatten()[smallest_distances_indices].squeeze()) % size_2
    return emb1_indices, emb2_indices#, dist_sorted

def min_max_scale(distances):
    min_val = np.min(distances)
    max_val = np.max(distances)
    return (distances - min_val) / (max_val - min_val)
def z_score_standardize(distances):
    mean = np.mean(distances)
    std = np.std(distances)
    return (distances - mean) / std
def find_closest_points_best_buddy(embeddings1, embeddings2, num_of_pairs=40, max_non_unique_correspondences=3):
    classification_1 = np.argmax(embeddings1[:, :5], axis=1)
    classification_2 = np.argmax(embeddings2[:, :5], axis=1)

    # Initialize NearestNeighbors instance for embeddings1 and embeddings2
    nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2)
    nbrs2 = NearestNeighbors(n_neighbors=max_non_unique_correspondences, algorithm='auto').fit(embeddings1)

    # Find the indices and distances of the closest points
    distances1, indices1 = nbrs1.kneighbors(embeddings1)
    distances2, indices2 = nbrs2.kneighbors(embeddings2)

    # remove point pairings which have different classificatio
    mask_cls = (classification_1 != classification_2)
    distances1[:, 0][mask_cls[indices1[:, 0]]] = np.inf



    # remove plane points
    mask_plane = (classification_1 == 0)
    distances1[:, 0][mask_plane[indices1[:, 0]]] = np.inf

    duplicates = np.zeros(len(embeddings1))
    best_buddies = []

    for i, index in enumerate(indices1.squeeze()):
        # Check if the point is a best buddy
        if ( i in indices2[index] ) and ( classification_1[i] == classification_2[index] ):
        # if ( i in indices2[index] ):
            best_buddies.append((i, index))

    # Sort by distances and select the top num_neighbors
    best_buddies = sorted(best_buddies, key=lambda x: distances1[x[0]])
    best_buddies = best_buddies[:num_of_pairs]

    emb1_indices = np.array([x[0] for x in best_buddies])
    emb2_indices = np.array([x[1] for x in best_buddies])

    return emb1_indices, emb2_indices

def random_rotation_translation(pointcloud, translation=np.array([0,0,0])):
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
  rotation_matrix = Rotation.random().as_matrix()
  # Apply rotation to centered pointcloud
  rotated_cloud = (centered_cloud @ rotation_matrix.T)
  new_pointcloud = (rotated_cloud + center) + translation

  return new_pointcloud , rotation_matrix, translation

import numpy as np

def rotatePCLToCanonical(point_cloud):
    """
    Rotates the point cloud to align its principal direction with the z-axis.

    Args:
      point_cloud: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      rotated_point_cloud: numpy array of shape (N, 3), the rotated point cloud.
      rotation_matrix: numpy array of shape (3, 3), the rotation matrix used.
    """
    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    # Calculate the rotation matrix to align the normal with the z-axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal_at_centroid, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal_at_centroid, z_axis)

    if s == 0:  # Already aligned with z-axis
        rotation_matrix = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

    # Rotate the point cloud
    rotated_point_cloud = point_cloud @ rotation_matrix.T

    return rotated_point_cloud, rotation_matrix

def fit_surface_quadratic_constrained(points):
    """
    Fits a quadratic surface constrained to f = 0 to a centered point cloud.

    Args:
      points: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      k1, k2: Principal curvatures of the surface.
      mean_error, median_error: Mean and median projection errors of the points to the surface.
    """
    # Center the points around the mean
    centroid = points.mean(axis=0)
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

    # Compute Gaussian and Mean curvature
    K = (4 * (a * b) - (c ** 2)) / ((1 + d ** 2 + e ** 2) ** 2)
    H = (a * (1 + e ** 2) - d * e * c + b * (1 + d ** 2)) / (((d ** 2) + (e ** 2) + 1) ** 1.5)
    discriminant = H ** 2 - K
    k1 = H + np.sqrt(discriminant)
    k2 = H - np.sqrt(discriminant)

    # Compute the z values of the fitted surface
    z_fitted = (a * centered_points[:, 0] ** 2 + b * centered_points[:, 1] ** 2 +
                c * centered_points[:, 0] * centered_points[:, 1] +
                d * centered_points[:, 0] + e * centered_points[:, 1])

    # Compute projection errors
    errors = np.abs(z - z_fitted)
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    return k1, k2, mean_error, median_error

def rotate_and_fit(point_cloud):
    """
    Rotates the point cloud to canonical position and fits a quadratic surface.

    Args:
      point_cloud: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      k1, k2: Principal curvatures of the fitted surface.
      mean_error, median_error: Mean and median projection errors of the points to the surface.
    """
    # Rotate point cloud to canonical position
    rotated_point_cloud, rotation_matrix = rotatePCLToCanonical(point_cloud)

    # Fit the rotated point cloud with the quadratic surface
    k1, k2, mean_error, median_error = fit_surface_quadratic_constrained(rotated_point_cloud)

    return k1, k2, mean_error, median_error


def calcDist(src_knn_pcl, scaling_mode):
    pcl = src_knn_pcl[0].permute(1,2,0)
    pairwise_distances = torch.cdist(pcl, pcl, p=2)
    num_points = pcl.shape[1]
    diam = (((torch.max(pairwise_distances[:, 0, :], dim=1))[0]))

    if scaling_mode == "mean":
        d_mean = (torch.mean(diam)).item()
        scale = 5.121 / d_mean

    elif scaling_mode == "median":
        d_median = (torch.median(diam)).item()
        scale = 4.8959 / d_median

    elif scaling_mode == "max":
        d_max = (torch.max(diam)).item()
        scale =  13.24 / d_max

    elif scaling_mode == "min":
        d_min = (torch.min(diam)).item()
        scale = 2.2962 / d_min
    elif scaling_mode == "d_90":
        d_90 = (torch.quantile(diam, 0.9)).item()
        scale = 6.9937 / d_90
    elif scaling_mode == "axis":
        diameter_med = torch.median(torch.median((torch.max(abs(pcl),dim=1))[0] , dim=0)[0])
        # scale = 2.206 / diameter_med
        scale = 2.3478 / diameter_med
    elif scaling_mode == "one":
        median_of_median_axis = torch.median(torch.median((torch.max(pcl, dim=1)[0]-torch.min(pcl, dim=1)[0]), dim=0)[0])
        scale = 1 / median_of_median_axis        # scale = 2 / median_of_median_axis
    else:
        median_of_median_axis = torch.median(
        torch.median((torch.max(pcl, dim=1)[0] - torch.min(pcl, dim=1)[0]), dim=0)[0])
        scale = 1 / median_of_median_axis
        scale = scale * float(scaling_mode)
    return scale
def classifyPoints(model_name=None, pcl_src=None,pcl_interest=None, args_shape=None, scaling_factor=None, device='cpu'):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'models_weights/{model_name}.pt',weights_only=True))
    model.to(device)
    model.eval()
    # neighbors_centered = get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k=41)
    neighbors_centered = get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k=21)
    src_knn_pcl = torch.tensor(neighbors_centered)
    if isinstance(scaling_factor, str):
        scaling_factor_final = (calcDist(src_knn_pcl, scaling_factor)).item()
    else:
        scaling_factor_final = scaling_factor
    src_knn_pcl = scaling_factor_final * src_knn_pcl

    output = model(src_knn_pcl)
    return output, scaling_factor_final
def get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k):
    """
    Returns the k nearest neighbors for each point in the point cloud.

    Args:
        point_cloud (np.ndarray): Point cloud of shape (pcl_size, 3)
        k (int): Number of nearest neighbors to return

    Returns:
        np.ndarray: Array of shape (1, 3, pcl_size, k) containing the k nearest neighbors for each point
    """
    pcl_size = pcl_interest.shape[0]
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(pcl_src)
    distances, indices = neigh.kneighbors(pcl_interest)

    neighbors_centered = np.empty((1, 3, pcl_size, k), dtype=pcl_src.dtype)
    # Each point cloud should be centered around first point which is at the origin
    for i in range(pcl_size):
        orig = pcl_src[indices[i, :]] - pcl_interest[i,:]
        if not (np.array_equal(orig[0,], np.array([0,0,0]))):
            orig = np.vstack([np.array([0,0,0]), orig])[:-1]
        neighbors_centered[0, :, i, :] = orig.T

    return neighbors_centered

def farthest_point_sampling_o3d(point_cloud, k):
    o3d_pcd = to_o3d_pcd(point_cloud)
    downpcd_farthest = o3d_pcd.farthest_point_down_sample(k)
    return np.asarray(downpcd_farthest.points, dtype=np.float32)

def save_receptive_field(point_cloud, point, rfield=[1, 5, 10, 20], filename="plot.html", dir=r"./"):
    fig = go.Figure()

    # Define a color list for the receptive fields
    colors = ['blue', 'orange', 'brown', 'red', 'green', 'pink', 'purple', 'cyan', 'magenta']

    # Plot the original point cloud in gray
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers', marker=dict(size=2, color='gray'),
        name='Original Point Cloud'
    ))

    # Plot the interest point in green
    fig.add_trace(go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]],
        mode='markers', marker=dict(size=5, color='green'),
        name='Interest Point'
    ))

    # Process each receptive field scale
    for i, scale in enumerate(rfield):
        if i >= len(colors):
            break
        if ((int)(len(point_cloud) // scale) < 21):
            break

        # Subsample point cloud according to the scale
        subsampled_pc = farthest_point_sampling_o3d(point_cloud, k=int(len(point_cloud) // scale))

        # Get 40 nearest neighbors centered around the given point
        # neighbors_centered = (get_k_nearest_neighbors_diff_pcls(subsampled_pc, point.reshape(1,3), 41).squeeze()).T
        neighbors_centered = (get_k_nearest_neighbors_diff_pcls(subsampled_pc, point.reshape(1,3), 21).squeeze()).T
        shape_size = np.abs(np.max(neighbors_centered, axis=0) -np.min(neighbors_centered, axis=0))
        # Translate neighbors back to the original coordinates
        neighbors_uncentered = neighbors_centered + point

        # Plot the neighbors for this receptive field in the specified color
        fig.add_trace(go.Scatter3d(
            x=neighbors_uncentered[:, 0], y=neighbors_uncentered[:, 1], z=neighbors_uncentered[:, 2],
            mode='markers', marker=dict(size=3, color=colors[i]),
            name=f'Receptive Field {scale}; Size: {str(shape_size)}'
        ))

    fig.update_layout(
        title=f'Receptive Fields Visualization',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=100)
    )

    # Save the figure as an HTML file
    fig.write_html(os.path.join(dir,filename))