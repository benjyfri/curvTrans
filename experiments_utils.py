import numpy as np
import torch
from plotting_functions import *
from ransac import *
from threedmatch import *
from indoor import *
import platform
def load_data(partition='test', divide_data=1):
    DATA_DIR = r'C:\\Users\\benjy\\Desktop\\curvTrans\\bbsWithShapes\\data'
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
            noisy_pointcloud += noise
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
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
        colors = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
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
    colors = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
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
    shapes = [86, 174, 51]
    shapes = [47, 86, 162, 174, 176, 179]
    # shapes = [162]
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
            noisy_pointcloud += noise
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
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

def visualizeShapesWithEmbeddings3dMatchCorners(model_name=None, args_shape=None, scaling_factor=None, rgb=False):
    train_set = IndoorDataset(data_augmentation=False)
    # sample = train_set.__getitem__(10)
    for k in range(5):
        pointcloud = train_set.__getitem__(k)[0]
        # bin_file = "000098.bin"
        # pointcloud = read_bin_file(bin_file)
        noisy_pointcloud = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors = classifyPoints(model_name=model_name, pcl_src=pointcloud, pcl_interest=pointcloud,
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

def view_stabiity(cls_args=None,num_worst_losses = 3, scaling_factor=None, scales=1, receptive_field=[1, 2], amount_of_interest_points=100,
                                    num_of_ransac_iter=100, shapes=[86, 162, 174, 176, 179], plot_graphs=0, create_pcls_func=None, given_pcls=None):
    pcls, label = load_data()
    finished = False
    shapes = [0,51,54, 86, 174]
    # shapes = [51]
    for k in shapes:
        if finished==True:
            break
        if given_pcls is None:
            pointcloud = pcls[k][:]
            # save_receptive_field(pointcloud, pointcloud[10, :], [1,5,10,15,20,25], f"{k}_lll.html")
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

            noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
            noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
            noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
            noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        else:
            noisy_pointcloud_1 = given_pcls[0]
            noisy_pointcloud_2 = given_pcls[1]
            chosen_point = given_pcls[2].detach().cpu().numpy()
            finished=True


        emb_1 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1, args_shape=cls_args, scaling_factor=scaling_factor)
        emb_2 = classifyPoints(model_name=cls_args.exp, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2, args_shape=cls_args, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy().squeeze()
        emb_2 = emb_2.detach().cpu().numpy().squeeze()

        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2, chosen_point)
        # multiscale embeddings
        if scales > 1:
            for scale in receptive_field[1:]:
                subsampled_1 = farthest_point_sampling_o3d(noisy_pointcloud_1, k=(int)(len(noisy_pointcloud_1) // scale))
                subsampled_2 = farthest_point_sampling_o3d(noisy_pointcloud_2, k=(int)(len(noisy_pointcloud_2) // scale))

                global_emb_1 = classifyPoints(model_name=cls_args.exp,
                                              pcl_src=subsampled_1,
                                              pcl_interest=noisy_pointcloud_1, args_shape=cls_args,
                                              scaling_factor=scaling_factor)

                global_emb_2 = classifyPoints(model_name=cls_args.exp,
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
    max_non_unique_correspondences=3,
    avoid_planes=True,
    avoid_diff_classification=True
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
    nbrs1 = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings2)
    nbrs2 = NearestNeighbors(n_neighbors=max_non_unique_correspondences, algorithm='auto').fit(embeddings1)

    # Find the indices and distances of the closest points
    distances1, indices1 = nbrs1.kneighbors(embeddings1)
    distances2, indices2 = nbrs2.kneighbors(embeddings2)

    # Optionally filter out point pairings with different classifications
    if avoid_diff_classification:
        classification_mask = classification_1 != classification_2[indices1[:, 0]]
        distances1[:, 0][classification_mask] = np.inf

    # Filter out duplicates
    duplicates = np.zeros(len(embeddings1))
    best_buddies = []

    for i, index in enumerate(indices1.squeeze()):
        # Check if the point is a best buddy
        if i in indices2[index] and distances1[i, 0] != np.inf:
            best_buddies.append((i, index))

    # Sort by distances and select the top num_of_pairs
    best_buddies = sorted(best_buddies, key=lambda x: distances1[x[0], 0])
    best_buddies = best_buddies[:num_of_pairs]

    # Map back to original indices
    emb1_indices = np.array([original_indices_1[x[0]] for x in best_buddies])
    emb2_indices = np.array([original_indices_2[x[1]] for x in best_buddies])

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
        classification_mask = classification_1 != classification_2[nearest_emb2_indices]
        # Mark distances for invalid pairs as infinity
        distances[:, 0][classification_mask] = np.inf

    # Identify valid indices based on distances
    smallest_distances_indices = np.argsort(distances.flatten())
    first_inf_index = np.where(distances.flatten()[smallest_distances_indices] == np.inf)[0][
        0] if np.inf in distances else len(distances.flatten())
    num_of_pairs_2_take = min(num_of_pairs, first_inf_index)
    smallest_distances_indices = smallest_distances_indices[:num_of_pairs_2_take]

    # Map back to original indices
    filtered_emb1_indices = smallest_distances_indices.squeeze() % size_1
    filtered_emb2_indices = (indices.flatten()[smallest_distances_indices].squeeze()) % size_2
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
        scale = 2.206 / diameter_med
    else:
        d_min = (torch.min(diam)).item()
        scale = 2.2962 / d_min
        scale = scale * scaling_mode
    return scale

def calcDistTorch(src_knn_pcl, scaling_mode):
    """
    Calculate the scaling factor based on the pairwise distances in the point clouds for a batch of point clouds.

    Args:
        src_knn_pcl (torch.Tensor): Source k-nearest neighbors point clouds of shape (batch_size, 1, 3, num_points, k)
        scaling_mode (str or float): Scaling mode to use ('mean', 'median', 'max', 'min', 'd_90', or a float scaling factor)

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing the scaling factor for each batch.
    """
    batch_size = src_knn_pcl.shape[0]
    pcl = src_knn_pcl.permute(0, 2, 3, 1)  # Shape: (batch_size, num_points, k, 3)
    pairwise_distances = torch.cdist(pcl, pcl, p=2)  # Shape:

    diam = torch.max(pairwise_distances[:, :, 0, :], dim=2)[0]  # Shape: (batch_size, num_points)

    if scaling_mode == "mean":
        d_mean = torch.mean(diam, dim=1)
        scale = 13.23 / d_mean

    elif scaling_mode == "median":
        d_median = torch.median(diam, dim=1)[0]
        scale = 12.75 / d_median

    elif scaling_mode == "max":
        d_max = torch.max(diam, dim=1)[0]
        scale = 37.06 / d_max

    elif scaling_mode == "min":
        d_min = torch.min(diam, dim=1)[0]
        scale = 2.22 / d_min

    elif scaling_mode == "d_90":
        d_90 = torch.quantile(diam, 0.9, dim=1)
        scale = 19.13 / d_90

    else:
        d_min = torch.min(diam, dim=1)[0]
        scale = 2.22 / d_min
        scale = scale * scaling_mode

    return scale
def classifyPoints(model_name=None, pcl_src=None,pcl_interest=None, args_shape=None, scaling_factor=None, device='cpu'):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'models_weights/{model_name}.pt'))
    model.to(device)
    model.eval()
    if isinstance(pcl_src, torch.Tensor):
        neighbors_centered = get_k_nearest_neighbors_diff_pcls_torch(pcl_src, pcl_interest, k=41)
        src_knn_pcl = neighbors_centered
        scaling_factor_final = calcDistTorch(src_knn_pcl, scaling_factor)
        scaling_factor_final.to(device)
        src_knn_pcl = (scaling_factor_final.view(scaling_factor_final.shape[0], 1, 1, 1)) * src_knn_pcl
    else:
        # neighbors_centered = get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k=41)
        neighbors_centered = get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k=21)
        src_knn_pcl = torch.tensor(neighbors_centered)
        scaling_factor_final = calcDist(src_knn_pcl, scaling_factor)
        # scaling_factor_final = (2.42/0.15)
        # scaling_factor_final = (1.098/0.117)
        src_knn_pcl = scaling_factor_final * src_knn_pcl

    output = model(src_knn_pcl)
    return output
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

        # Translate neighbors back to the original coordinates
        neighbors_uncentered = neighbors_centered + point

        # Plot the neighbors for this receptive field in the specified color
        fig.add_trace(go.Scatter3d(
            x=neighbors_uncentered[:, 0], y=neighbors_uncentered[:, 1], z=neighbors_uncentered[:, 2],
            mode='markers', marker=dict(size=3, color=colors[i]),
            name=f'Receptive Field {scale}'
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