import numpy as np
from plotting_functions import *
from ransac import *
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
        noisy_pointcloud = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud = noisy_pointcloud.astype(np.float32)
        colors = checkOnShapes(model_name=model_name,
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

        # # Plot the maximum value embedding with specified colors
        # max_embedding_index = surface_labels
        # max_embedding_colors = np.array(['red', 'blue', 'green', 'pink'])[max_embedding_index]
        #
        # data_max_embedding = []
        # colors_shape = ['red', 'blue', 'green', 'pink']
        # names = ['plane', 'peak/pit', 'valley/ridge', 'saddle']
        # for color, name in zip(colors_shape, names):
        #     indices = np.where(max_embedding_colors == color)[0]
        #     data_max_embedding.append(
        #         go.Scatter3d(
        #             x=pointcloud[indices, 0],
        #             y=pointcloud[indices, 1],
        #             z=pointcloud[indices, 2],
        #             mode='markers',
        #             marker=dict(
        #                 size=2,
        #                 opacity=0.8,
        #                 color=color
        #             ),
        #             name=f'Max Value Embedding - {name}'
        #         )
        #     )
        # fig_max_embedding = go.Figure(data=data_max_embedding, layout=layout)
        # fig_max_embedding.show()
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

def view_stabiity(model_name=None, args_shape=None, scaling_factor=None):
    pcls, label = load_data()
    shapes = [82, 83, 86, 174]
    for k in shapes:
        pointcloud = pcls[k][:]
        rotated_pcl, rotation_matrix, _ = random_rotation_translation(pointcloud)

        noisy_pointcloud_1 = pointcloud + np.random.normal(0, 0.01, pointcloud.shape)
        noisy_pointcloud_1 = noisy_pointcloud_1.astype(np.float32)
        noisy_pointcloud_2 = rotated_pcl + np.random.normal(0, 0.01, rotated_pcl.shape)
        noisy_pointcloud_2 = noisy_pointcloud_2.astype(np.float32)

        emb_1 = classifyPoints(model_name=model_name, pcl_src=noisy_pointcloud_1, pcl_interest=noisy_pointcloud_1, args_shape=args_shape, scaling_factor=scaling_factor)
        emb_2 = classifyPoints(model_name=model_name, pcl_src=noisy_pointcloud_2, pcl_interest=noisy_pointcloud_2, args_shape=args_shape, scaling_factor=scaling_factor)

        emb_1 = emb_1.detach().cpu().numpy()
        emb_2 = emb_2.detach().cpu().numpy()


        plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1, emb_2)
        # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1[:,:4], emb_2[:,:4])
        # plot_point_cloud_with_colors_by_dist_2_pcls(noisy_pointcloud_1, noisy_pointcloud_2, emb_1[:,4:], emb_2[:,4:])

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
  theta_x = np.random.rand() * 2 * np.pi
  theta_y = np.random.rand() * 2 * np.pi
  theta_z = np.random.rand() * 2 * np.pi
  rotation_matrix = (Rotation.from_euler("xyz", [theta_x, theta_y, theta_z], degrees=False)).as_matrix()
  # Apply rotation to centered pointcloud
  rotated_cloud = (centered_cloud @ rotation_matrix)
  new_pointcloud = (rotated_cloud + center) + translation

  return new_pointcloud , rotation_matrix, translation

def classifyPoints(model_name=None, pcl_src=None,pcl_interest=None, args_shape=None, scaling_factor=None):
    model = shapeClassifier(args_shape)
    model.load_state_dict(torch.load(f'models_weights/{model_name}.pt'))
    model.eval()
    neighbors_centered = get_k_nearest_neighbors_diff_pcls(pcl_src, pcl_interest, k=41)
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

def find_mean_diameter_for_specific_coordinate(specific_coordinates):
    pairwise_distances = torch.cdist(specific_coordinates.unsqueeze(2), specific_coordinates.unsqueeze(2))
    largest_dist = pairwise_distances.view(specific_coordinates.shape[0], -1).max(dim=1).values
    mean_distance = torch.mean(largest_dist)
    return mean_distance