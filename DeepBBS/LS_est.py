import os
import gc
import torch
from data import ModelNet40
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
import numpy as np
import argparse
from util import npmat2euler, transform_point_cloud, cdist_torch
import plotly.graph_objects as go
def fit_surface_quadratic_constrained(points, k=20):
    """
    Fits a quadratic surface constrained to f = 0 to a centered point cloud.

    Args:
      points: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      H - Mean curvature esitmate
      K - Gaussian curvature esitmate
    """

    batch_size, num_points, _ = points.size()

    # Reshape the points tensor for broadcasting
    points_tensor = points  # Shape: (num_of_batches, size_of_batch, 3)

    # Calculate L2 distances between points
    delta = points_tensor.unsqueeze(2) - points_tensor.unsqueeze(
        1)  # Shape: (num_of_batches, size_of_batch, size_of_batch, 3)
    squared_distances = torch.sum(delta ** 2, dim=-1)  # Shape: (num_of_batches, size_of_batch, size_of_batch)

    # Find the indices of k nearest neighbors
    _, indices = torch.topk(squared_distances, k+1, largest=False,
                            sorted=True)  # Shape: (num_of_batches, size_of_batch, k)

    # Gather neighbor points using advanced indexing
    neighbors = torch.gather(points_tensor.unsqueeze(2).expand(-1, -1, k+1, -1),
                             # Shape: (num_of_batches, size_of_batch, k, 3)
                             dim=1,
                             index=indices.unsqueeze(-1).expand(-1, -1, -1, 3))

    # Center the points around the mean
    # centroid = points[:,0,:]
    # centered_pcls = neighbors - centroid[:, np.newaxis, np.newaxis, :]

    # Reshape for efficient operations
    pcls_with_knn = neighbors.reshape(batch_size, num_points, k+1, 3)

    # Allocate coefficients (avoid loops)
    coeffs = np.zeros((batch_size, num_points, 5))

    # Each patch of KNN calculate its coefficients
    for i in range(batch_size):
        for j in range(num_points):
            # Extract current point and its kNN patch
            curr_point = pcls_with_knn[i, j, 0, :]
            knn_patch = pcls_with_knn[i, j, :, :]

            # Center kNN patch around current point
            centered_patch = knn_patch - curr_point

            # Design matrix for the patch (vectorized operations)
            X = np.c_[centered_patch[:, 0] ** 2, centered_patch[:, 1] ** 2,
                      centered_patch[:, 0] * centered_patch[:, 1],
            centered_patch[:, 0], centered_patch[:, 1]]

            # Solve least squares for the patch
            patch_coeffs = np.linalg.lstsq(X, centered_patch[:, 2], rcond=None)[0]

            # Store coefficients for this patch
            coeffs[i, j, :] = patch_coeffs
    # Extract coefficients for each batch
    a, b, c, d, e = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3], coeffs[..., 4]

    # Compute Gaussian curvature (K) and Mean curvature (H)
    K = (4 * (a * b) - (c ** 2)) / (1 + d ** 2 + e ** 2)
    H = (2 * a * (1 + c ** 2) - 2 * d * e * c + 2 * b * (1 + d ** 2)) / (((d ** 2) + (e ** 2) + 1) ** 1.5)

    return H, K



def find_rotation_translation_with_hk_as_color(src, H_src, K_src, target, H_target, K_target, threshold=0.5):
    batch_size, points, channels = src.shape

    all_h = np.concatenate((H_src, H_target), axis=0)
    all_k = np.concatenate((K_src, K_target), axis=0)

    normalized_h_values = _normalize_dimension(all_h)
    normalized_k_values = _normalize_dimension(all_k)

    source_h_normalized = normalized_h_values[:batch_size]
    target_h_normalized = normalized_h_values[batch_size:]
    source_k_normalized = normalized_k_values[:batch_size]
    target_k_normalized = normalized_k_values[batch_size:]

    source_colors = np.stack((source_h_normalized, source_k_normalized, 0.5*(source_h_normalized+source_k_normalized)), axis=-1)
    target_colors = np.stack((target_h_normalized, target_k_normalized, 0.5*(target_h_normalized+target_k_normalized)), axis=-1)
    # source_colors = np.stack((np.zeros_like(source_k_normalized), np.zeros_like(source_k_normalized), np.zeros_like(source_k_normalized)), axis=-1)
    # target_colors = np.stack(( np.zeros_like(target_k_normalized),  np.zeros_like(target_k_normalized), np.zeros_like(target_k_normalized)), axis=-1)


    source_clouds = []
    target_clouds = []

    for i in range(batch_size):
        top_src_H_indices = np.argpartition(source_h_normalized, -50, axis=None)[-50:]
        bottom_src_H_indices = np.argpartition(source_h_normalized, 50, axis=None)[:50]
        top_src_K_indices = np.argpartition(source_k_normalized, -50, axis=None)[-50:]
        bottom_src_K_indices = np.argpartition(source_k_normalized, 50, axis=None)[:50]
        source_combined_array = np.concatenate((top_src_H_indices, bottom_src_H_indices, top_src_K_indices, bottom_src_K_indices))

        # Remove duplicates
        source_unique_combined_array = np.unique(source_combined_array)


        top_target_H_indices = np.argpartition(target_h_normalized, -50, axis=None)[-50:]
        bottom_target_H_indices = np.argpartition(target_h_normalized, 50, axis=None)[:50]
        top_target_K_indices = np.argpartition(target_k_normalized, -50, axis=None)[-50:]
        bottom_target_K_indices = np.argpartition(target_k_normalized, 50, axis=None)[:50]
        target_combined_array = np.concatenate(
            (top_target_H_indices, bottom_target_H_indices, top_target_K_indices, bottom_target_K_indices))

        # Remove duplicates
        target_unique_combined_array = np.unique(target_combined_array)

        source_cloud = o3d.geometry.PointCloud()
        # above_threshold_mask = np.any(np.abs(source_colors[i]-0.5) > 0.4765, axis=-1)
        # indices_src = np.argwhere(above_threshold_mask)
        source_cloud.points = o3d.utility.Vector3dVector(src[i,(source_unique_combined_array[:,0]), :])
        source_cloud.colors = o3d.utility.Vector3dVector(source_colors[i,(source_unique_combined_array[:,0]), :])
        source_cloud.estimate_normals()
        source_clouds.append(source_cloud)

        target_cloud = o3d.geometry.PointCloud()
        # above_threshold_mask = np.any(np.abs(target_colors[i]-0.5) > 0.4765, axis=-1)
        # indices_tar = np.argwhere(above_threshold_mask)
        target_cloud.points = o3d.utility.Vector3dVector(target[i,(target_unique_combined_array[:,0]), :])
        target_cloud.colors = o3d.utility.Vector3dVector(target_colors[i,(target_unique_combined_array[:,0]), :])
        target_cloud.estimate_normals()
        target_clouds.append(target_cloud)

    R_list = []
    t_list = []

    for i in range(batch_size):
        try:
            reg_p2p = o3d.pipelines.registration.registration_colored_icp(
                source_clouds[i], target_clouds[i], 0.5,  np.eye(4))

        except RuntimeError as e:
            # o3d.visualization.draw_geometries([source_clouds[i]],
            #                                   zoom=0.3412,
            #                                   front=[0.4257, -0.2125, -0.8795],
            #                                   lookat=[2.6172, 2.0475, 1.532],
            #                                   up=[-0.0694, -0.9768, 0.2024])
            # o3d.visualization.draw_geometries([target_clouds[i]],
            #                                   zoom=0.3412,
            #                                   front=[0.4257, -0.2125, -0.8795],
            #                                   lookat=[2.6172, 2.0475, 1.532],
            #                                   up=[-0.0694, -0.9768, 0.2024])
            # fig = plot_registration( np.asarray(source_clouds[i].points), np.asarray(target_clouds[i].points))
            # fig.show()
            print(f'bummer')
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_clouds[i], target_clouds[i], 0.5, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

        R_list.append(reg_p2p.transformation[:3, :3])
        t_list.append(reg_p2p.transformation[:3, 3])
    R = np.stack(R_list, axis=0)
    t = np.stack(t_list, axis=0)

    return R, t

def _normalize_dimension(values):
    # Normalize values to the range [0, 1]
    min_val = np.min(values)
    max_val = np.max(values)
    normalized_values = (values - min_val) / (max_val - min_val)
    return normalized_values


def train_one_epoch(args, train_loader):
    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    eulers_ab = []

    for src, target, rotation_ab, translation_ab, euler_ab in tqdm(train_loader):
        batch_size = src.size(0)
        num_examples += batch_size
        src = src.permute(0,2,1)
        target = target.permute(0,2,1)
        H_src, K_src = fit_surface_quadratic_constrained(src)
        H_target, K_target = fit_surface_quadratic_constrained(target)
        R, t =find_rotation_translation_with_hk_as_color(src, H_src, K_src, target, H_target, K_target)
        # transformed_src = transform_point_cloud(src.permute(0,2,1), torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32))
        # transformed_src = transformed_src.permute(0,2,1)
        # fig = plot_registration(transformed_src[0].cpu().numpy(), target[0].cpu().numpy())
        #
        # # Show the plot
        # fig.show()
        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(R)
        translations_ab_pred.append(t)
        eulers_ab.append(euler_ab.numpy())

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)

    return rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab

def train(args, train_loader):
        train_rotations_ab, train_translations_ab, train_rotations_ab_pred, train_translations_ab_pred, \
        train_eulers_ab = train_one_epoch(args, train_loader)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))
        print('==TRAIN==')
        print('rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (train_r_mse_ab, train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
        return (train_r_mse_ab, train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab)
def plot_registration(transformed_src, target):
    fig = go.Figure()

    # Create a scatter plot trace for the transformed source point cloud
    fig.add_trace(go.Scatter3d(
        x=transformed_src[:, 0],
        y=transformed_src[:, 1],
        z=transformed_src[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.8
        ),
        name='Transformed Source'
    ))

    # Create a scatter plot trace for the target point cloud
    fig.add_trace(go.Scatter3d(
        x=target[:, 0],
        y=target[:, 1],
        z=target[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.8
        ),
        name='Target'
    ))

    # Update layout of the plot
    fig.update_layout(
        title='Predicted Registration',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    return fig


def initParser():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')

    ######################## Network Parameters ########################
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')

    ######################## Model Parameters ########################
    parser.add_argument('--alpha_factor', type=float, default=4)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--DeepBBS_pp', dest='DeepBBS_pp', action='store_true')
    parser.add_argument('--DeepBBS', dest='DeepBBS_pp', action='store_false')
    parser.set_defaults(DeepBBS_pp=True)

    ######################## Training Parameters ########################
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoint_dir', type=str, default='')

    ######################## Dataset Parameters ########################
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--different_pc', type=bool, default=False)
    parser.add_argument('--n_subsampled_points', type=int, default=1024, metavar='N',
                        help='Num of subsampled points to use')

    ######################## Testing Parameters ########################
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--deep_min_diff', type=float, default=0.4, metavar='N')
    parser.add_argument('--deep_max_iter', type=int, default=5, metavar='N')
    parser.add_argument('--spatial_min_diff', type=float, default=0.01, metavar='N')
    parser.add_argument('--spatial_min_iter', type=int, default=30, metavar='N')
    parser.add_argument('--spatial_max_iter', type=int, default=45, metavar='N')

    parser.add_argument('--use_wandb', type=int, default=1, metavar='N')
    parser.add_argument('--use_fpfh', type=int, default=0, metavar='N')
    parser.add_argument('--use_persistent_homology', type=int, default=0, metavar='N')
    parser.add_argument('--use_density', type=int, default=0, metavar='N')
    parser.add_argument('--use_inverse_density', type=int, default=0, metavar='N')
    parser.add_argument('--use_eigens', type=int, default=0, metavar='N')
    parser.add_argument('--zero_padding', type=int, default=0, metavar='N')
    parser.add_argument('--make_feat_random', type=int, default=0, metavar='N')
    parser.add_argument('--add_before_transformer', type=int, default=0, metavar='N')
    parser.add_argument('--add_after_transformer', type=int, default=0, metavar='N')
    parser.add_argument('--SVDHEAD_emb_dims', type=int, default=0, metavar='N')
    parser.add_argument('--divide_data', type=int, default=128, metavar='N')
    parser.add_argument('--sigma_factor', type=int, default=1, metavar='N')
    parser.add_argument('--clip_factor', type=int, default=1, metavar='N')

    args = parser.parse_args()
    return args
def main():
    args = initParser()
    train_loader = DataLoader(
        ModelNet40(num_points=args.num_points,sigma_factor=args.sigma_factor,clip_factor=args.clip_factor, divide_data=args.divide_data, num_subsampled_points=args.n_subsampled_points, partition='train',
                   gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor,
                   random_point_order=True, different_pc=args.different_pc), batch_size=args.batch_size,
                   shuffle=True, drop_last=True)
    test_loader = DataLoader(
        ModelNet40(num_points=args.num_points,sigma_factor=args.sigma_factor,clip_factor=args.clip_factor, divide_data=args.divide_data, num_subsampled_points=args.n_subsampled_points, partition='test',
                   gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor,
                   random_point_order=True, different_pc=args.different_pc), batch_size=1,
                   shuffle=False, drop_last=False)

    train_r_mse_ab_list = []
    train_r_rmse_ab_list = []
    train_r_mae_ab_list = []
    train_t_mse_ab_list = []
    train_t_rmse_ab_list = []
    train_t_mae_ab_list = []
    for i in range (20):
        print(f'traing iteration: {i}')
        (train_r_mse_ab, train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab) = train(args, test_loader)
        train_r_mse_ab_list.append(train_r_mse_ab)
        train_r_rmse_ab_list.append(train_r_rmse_ab)
        train_r_mae_ab_list.append(train_r_mae_ab)
        train_t_mse_ab_list.append(train_t_mse_ab)
        train_t_rmse_ab_list.append(train_t_rmse_ab)
        train_t_mae_ab_list.append(train_t_mae_ab)
    print(f'train_r_mse_ab Mean: {np.mean(train_r_mse_ab_list)}')
    print(f'train_r_mse_ab Median: {np.median(train_r_mse_ab_list)}')
    print(f'train_r_rmse_ab_list Mean: {np.mean(train_r_rmse_ab_list)}')
    print(f'train_r_rmse_ab_list Median: {np.median(train_r_rmse_ab_list)}')
    print(f'train_r_mae_ab_list Mean: {np.mean(train_r_mae_ab_list)}')
    print(f'train_r_mae_ab_list Median: {np.median(train_r_mae_ab_list)}')

    print(f'train_t_mse_ab_list Mean: {np.mean(train_t_mse_ab_list)}')
    print(f'train_t_mse_ab_list Median: {np.median(train_t_mse_ab_list)}')
    print(f'train_t_rmse_ab_list Mean: {np.mean(train_t_rmse_ab_list)}')
    print(f'train_t_rmse_ab_list Median: {np.median(train_t_rmse_ab_list)}')
    print(f'train_t_mae_ab_list Mean: {np.mean(train_t_mae_ab_list)}')
    print(f'train_t_mae_ab_list Median: {np.median(train_t_mae_ab_list)}')


if __name__ == '__main__':
    main()
