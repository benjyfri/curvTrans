import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import sys

sys.path.append('../../')
# import script.common as cm
# # from ThreeDMatch.Test.tools import get_pcd, get_keypts
from sklearn.neighbors import KDTree
import importlib
import open3d
import numpy as np


from models import shapeClassifier
from plotting_functions import *
import argparse

def calcDist(src_knn_pcl):
    pcl = src_knn_pcl[0].permute(1,2,0)
    median_of_median_axis = torch.median(torch.median((torch.max(pcl, dim=1)[0] - torch.min(pcl, dim=1)[0]), dim=0)[0])
    scale = 1 / median_of_median_axis
    return scale

def create_emb(input,model, scales=[1,3,5,7,9,11,13,15]):
    cur_input = input[:, :21, :]
    neighbors_centered = cur_input.permute(0, 2, 1).unsqueeze(2)
    scaling_factor = calcDist(neighbors_centered).item()
    src_knn_pcl = scaling_factor * neighbors_centered
    emb = model(src_knn_pcl).squeeze()
    pcls=[cur_input]
    # multiscale embeddings
    for scale in scales[1:]:
        cur_input = input[:,:21*scale,:]
        cur_input = farthest_point_sampling(cur_input, k=21)
        pcls.append(cur_input)
        neighbors_centered = cur_input.permute(0, 2, 1).unsqueeze(2)
        scaling_factor = calcDist(neighbors_centered).item()
        src_knn_pcl = scaling_factor * neighbors_centered
        cur_emb = model(src_knn_pcl).squeeze()
        emb = torch.cat((emb, cur_emb), dim=1)
    return emb

def get_keypts(keyptspath, filename):
    keypts = np.fromfile(os.path.join(keyptspath, filename + '.keypts.bin'), dtype=np.float32)
    num_keypts = int(keypts[0])
    keypts = keypts[1:].reshape([num_keypts, 3])
    return keypts
def get_pcd(pcdpath, filename):
    return open3d.io.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))
def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def build_patch_input(pcd, keypts, vicinity=0.3, num_points_per_patch=2048):
    refer_pts = keypts.astype(np.float32)
    pts = np.array(pcd.points).astype(np.float32)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    # ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)
    _, ind_local = tree.query(refer_pts[:, 0:3], k=num_points_per_patch)
    local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
    for i in range(num_patches):
        local_neighbors = pts[ind_local[i], :]
        # if local_neighbors.shape[0] >= num_points_per_patch:
        #     temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
        #     local_neighbors = local_neighbors[temp]
        #     # local_neighbors[-1, :] = refer_pts[i, :]
        #     local_neighbors[0, :] = refer_pts[i, :]
        # else:
        #     fix_idx = np.asarray(range(local_neighbors.shape[0]))
        #     while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
        #         fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
        #     random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
        #                                   replace=False)
        #     choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        #     local_neighbors = local_neighbors[choice_idx]
        #     # local_neighbors[-1, :] = refer_pts[i, :]
        #     local_neighbors[0, :] = refer_pts[i, :]
        local_neighbors[0, :] = refer_pts[i, :]
        local_neighbors = local_neighbors - refer_pts[i, :]
        # distances = np.linalg.norm(local_neighbors, axis=1)
        # sorted_indices = np.argsort(distances)
        # local_neighbors = local_neighbors[sorted_indices]
        local_patches[i] = local_neighbors
    return local_patches

def prepare_patch(pcdpath, filename, keyptspath, trans_matrix):
    pcd = get_pcd(pcdpath, filename)
    keypts = get_keypts(keyptspath, filename)
    # load D3Feat keypts
    if is_D3Feat_keypts:
        keypts_path = './D3Feat_contralo-54-pred/keypoints/' + pcdpath.split('/')[-2] + '/' + filename + '.npy'
        keypts = np.load(keypts_path)
        keypts = keypts[-5000:, :]
    if is_rotate_dataset:
        # Add arbitrary rotation
        # rotate terminal frament with an arbitrary angle around the z-axis
        angles_3d = np.random.rand(3) * np.pi * 2
        R = angles2rotation_matrix(angles_3d)
        T = np.identity(4)
        T[:3, :3] = R
        pcd.transform(T)
        keypts_pcd = make_open3d_point_cloud(keypts)
        keypts_pcd.transform(T)
        keypts = np.array(keypts_pcd.points)
        trans_matrix.append(T)

    local_patches = build_patch_input(pcd, keypts, num_points_per_patch=301)  # [num_keypts, 1024, 4]
    return local_patches


def generate_descriptor(model, desc_name, pcdpath, keyptspath, descpath):
    model.eval()
    num_frag = len(os.listdir(pcdpath))
    num_desc = len(os.listdir(descpath))
    trans_matrix = []
    if num_frag == num_desc:
        print("Descriptor already prepared.")
        return
    for j in range(num_frag):
        local_patches = prepare_patch(pcdpath, 'cloud_bin_' + str(j), keyptspath, trans_matrix)
        input_ = torch.tensor(local_patches.astype(np.float32))
        B = input_.shape[0]
        input_ = input_.cuda()
        model = model.cuda()
        # calculate descriptors
        desc_list = []
        start_time = time.time()
        desc_len = 32
        desc_len = 150
        # desc_len = 75
        step_size = 100
        step_size = B
        iter_num = int(np.ceil(B / step_size))
        for k in range(iter_num):
            if k == iter_num - 1:
                input_ = (input_[k * step_size:, :, :])
            else:
                input_ = (input_[k * step_size: (k + 1) * step_size, :, :])

            desc = create_emb(input_,model, scales=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

            desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
            del desc
        step_time = time.time() - start_time
        print(f'Finish {B} descriptors spend {step_time:.4f}s')
        desc = np.concatenate(desc_list, 0).reshape([B, desc_len])
        np.save(descpath + 'cloud_bin_' + str(j) + f".desc.{desc_name}.bin", desc.astype(np.float32))
    if is_rotate_dataset:
        scene_name = pcdpath.split('/')[-2]
        all_trans_matrix[scene_name] = trans_matrix
def configArgsPCT():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--wandb_proj', type=str, default='MLP-Contrastive-Ablation', metavar='N',
                        help='Name of the wandb project name to upload the run data')
    parser.add_argument('--exp_name', type=str, default='c_cntr_sep_1_std01_long', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--use_wandb', type=int, default=1, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--contr_margin', type=float, default=1.0, metavar='N',
                        help='margin used for contrastive loss')
    parser.add_argument('--use_lap_reorder', type=int, default=1, metavar='N',
                        help='reorder points by laplacian order ')
    parser.add_argument('--lap_eigenvalues_dim', type=int, default=5, metavar='N',
                        help='use eigenvalues in as input')
    parser.add_argument('--use_second_deg', type=int, default=1, metavar='N',
                        help='use second degree embedding ')
    parser.add_argument('--lpe_normalize', type=int, default=1, metavar='N',
                        help='use normalized laplacian')
    parser.add_argument('--std_dev', type=float, default=0.05, metavar='N',
                        help='amount of noise to add to data')
    parser.add_argument('--max_curve_diff', type=float, default=2, metavar='N',
                        help='max difference in curvature for contrastive loss')
    parser.add_argument('--min_curve_diff', type=float, default=0.05, metavar='N',
                        help='min difference in curvature for contrastive loss')
    parser.add_argument('--clip', type=float, default=0.25, metavar='N',
                        help='clip noise')
    parser.add_argument('--contr_loss_weight', type=float, default=0.1, metavar='N',
                        help='weight of contrastive loss')
    parser.add_argument('--lpe_dim', type=int, default=0, metavar='N',
                        help='laplacian positional encoding amount of eigens to take')
    parser.add_argument('--use_xyz', type=int, default=1, metavar='N',
                        help='use xyz coordinates as part of input')
    parser.add_argument('--classification', type=int, default=1, metavar='N',
                        help='use classification loss')
    parser.add_argument('--rotate_data', type=int, default=1, metavar='N',
                        help='use rotated data')
    parser.add_argument('--cube', type=int, default=0, metavar='N',
                        help='Normalize data into 1 cube')
    parser.add_argument('--num_neurons_per_layer', type=int, default=64, metavar='N',
                        help='how many neurons per layer to use')
    parser.add_argument('--num_mlp_layers', type=int, default=5, metavar='N',
                        help='how many mlp layers to use')
    parser.add_argument('--output_dim', type=int, default=10, metavar='N',
                        help='how many labels are used')
    parser.add_argument('--lr_jumps', type=int, default=15, metavar='N',
                        help='Lower lr *0.1 every amount of jumps')
    parser.add_argument('--sampled_points', type=int, default=20, metavar='N',
                        help='How many points where sampled around centroid')
    args = parser.parse_args()
    return args


import torch

def get_k_nearest_neighbors_diff_pcls_torch(pcl_src, pcl_interest, k):
    """
    Returns the k nearest neighbors for each point in the point cloud for a batch of point clouds.

    Args:
        pcl_src (torch.Tensor): Source point cloud of shape (batch_size, pcl_size_src, 3)
        pcl_interest (torch.Tensor): Interest point cloud of shape (batch_size, pcl_size_interest, 3)
        k (int): Number of nearest neighbors to return

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 3, pcl_size_interest, k) containing the k nearest neighbors for each point
    """
    # Calculate distances between points in pcl_interest and pcl_src
    distances = torch.cdist(pcl_interest, pcl_src)  # Shape: (batch_size, pcl_size_interest, pcl_size_src)

    # Get indices of the k nearest neighbors
    _, indices = torch.topk(distances, k, dim=-1, largest=False)  # Shape: (batch_size, pcl_size_interest, k)

    # Gather k nearest neighbors from pcl_src using indices
    neighbors = torch.gather(
        pcl_src.unsqueeze(1).expand(-1, pcl_interest.size(1), -1, -1),  # Shape: (batch_size, pcl_size_interest, pcl_size_src, 3)
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, 3)  # Shape: (batch_size, pcl_size_interest, k, 3)
    )  # Resulting shape: (batch_size, pcl_size_interest, k, 3)

    # Compute differences (center neighbors around pcl_interest points)
    neighbors_centered = (neighbors - pcl_interest.unsqueeze(2))  # Shape: (batch_size, pcl_size_interest, k, 3)

    # Ensure the first neighbor is at the origin (if it's not already [0,0,0])
    zeros = torch.zeros((1, 1, 1, 3), dtype=neighbors.dtype, device=neighbors.device)
    neighbors_centered[:, :, 0, :] = torch.where(
        torch.all(neighbors_centered[:, :, 0, :] == 0, dim=-1, keepdim=True),
        zeros,
        neighbors_centered[:, :, 0, :]
    )

    # Transpose to match the desired output shape: (batch_size, 3, pcl_size_interest, k)
    return neighbors_centered.permute(0, 3, 1, 2)


def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Performs farthest point sampling (FPS) on a batch of 3D point clouds.

    Args:
        points (torch.Tensor): A tensor of shape (batch_size, num_points, 3) representing the 3D point clouds.
        k (int): The number of points to sample.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, k, 3) representing the sampled 3D point clouds.
    """
    batch_size, num_points, _ = points.shape

    # Initialize tensors
    device = points.device
    sampled_indices = torch.zeros((batch_size, k), dtype=torch.long, device=device)
    distances = torch.full((batch_size, num_points), float('inf'), device=device)

    # Start with the first point in each batch
    sampled_indices[:, 0] = 0
    sampled_points = points[:, 0, :].unsqueeze(1)  # (batch_size, 1, 3)

    for i in range(1, k):
        # Compute distances from the last sampled point to all points
        last_sampled_point = sampled_points[:, -1, :].unsqueeze(1)  # (batch_size, 1, 3)
        dist_to_last = torch.sum((points - last_sampled_point) ** 2, dim=-1)  # (batch_size, num_points)

        # Update minimum distances
        distances = torch.min(distances, dist_to_last)

        # Select the farthest point for each batch
        farthest_indices = torch.argmax(distances, dim=-1)  # (batch_size,)
        sampled_indices[:, i] = farthest_indices

        # Gather the new sampled point
        new_sampled_point = points[torch.arange(batch_size, device=device), farthest_indices]  # (batch_size, 3)
        sampled_points = torch.cat([sampled_points, new_sampled_point.unsqueeze(1)], dim=1)  # (batch_size, i+1, 3)

    return sampled_points

if __name__ == '__main__':
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

    args = configArgsPCT()
    print(args.exp_name)
    # experiment_id = time.strftime('%m%d%H%M')
    experiment_id = args.exp_name
    model_str = experiment_id  # sys.argv[1]
    if not os.path.exists(f"SpinNet_desc_{model_str}/"):
        os.mkdir(f"SpinNet_desc_{model_str}")

    # # dynamically load the model
    # module_file_path = '../model.py'
    # shutil.copy2(os.path.join('.', '../../network/SpinNet.py'), module_file_path)
    # module_name = ''
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # model = module.Descriptor_Net(0.30, 9, 80, 40, 0.04, 30, '3DMatch')
    # model = nn.DataParallel(model, device_ids=[0])
    # model.load_state_dict(torch.load('../../pre-trained_models/3DMatch_best.pkl'))

    model = shapeClassifier(args)
    model.load_state_dict(torch.load(f'.././models_weights/{args.exp_name}.pt'))
    model.to("cuda")
    model.eval()

    all_trans_matrix = {}
    is_rotate_dataset = False
    is_D3Feat_keypts = False
    for scene in scene_list:
        pcdpath = f"./../data/3DMatch/fragments/{scene}/"
        interpath = f"./../data/3DMatch/intermediate-files-real/{scene}/"
        keyptspath = interpath
        descpath = os.path.join('.', f"SpinNet_desc_{model_str}/{scene}/")
        if not os.path.exists(descpath):
            os.makedirs(descpath)
        start_time = time.time()
        print(f"Begin Processing {scene}")
        generate_descriptor(model, desc_name='SpinNet', pcdpath=pcdpath, keyptspath=keyptspath, descpath=descpath)
        print(f"Finish in {time.time() - start_time}s")
    if is_rotate_dataset:
        np.save(f"trans_matrix", all_trans_matrix)
