import os
# scaling_factors_hardcoded = {
#     'gazebo_summer': [10.32220458984375, 6.417322158813477, 5.009559631347656, 4.237317085266113, 3.735275983810425, 3.3757312297821045, 3.103226900100708, 2.889331579208374, 2.70857834815979, 2.5615227222442627, 2.431459426879883, 2.3200185298919678, 2.2235708236694336, 2.1366283893585205, 2.11397385597229]
# , 'gazebo_winter':[13.32024097442627, 7.490845203399658, 5.6744232177734375, 4.707010269165039, 4.0800700187683105, 3.6573755741119385, 3.325899839401245, 3.0692789554595947, 2.8636817932128906, 2.6887903213500977, 2.5416977405548096, 2.418029308319092, 2.3094842433929443, 2.2159509658813477, 2.194293975830078]
# ,
#     'wood_autmn':[12.897363662719727, 7.443971633911133, 5.647600173950195, 4.709465503692627, 4.106546878814697, 3.671670913696289, 3.34374737739563, 3.08815598487854, 2.8795852661132812, 2.7065727710723877, 2.557424545288086, 2.4334781169891357, 2.3227617740631104, 2.225461006164551, 2.2020461559295654]
# ,
#     'wood_summer':[12.668914794921875, 7.33339786529541, 5.562127113342285, 4.632084846496582, 4.031224727630615, 3.610067844390869, 3.291415214538574, 3.0450525283813477, 2.843318462371826, 2.675147771835327, 2.5318946838378906, 2.406442403793335, 2.296908378601074, 2.2011005878448486, 2.178225517272949]
# }
scaling_factors_hardcoded = {
    'gazebo_summer': [8.48, 5.55, 4.41, 3.77, 3.34, 3.03, 2.8, 2.61, 2.45, 2.32, 2.21, 2.12, 2.03, 1.95, 1.93],
    'gazebo_winter':[6.37, 4.02, 3.14, 2.65, 2.33, 2.1, 1.92, 1.79, 1.67, 1.58, 1.5, 1.44, 1.38, 1.32, 1.31],
    'wood_autmn':[6.89, 4.41, 3.47, 2.95, 2.6, 2.35, 2.17, 2.02, 1.9, 1.8, 1.71, 1.63, 1.56, 1.51, 1.49],
    'wood_summer':[6.96, 4.44, 3.49, 2.96, 2.61, 2.36, 2.18, 2.03, 1.91, 1.81, 1.72, 1.65, 1.58, 1.53, 1.51]
}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import glob
import sys

sys.path.append('./../')
# import script.common as cm
import open3d
from tools import get_pcd, get_ETH_keypts, get_desc, loadlog
from sklearn.neighbors import KDTree
import importlib
from plotting_functions import *
import argparse

# def calcDist(src_knn_pcl):
#     pcl = src_knn_pcl[0].permute(1,2,0)
#     median_of_median_axis = torch.median(torch.median((torch.max(pcl, dim=1)[0] - torch.min(pcl, dim=1)[0]), dim=0)[0])
#     scale = 1 / median_of_median_axis
#     return scale
def calcDist(local_patches):
    median_of_median_axis = torch.median(torch.median((torch.max(local_patches, dim=1)[0] - torch.min(local_patches, dim=1)[0]), dim=0)[0])
    scale = 1 / median_of_median_axis
    return scale

def create_emb(input, model, scaling_factors, scales=[1, 3, 5, 7, 9, 11, 13, 15]):
    emb_list = []  # Store embeddings before concatenation

    for scale, scaling_factor in zip(scales, scaling_factors):
        cur_input = input[:, :21 * scale, :]
        if scale>1:
            cur_input = farthest_point_sampling(cur_input, k=21)  # Ensure function exists
        # cur_scaling_factor = calcDist(cur_input)
        neighbors_centered = cur_input.permute(0, 2, 1).unsqueeze(2)
        src_knn_pcl = scaling_factor * neighbors_centered
        # src_knn_pcl = cur_scaling_factor * neighbors_centered

        cur_emb = model(src_knn_pcl).squeeze()  # Ensure squeezing works as expected
        emb_list.append(cur_emb)  # Store embeddings instead of direct concatenation

    emb = torch.cat(emb_list, dim=1)  # Concatenate after loop
    return emb
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
        # else:
        #     fix_idx = np.asarray(range(local_neighbors.shape[0]))
        #     while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
        #         fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
        #     random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
        #                                   replace=False)
        #     choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        #     local_neighbors = local_neighbors[choice_idx]
        #     # local_neighbors[-1, :] = refer_pts[i, :]
        local_neighbors[0, :] = refer_pts[i, :]
        local_neighbors = local_neighbors - refer_pts[i, :]
        # distances = np.linalg.norm(local_neighbors, axis=1)
        # sorted_indices = np.argsort(distances)
        # local_neighbors = local_neighbors[sorted_indices]
        local_patches[i] = local_neighbors
    return local_patches



def prepare_patch(pcdpath, filename, keyptspath, trans_matrix):
    pcd = get_pcd(pcdpath, filename)
    keypts = get_ETH_keypts(pcd, keyptspath, filename)
    # if is_rotate_dataset:
    #     # Add arbitrary rotation
    #     # rotate terminal frament with an arbitrary angle around the z-axis
    #     angles_3d = np.random.rand(3) * np.pi * 2
    #     R = angles2rotation_matrix(angles_3d)
    #     T = np.identity(4)
    #     T[:3, :3] = R
    #     pcd.transform(T)
    #     keypts_pcd = make_open3d_point_cloud(keypts)
    #     keypts_pcd.transform(T)
    #     keypts = np.array(keypts_pcd.points)
    #     trans_matrix.append(T)
    # local_patches = build_patch_input(pcd, keypts, des_r)  # [num_keypts, 1024, 4]
    local_patches = build_patch_input(pcd, keypts, num_points_per_patch=301) # [num_keypts, 1024, 4]
    return keypts,local_patches
def find_scaling_factor(model, desc_name, pcdpath, keyptspath, descpath, output_dim):
    fragments = glob.glob(pcdpath + '*.ply')
    num_frag = len(fragments)
    all_local_pathces = []
    yay = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"10":[],"11":[],"12":[],"13":[],"14":[],"15":[]}
    for j in range(num_frag):
        keypts, local_patches =  prepare_patch(pcdpath, 'Hokuyo_' + str(j), keyptspath, None)
        local_patches=torch.tensor(local_patches.astype(np.float32))
        for scale in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            cur_input = local_patches[:, :21 * scale, :]
            cur_input = farthest_point_sampling(cur_input, k=21)
            yay[str(scale)].append(cur_input)
    # cur_all_input = torch.cat(all_local_pathces, dim=0)
    a = [calcDist((torch.cat(yay[str(scale)], dim=0))).item() for scale in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    # neighbors_centered = cur_all_input.permute(0, 2, 1).unsqueeze(2)
    # scaling_factor = calcDist(neighbors_centered).item()
    print(a)
    return 1

def generate_descriptor(model, desc_name, pcdpath, keyptspath, descpath, output_dim):
    scaling_factors = scaling_factors_hardcoded[str(str.split(pcdpath, r'/')[-2])]
    model.eval()
    fragments = glob.glob(pcdpath + '*.ply')
    num_frag = len(fragments)
    num_desc = len(os.listdir(descpath))
    trans_matrix = []
    if num_frag == num_desc:
        print("Descriptor already prepared.")
        return
    for j in range(num_frag):
        keypts,local_patches = prepare_patch(pcdpath, 'Hokuyo_' + str(j), keyptspath, trans_matrix)
        input_ = torch.tensor(local_patches.astype(np.float32))
        B = input_.shape[0]
        input_ = input_.cuda()
        model = model.cuda()
        # calculate descriptors
        desc_list = []
        start_time = time.time()
        desc_len = 32
        desc_len = 15 * output_dim
        # desc_len = 75
        step_size = 100
        step_size = B
        iter_num = int(np.ceil(B / step_size))
        for k in range(iter_num):
            if k == iter_num - 1:
                input_ = (input_[k * step_size:, :, :])
            else:
                input_ = (input_[k * step_size: (k + 1) * step_size, :, :])
            desc = create_emb(input_, model, scaling_factors=scaling_factors,scales=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            if torch.count_nonzero(torch.isnan(desc))>0:
                print(desc)
                raise Exception("Nan value occurred!!")
            desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
            del desc
        step_time = time.time() - start_time
        print(f'Finish {B} descriptors spend {step_time:.4f}s')
        desc = np.concatenate(desc_list, 0).reshape([B, desc_len])
        np.save(descpath + 'Hokuyo_' + str(j) + f".desc.{desc_name}.bin", desc.astype(np.float32))
    # if is_rotate_dataset:
    #     scene_name = pcdpath.split('/')[-2]
    #     all_trans_matrix[scene_name] = trans_matrix

def run_preparation_ETH():
    scene_list = [
        'gazebo_summer',
        'gazebo_winter',
        'wood_autmn',
        'wood_summer',
    ]
    model_names =['x_cntr0_std005_long','y_cntr0_std0_long','z_cntr0_std01_long','a_cntr1_std005_long','b_cntr1_std01_long','c_cntr_sep_1_std01_long']
    # model_names =['z_cntr0_std01_long','a_cntr1_std005_long','b_cntr1_std01_long','c_cntr_sep_1_std01_long']
    output_dims = [5,5,5,5,5,10]
    args = configArgsPCT()
    for model_name,output_dim in zip(model_names, output_dims):
        print(f'------------------------------------------')
        print(f'{model_name}, out_dim = {output_dim}')
        print(f'------------------------------------------')
        args.exp_name = model_name
        args.output_dim = output_dim
        print(args.exp_name)
        # experiment_id = time.strftime('%m%d%H%M')
        experiment_id = "ETH_"+args.exp_name
        model_str = experiment_id  # sys.argv[1]

        if not os.path.exists(f"SpinNet_desc_{model_str}/"):
            os.mkdir(f"SpinNet_desc_{model_str}")

        model = shapeClassifier(args)
        model.load_state_dict(torch.load(f'.././models_weights/{args.exp_name}.pt', weights_only=True))
        model.to("cuda")
        model.eval()
        # # dynamically load the model
        # module_file_path = './model.py'
        # shutil.copy2(os.path.join('.', './../network/SpinNet.py'), './model.py')
        # module_name = ''
        # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        # module = importlib.util.module_from_spec(module_spec)
        # module_spec.loader.exec_module(module)
        #
        # des_r = 0.8
        # model = module.Descriptor_Net(des_r, 9, 80, 40, 0.10, 30, '3DMatch')
        # model = nn.DataParallel(model, device_ids=[0])
        # model.load_state_dict(torch.load('./../pre-trained_models/3DMatch_best.pkl'))
        all_trans_matrix = {}
        is_rotate_dataset = False

        for scene in scene_list:
            pcdpath = f"./../data/ETH/{scene}/"
            interpath = f"./../data/ETH/{scene}/01_Keypoints/"
            keyptspath = interpath
            descpath = os.path.join('.', f"SpinNet_desc_{model_str}/{scene}/")
            if not os.path.exists(descpath):
                os.makedirs(descpath)
            start_time = time.time()
            print(f"Begin Processing {scene}")
            # x = find_scaling_factor(model, desc_name='SpinNet', pcdpath=pcdpath, keyptspath=keyptspath, descpath=descpath, output_dim=output_dim)
            # print(x)
            generate_descriptor(model, desc_name='SpinNet', pcdpath=pcdpath, keyptspath=keyptspath, descpath=descpath, output_dim=output_dim)
            print(f"Finish in {time.time() - start_time}s")
            if is_rotate_dataset:
                np.save(f"trans_matrix", all_trans_matrix)

if __name__ == '__main__':
    run_preparation_ETH()