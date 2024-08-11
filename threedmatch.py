"""Dataloader for 3DMatch dataset

Modified from Predator source code by Shengyu Huang:
  https://github.com/overlappredator/OverlapPredator/blob/main/datasets/indoor.py
"""
from typing import Union, Tuple
import open3d as o3d
import logging
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# from utils.se3_numpy import se3_init, se3_transform, se3_inv
import transforms as Transforms


# import numpy as np


def se3_init(rot, trans):
    pose = np.concatenate([rot, trans], axis=-1)
    return pose


def se3_cat(a, b):
    """Concatenates two SE3 transforms"""
    rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
    rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

    rot = rot_a @ rot_b
    trans = rot_a @ trans_b + trans_a
    dst = se3_init(rot, trans)
    return dst


def se3_inv(pose):
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)


def se3_transform(pose, xyz):
    """Apply rigid transformation to points

    Args:
        pose: ([B,] 3, 4)
        xyz: ([B,] N, 3)

    Returns:

    """

    assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

    return transformed

class ThreeDMatchDataset(Dataset):

    def __init__(self, cfg, phase, transforms=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            info_fname = f'datasets/3dmatch/{phase}_info.pkl'
            pairs_fname = f'{phase}_pairs-overlapmask.h5'
        else:
            info_fname = f'datasets/3dmatch/{phase}_{cfg.benchmark}_info.pkl'
            pairs_fname = f'{phase}_{cfg.benchmark}_pairs-overlapmask.h5'

        with open(info_fname, 'rb') as fid:
            self.infos = pickle.load(fid)

        self.base_dir = None
        if isinstance(cfg.root, str):
            if os.path.exists(f'{cfg.root}/train'):
                self.base_dir = cfg.root
        else:
            for r in cfg.root:
                if os.path.exists(f'{r}/train'):
                    self.base_dir = r
                break
        if self.base_dir is None:
            raise AssertionError(f'Dataset not found in {cfg.root}')
        else:
            self.logger.info(f'Loading data from {self.base_dir}')

        self.cfg = cfg

        if os.path.exists(os.path.join(self.base_dir, pairs_fname)):
            self.pairs_data = h5py.File(os.path.join(self.base_dir, pairs_fname), 'r')
        else:
            self.logger.warning(
                'Overlapping regions not precomputed. '
                'Run data_processing/compute_overlap_3dmatch.py to speed up data loading')
            self.pairs_data = None

        self.search_voxel_size = cfg.overlap_radius
        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self, item):

        # get transformation and point cloud
        pose = se3_init(self.infos['rot'][item], self.infos['trans'][item])  # transforms src to tgt
        pose_inv = se3_inv(pose)
        src_path = self.infos['src'][item]
        tgt_path = self.infos['tgt'][item]
        src_xyz = torch.load(os.path.join(self.base_dir, src_path))
        tgt_xyz = torch.load(os.path.join(self.base_dir, tgt_path))
        overlap_p = self.infos['overlap'][item]

        # Get overlap region
        if self.pairs_data is None:
            src_overlap_mask, tgt_overlap_mask, src_tgt_corr = compute_overlap(
                se3_transform(pose, src_xyz),
                tgt_xyz,
                self.search_voxel_size,
            )
        else:
            src_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/src_mask'])
            tgt_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/tgt_mask'])
            src_tgt_corr = np.asarray(self.pairs_data[f'pair_{item:06d}/src_tgt_corr'])

        data_pair = {
            'src_xyz': torch.from_numpy(src_xyz).float(),
            'tgt_xyz': torch.from_numpy(tgt_xyz).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(src_tgt_corr),  # indices
            'pose': torch.from_numpy(pose).float(),
            'idx': item,
            'src_path': src_path,
            'tgt_path': tgt_path,
            'overlap_p': overlap_p,
        }

        if self.transforms is not None:
            self.transforms(data_pair)  # Apply data augmentation

        return data_pair

def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes region of overlap between two point clouds.

    Args:
        src: Source point cloud, either a numpy array of shape (N, 3) or
          Open3D PointCloud object
        tgt: Target point cloud similar to src.
        search_voxel_size: Search radius

    Returns:
        has_corr_src: Whether each source point is in the overlap region
        has_corr_tgt: Whether each target point is in the overlap region
        src_tgt_corr: Indices of source to target correspondences
    """

    if isinstance(src, np.ndarray):
        src_pcd = to_o3d_pcd(src)
        src_xyz = src
    else:
        src_pcd = src
        src_xyz = np.asarray(src.points)

    if isinstance(tgt, np.ndarray):
        tgt_pcd = to_o3d_pcd(tgt)
        tgt_xyz = tgt
    else:
        tgt_pcd = tgt
        tgt_xyz = tgt.points

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches
    tgt_corr = np.full(tgt_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, search_voxel_size)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]
    src_corr = np.full(src_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, search_voxel_size)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(tgt_corr[src_corr] == np.arange(len(src_corr)),
                                        src_corr > 0)
    src_tgt_corr = np.stack([np.nonzero(src_corr_is_mutual)[0],
                             src_corr[src_corr_is_mutual]])

    has_corr_src = src_corr >= 0
    has_corr_tgt = tgt_corr >= 0

    return has_corr_src, has_corr_tgt, src_tgt_corr


def to_o3d_pcd(xyz, colors=None, normals=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

