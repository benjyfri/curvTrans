import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import os
import urllib.request
import zipfile
import shutil
import faiss  # Make sure you have FAISS installed

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://share.phys.ethz.ch/~gsg/pairwise_reg/modelnet40_ply_hdf5_2048.zip'
        zipfile_path = os.path.join(BASE_DIR, os.path.basename(www))

        # Download the file
        urllib.request.urlretrieve(www, zipfile_path)

        # Unzip the file
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # # Move the extracted folder to the data directory
        # extracted_folder = os.path.splitext(os.path.basename(www))[0]
        # extracted_folder_path = os.path.join(BASE_DIR, extracted_folder)
        # shutil.move(extracted_folder_path, DATA_DIR)

        # Remove the downloaded zip file
        os.remove(zipfile_path)


def load_data(partition, divide_data=1):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
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
    # return all_data[:8,:,:], all_label[:8,:]


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768, random_spherical=False):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]

    # Convert point clouds to float32
    pointcloud1 = pointcloud1.astype('float32')
    pointcloud2 = pointcloud2.astype('float32')

    # Create an index for pointcloud1 and add points to it
    index1 = faiss.IndexFlatL2(pointcloud1.shape[1])  # L2 distance
    index1.add(pointcloud1)

    # Create an index for pointcloud2 and add points to it
    index2 = faiss.IndexFlatL2(pointcloud2.shape[1])  # L2 distance
    index2.add(pointcloud2)

    if random_spherical:
        random_p1 = np.random.randn(1, 3)
        random_p1 /= np.linalg.norm(random_p1, axis=1)
        random_p1 *= 500
    else:
        random_p1 = pointcloud1[np.random.randint(0, num_points, size=(1)), :]

    # Perform nearest neighbor search for random_p1 in pointcloud1
    _, idx1 = index1.search(random_p1, num_subsampled_points)

    if random_spherical:
        random_p2 = np.random.randn(1, 3)
        random_p2 /= np.linalg.norm(random_p2, axis=1)
        random_p2 *= 500
    else:
        random_p2 = pointcloud2[np.random.randint(0, num_points, size=(1)), :]

    # Perform nearest neighbor search for random_p2 in pointcloud2
    _, idx2 = index2.search(random_p2, num_subsampled_points)

    return pointcloud1[idx1[0], :].T, pointcloud2[idx2[0], :].T

class ModelNet40(Dataset):
    def __init__(self, num_points,sigma_factor=1,clip_factor=1, divide_data=1, num_subsampled_points=768, partition='train', gaussian_noise=False, unseen=False, factor=4, src_unbalance=False, tgt_unbalance=False, random_point_order=True, different_pc=False):
        self.data, self.label = load_data(partition, divide_data)
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        if different_pc:
            self.num_points *= 2
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        self.src_unbalance = src_unbalance
        self.tgt_unbalance = tgt_unbalance
        self.random_point_order = random_point_order
        self.different_pc = different_pc
        self.sigma_factor = sigma_factor
        self.clip_factor = clip_factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        # start_time = time.time()
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        if self.different_pc:
            ind = np.random.permutation(self.num_points)
            pointcloud1 = pointcloud1[:,ind[:round(self.num_points/2)]]
            pointcloud2 = pointcloud2[:,ind[round(self.num_points/2):]]
        if self.random_point_order:
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T
        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1, sigma=self.sigma_factor*0.01, clip=self.clip_factor*0.05)
            pointcloud2 = jitter_pointcloud(pointcloud2, sigma=self.sigma_factor*0.01, clip=self.clip_factor*0.05)
        if self.src_unbalance:
            pointcloud1 = pointcloud1[:,:512]
        if self.tgt_unbalance:
            pointcloud2 = pointcloud2[:, :512]
        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)
        if self.random_point_order:
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), euler_ab.astype('float32')

    def __len__(self):
        return self.data.shape[0]