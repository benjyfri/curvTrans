import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm
import os
import wandb
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from PCT_Pytorch_main.model import *
from PCT_Pytorch_main.util import *
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
import h5py
import sklearn.metrics as metrics
import time


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.point_clouds_group = self.hdf5_file['point_clouds']
        self.num_point_clouds = len(self.point_clouds_group)
        self.indices = list(range(self.num_point_clouds))

    def __len__(self):
        return self.num_point_clouds

    def __getitem__(self, idx):
        point_cloud_name = f"point_cloud_{self.indices[idx]}"

        # Load point cloud data
        point_cloud = torch.tensor(self.point_clouds_group[point_cloud_name], dtype=torch.float32)
        lpe = createLPE(point_cloud)
        # Load metadata from attributes
        info = {key: self.point_clouds_group[point_cloud_name].attrs[key] for key in
                    self.point_clouds_group[point_cloud_name].attrs}

        return {"point_cloud": point_cloud, "lpe": lpe, "info": info}

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2, num_layers=2, num_heads=3, positional_encoding_type='sinusoidal'):
        super(TransformerNetwork, self).__init__()
        # Positional encoding layer
        if positional_encoding_type == 'sinusoidal':
            self.positional_encoding_type = 'sinusoidal'
            self.positional_encoding = nn.Embedding(21, input_dim)
        elif positional_encoding_type == 'learnable':
            self.positional_encoding_type = 'learnable'
            self.positional_encoding = nn.Parameter(torch.rand(21, input_dim))
        elif positional_encoding_type == 'laplacian':
            self.positional_encoding_type = 'laplacian'
            self.positional_encoding = None
        else:
            raise ValueError("Invalid positional_encoding_type. Choose 'sinusoidal' or 'learnable'.")

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        # Fully connected layer for output
        self.fc_output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Add positional encoding to the input
        if isinstance(self.positional_encoding, nn.Embedding):
            position = torch.arange(0, 21).unsqueeze(1).to(x.device)
            x = x + self.positional_encoding(position).permute(1, 0, 2)
        elif isinstance(self.positional_encoding, nn.Parameter):
            x = x + self.positional_encoding.unsqueeze(0)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Fully connected layer for output
        output = self.fc_output(x)

        # Split the output into k1 and k2
        k1, k2 = torch.chunk(output, 2, dim=1)

        return k1, k2
def createLPE(data):
    umbrella = create_triangles_ring(data[1:, :], data[0, :])
    centroids = umbrella[:, 2, :]
    p1 = umbrella[:, 0, :]
    p2 = umbrella[:, 1, :]
    centroid_to_p1_distances = np.linalg.norm(centroids - p1, axis=1)
    p1_to_p2_distances = np.linalg.norm(p1 - p2, axis=1)

    # Combine the distances into a single list
    distances_list = list(centroid_to_p1_distances) + list(p1_to_p2_distances)

    a = [0 for x in range(1, 21)]
    b = [x for x in range(1, 21)]
    shifted = b[1:] + b[:1]
    a.extend(b)
    b.extend(shifted)
    old_a = a.copy()
    a.extend(b)
    b.extend(old_a)
    distances_list.extend(distances_list)
    row = np.array(a)
    col = np.array(b)
    weights = np.array(distances_list)
    mat = csr_matrix((weights, (row, col)), shape=(21, 21)).toarray()
    lap = csgraph.laplacian(mat)
    lpe = laplacian_pe(lap, 15)
    return lpe
def laplacian_pe(lap, k):

    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(lap)
    kpartition_indices = np.argpartition(EigVal, k + 1)[:k + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = np.real(EigVec[:, topk_indices])

    return topk_EigVec
def create_triangles_ring(point_cloud, centroid):
    num_points = point_cloud.shape[0]
    triangles = np.zeros((num_points, 3, 3))

    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    # Project points onto the plane perpendicular to the normal
    projected_points = point_cloud - np.outer(np.dot(point_cloud, normal_at_centroid), normal_at_centroid)
    centroid_projected = centroid - np.outer(np.dot(centroid, normal_at_centroid), normal_at_centroid)
    # Calculate angles of each point with respect to the x-axis on the XY plane
    angles = np.arctan2(projected_points[:, 1] - centroid_projected[0,1], projected_points[:, 0] - centroid_projected[0,0])

    # Wrap angles to [0, 2*pi)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    # Sort the points based on angles
    sorted_indices = np.argsort(angles)
    sorted_points = point_cloud[sorted_indices]

    for i in range(0, num_points):
        triangles[i] = np.array([sorted_points[i], sorted_points[(i + 1) % num_points], centroid])

    return triangles


def test(model, dataloader, loss_function, device, args):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    count = 0
    test_pred = []
    test_true = []
    with torch.no_grad():
        for batch in dataloader:
            data, lpe, info = batch['point_cloud'].to(device), batch['lpe'].to(device), batch['info']
            label_class = info['class'].to(device)
            if args.use_lpe == 1:
                data = torch.cat([data, lpe], dim=2).to(device)
            data = data.permute(0, 2, 1)

            logits = model(data)
            loss = loss_function(logits, label_class)
            preds = logits.max(dim=1)[1]
            test_true.append(label_class.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total_loss += loss.item()
            count = count + 1
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    average_loss = total_loss / (count * args.batch_size)
    return average_loss, test_acc ,  avg_per_class_acc


def train_and_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if args.use_wandb:
        wandb.login(key="ed8e8f26d1ee503cda463f300a605cb35e75ad23")
        wandb.init(project="Curvature-transformer-POC", name=args.exp_name)

    num_epochs = args.epochs
    learning_rate = args.lr

    # Create instances for training and testing datasets
    train_dataset = PointCloudDataset(file_path="train_surfaces.h5")
    test_dataset = PointCloudDataset(file_path='test_surfaces.h5')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Pct(args, output_channels=8).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    criterion = cal_loss

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        count = 0
        train_pred = []
        train_true = []
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                data, lpe, info = batch['point_cloud'].to(device), batch['lpe'].to(device), batch['info']
                label_class = info['class'].to(device)
                if args.use_lpe==1:
                    data= torch.cat([data, lpe], dim=2).to(device)
                data =  data.permute(0, 2, 1)
                logits = model(data)
                loss = criterion(logits, label_class)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += loss.item()
                count = count + 1

                preds = logits.max(dim=1)[1]
                train_true.append(label_class.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())


                tqdm_bar.set_postfix(train_loss=f'{(loss.item() / args.batch_size):.4f}')

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        acc_train = metrics.accuracy_score( train_true , train_pred )
        avg_per_class_acc_train = metrics.balanced_accuracy_score( train_true , train_pred )
        train_loss = (total_train_loss / (args.batch_size * count))

        test_loss, acc_test ,  avg_per_class_acc_test = test(model, test_dataloader, criterion, device, args)
        scheduler.step()
        print({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss, "acc_train": acc_train, "acc_test": acc_test,
                       "avg_per_class_acc_train":avg_per_class_acc_train, "avg_per_class_acc_test" : avg_per_class_acc_test })
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss, "acc_train": acc_train, "acc_test": acc_test,
                       "avg_per_class_acc_train":avg_per_class_acc_train, "avg_per_class_acc_test" : avg_per_class_acc_test })
    # Save the trained model if needed
    torch.save(model.state_dict(), args.exp_name+'_trained_model.pth')
def configArgsPCT():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=512, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=21,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--use_wandb', type=int, default=0, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--use_lpe', type=int, default=0, metavar='N',
                        help='use laplacian positional encoding')
    parser.add_argument('--loss', type=int, default=1, metavar='N',
                        help='0 is for K1 and K2; 1 for gaussian and mean curvature; 2 for variances ')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = configArgsPCT()
    args.use_wandb=1
    args.exp_name='classification_PCT'
    print(args)
    train_and_test(args)
    wandb.finish()
    print("yay")


