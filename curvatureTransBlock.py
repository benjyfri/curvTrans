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
from scipy.sparse import spmatrix
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
import h5py
import sklearn.metrics as metrics
import time
import torch.nn.functional as F
import torch.nn as nn
import dgl
def positional_encoding_nerf(points , channels_per_dim=5):
  """
  Creates positional encoding for a 3D point cloud using sinusoidal functions.

  Args:
      points: A NumPy array of shape (N, 3) representing the point cloud.

  Returns:
      A NumPy array of shape (N, C) where C is the number of encoding dimensions.
  """
  dims = points.shape[-1]  # Number of dimensions (3 for 3D points)
  channels = dims * channels_per_dim  # Two channels per dimension (sin and cos)
  encoding = np.zeros((points.shape[0], channels))

  for i in range(channels):
    frequency = 1 / np.power(10000, channels_per_dim * (i // channels_per_dim) / dims)
    channel_id = i % channels_per_dim
    if channel_id == 0:
      encoding[:, i] = np.sin(points[:, i // channels_per_dim] * frequency)
    else:
      encoding[:, i] = np.cos(points[:, i // channels_per_dim] * frequency)

  return encoding


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
class DGCNN(nn.Module):
    def __init__(self,input_dim=3, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=21)
        batch_size, num_points, num_dims = x.size()
        # x = x.permute(0, 2, 1)  # Adjust input dimensions for 1D convolution

        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x
class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, args):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.point_clouds_group = self.hdf5_file['point_clouds']
        self.num_point_clouds = len(self.point_clouds_group)
        self.indices = list(range(self.num_point_clouds))
        self.use_lpe = args.use_lpe
        self.lpe_dim = args.lpe_dim
    def __len__(self):
        return self.num_point_clouds

    def __getitem__(self, idx):
        point_cloud_name = f"point_cloud_{self.indices[idx]}"

        # Load point cloud data
        point_cloud = self.point_clouds_group[point_cloud_name]
        if self.use_lpe==1:
            lpe, pcl = createLPE(point_cloud, self.lpe_dim)
            point_cloud = torch.tensor(pcl, dtype=torch.float32)
        else:
            point_cloud = torch.tensor(np.array(point_cloud), dtype=torch.float32)
            lpe = torch.tensor([])
        # Load metadata from attributes
        info = {key: self.point_clouds_group[point_cloud_name].attrs[key] for key in
                    self.point_clouds_group[point_cloud_name].attrs}

        return {"point_cloud": point_cloud, "lpe": lpe, "info": info}


class MLP(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons_per_layer, output_size):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, num_neurons_per_layer))  # input layer

        for _ in range(num_layers - 1):
            layers.append(nn.BatchNorm1d(num_neurons_per_layer))  # Batch normalization
            layers.append(nn.ReLU())  # ReLU activation
            layers.append(nn.Linear(num_neurons_per_layer, num_neurons_per_layer))

        layers.append(nn.BatchNorm1d(num_neurons_per_layer))  # Batch normalization
        layers.append(nn.ReLU())  # ReLU activation
        layers.append(nn.Linear(num_neurons_per_layer, output_size))  # output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.model(x)
class TransformerNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=5, num_heads=1, num_layers=1, emb_dim=512):
        super(TransformerNetwork, self).__init__()
        # Define a list to hold the multihead attention, feedforward, and normalization layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads),
                nn.Linear(input_dim, 4 * input_dim),  # Feedforward layer
                nn.ReLU(),  # Nonlinearity
                nn.Linear(4 * input_dim, input_dim),  # Output projection
                nn.LayerNorm(input_dim)
            ]) for _ in range(num_layers)
        ])

        # Classifier layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Transpose input for the first MultiheadAttention layer
        x = x.permute(0, 2, 1)  # Assuming the input has dimensions (batch_size, 21, 3)

        # Apply multihead attention, feedforward, and residual layers
        for attn_layer, ff_layer, relu, out_proj, norm_layer in self.layers:
            attn_output, _ = attn_layer(x, x, x)
            x = x + attn_output
            x = norm_layer(x)

            ff_output = ff_layer(x)
            ff_output = relu(ff_output)  # Apply nonlinearity
            ff_output = out_proj(ff_output)
            x = x + ff_output  # Residual connection
            x = norm_layer(x)

        # Sum along the sequence dimension (assuming the sequence dimension is 21)
        attn_output_sum = x.sum(dim=1)

        # Classification layer
        output = self.fc(attn_output_sum)

        return output
class TransformerNetworkPCT(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, num_heads=1, num_layers=1, att_per_layer=4):
        super(TransformerNetworkPCT, self).__init__()
        # Define a list to hold the multihead attention, feedforward, and normalization layers
        self.args = args
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)
            for _ in range(att_per_layer)
        ])

        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        att_list = []
        # Apply multihead attention, feedforward, and residual layers
        for attn_layer in self.layers:
            x, _ = attn_layer(x, x, x)
            att_list.append(x)
        concatenated_tensor = torch.cat(att_list, dim=-1)
        x = concatenated_tensor.permute(0, 2, 1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = self.linear3(x)

        return x
def createLPE(data, lpe_dim):
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
    # mat = csr_matrix((weights, (row, col)), shape=(21, 21)).toarray()
    g = dgl.from_scipy(csr_matrix((weights, (row, col)), shape=(21, 21)))
    # lap = csgraph.laplacian(mat)
    # lpe = laplacian_pe(lap, 15)
    lpe = dgl.lap_pe(g, lpe_dim)
    pcl = np.concatenate([(data[0, :][np.newaxis]), p1])
    return lpe, pcl
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
    total_acc_loss = 0.0
    count = 0
    label_correct = {label: 0 for label in range(args.output_dim)}
    label_total = {label: 0 for label in range(args.output_dim)}

    with torch.no_grad():
        for batch in dataloader:
            data, lpe, info = batch['point_cloud'].to(device), batch['lpe'].to(device), batch['info']
            label_class = info['class'].to(device).long()
            if args.use_lpe == 1:
                data = torch.cat([data, lpe], dim=2).to(device)
            if args.use_second_deg:
                x, y, z = data.unbind(dim=2)
                data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
            data = data.permute(0, 2, 1)

            logits = model(data)
            loss = loss_function(logits, label_class)
            preds = logits.max(dim=1)[1]
            total_acc_loss += torch.mean((preds == label_class).float()).item()
            total_loss += loss.item()
            count += 1

            # Update per-label statistics
            for label in range(args.output_dim):
                correct_mask = (preds == label_class) & (label_class == label)
                label_correct[label] += correct_mask.sum().item()
                label_total[label] += (label_class == label).sum().item()

    # Overall accuracy
    test_acc = total_acc_loss / count
    average_loss = total_loss / (count * args.batch_size)

    # Calculate per-label accuracies
    label_accuracies = {label: label_correct[label] / label_total[label] if label_total[label] != 0 else 0.0
                       for label in range(args.output_dim)}

    return average_loss, test_acc, label_accuracies



def train_and_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(args)
    if args.use_wandb:
        wandb.login(key="ed8e8f26d1ee503cda463f300a605cb35e75ad23")
        wandb.init(project="Curvature-transformer-POC", name=args.exp_name)

    num_epochs = args.epochs
    learning_rate = args.lr

    # Create instances for training and testing datasets
    train_dataset = PointCloudDataset(file_path="train_surfaces.h5" , args=args)
    test_dataset = PointCloudDataset(file_path='test_surfaces.h5' , args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    input_dim = 3
    if args.use_second_deg:
        input_dim = 9
    if args.use_lpe==1:
        input_dim = 3 + args.lpe_dim
    if args.use_pct:
        model = TransformerNetworkPCT(input_dim=input_dim, output_dim=4, num_heads=args.num_of_heads, num_layers=args.num_of_attention_layers, att_per_layer=4).to(device)
    elif args.use_mlp:
        model = MLP(input_size= input_dim * 21, num_layers=args.num_mlp_layers, num_neurons_per_layer=args.num_neurons_per_layer, output_size=4).to(device)
    else:
        model = TransformerNetwork(input_dim=input_dim, output_dim=4, num_heads=args.num_of_heads, num_layers=args.num_of_attention_layers).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters in NN: {num_params}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    milestones = np.linspace(args.lr_jumps,num_epochs,num_epochs//args.lr_jumps)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        total_train_acc_loss = 0.0
        count = 0
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                data, lpe, info = batch['point_cloud'].to(device), batch['lpe'].to(device), batch['info']
                label_class = info['class'].to(device).long()
                if args.use_lpe==1:
                    data= torch.cat([data, lpe], dim=2).to(device)
                if args.use_second_deg:
                    x, y, z = data.unbind(dim=2)
                    data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
                data =  data.permute(0, 2, 1)
                logits = model(data)
                loss = criterion(logits, label_class)


                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += loss.item()
                preds = logits.max(dim=1)[1]
                total_train_acc_loss += torch.mean((preds == label_class).float()).item()

                count = count + 1

                tqdm_bar.set_postfix(train_loss=f'{(loss.item() / args.batch_size):.4f}')
        acc_train = (total_train_acc_loss / (count))
        train_loss = (total_train_loss / (args.batch_size * count))

        test_loss, acc_test, label_accuracies = test(model, test_dataloader, criterion, device, args)
        scheduler.step()
        print(f'LR: {current_lr}')
        print({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss, "acc_train": acc_train, "acc_test": acc_test,
                       "label_0": label_accuracies[0], "label_1": label_accuracies[1], "label_2": label_accuracies[2], "label_3": label_accuracies[3]})
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss, "acc_train": acc_train, "acc_test": acc_test,
                       "label_0": label_accuracies[0], "label_1": label_accuracies[1], "label_2": label_accuracies[2], "label_3": label_accuracies[3]})

    # Save the trained model
    # torch.save(model.state_dict(), f'{args.exp_name}_trained_model.pth')
    # print(f'Trained model saved to {args.exp_name}_trained_model.pth')

def configArgsPCT():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=512, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    parser.add_argument('--use_second_deg', type=int, default=0, metavar='N',
                        help='use second degree embedding ')
    parser.add_argument('--use_lpe', type=int, default=0, metavar='N',
                        help='use laplacian positional encoding')
    parser.add_argument('--use_pct', type=int, default=0, metavar='N',
                        help='use PCT transformer version')
    parser.add_argument('--use_mlp', type=int, default=0, metavar='N',
                        help='use PCT transformer version')
    parser.add_argument('--lpe_dim', type=int, default=0, metavar='N',
                        help='laplacian positional encoding amount of eigens to take')
    parser.add_argument('--num_of_heads', type=int, default=1, metavar='N',
                        help='how many attention heads to use')
    parser.add_argument('--num_neurons_per_layer', type=int, default=64, metavar='N',
                        help='how many neurons per layer to use')
    parser.add_argument('--num_mlp_layers', type=int, default=4, metavar='N',
                        help='how many mlp layers to use')
    parser.add_argument('--num_of_attention_layers', type=int, default=1, metavar='N',
                        help='how many attention layers to use')
    parser.add_argument('--att_per_layer', type=int, default=4, metavar='N',
                        help='how many attention heads in each layer')
    parser.add_argument('--output_dim', type=int, default=4, metavar='N',
                        help='how many labels are used')
    parser.add_argument('--lr_jumps', type=int, default=50, metavar='N',
                        help='Lower lr *0.1 every amount of jumps')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = configArgsPCT()
    train_and_test(args)
    # wandb.finish()


