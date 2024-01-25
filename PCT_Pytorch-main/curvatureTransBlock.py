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
import dgl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from model import *
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
class MLP(nn.Module):
    def __init__(self, input_size=68, hidden_size1=64, hidden_size2=64, hidden_size3=64, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x


def laplacian_pe(lap, k):

    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(lap)
    kpartition_indices = np.argpartition(EigVal, k + 1)[:k + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = np.real(EigVec[:, topk_indices])

    return topk_EigVec
class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [f for f in os.listdir(data_folder) if 'curve' not in f]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        index = idx
        if self.data_folder == './test_curve_transformer':
            index = index + 45000
        data_path = os.path.join(self.data_folder, f"{index}.npy")
        info_path = os.path.join(self.data_folder, f"{index}_curve_and_run_info.npy")

        data = np.load(data_path)
        umbrella = create_triangles_ring(data[1:,:], data[0,:])
        centroids = umbrella[:, 2, :]
        p1 = umbrella[:, 0, :]
        p2 = umbrella[:, 1, :]
        centroid_to_p1_distances = np.linalg.norm(centroids - p1, axis=1)
        p1_to_p2_distances = np.linalg.norm(p1 - p2, axis=1)

        # Combine the distances into a single list
        distances_list = list(centroid_to_p1_distances) + list(p1_to_p2_distances)

        a = [0 for x in range(1,21)]
        b = [x for x in range(1,21)]
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

        # g = dgl.graph((a, b))
        # print(torch.__version__)
        # lpe = dgl.laplacian_pe(g, 2)
        # g = dgl.to_bidirected(g)
        # print(g)
        # print("yay")
        # a =dgl.graph(umbrella)
        # fig = go.Figure()
        #
        # # Plot triangles
        # for i, t in enumerate(umbrella):
        #     x_vals = [t[0, 0], t[1, 0], t[2, 0], t[0, 0]]
        #     y_vals = [t[0, 1], t[1, 1], t[2, 1], t[0, 1]]
        #     z_vals = [t[0, 2], t[1, 2], t[2, 2], t[0, 2]]
        #
        #     fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', name=f'Triangle {i + 1}'))
        #
        # angle_sum = 0
        # # Plot original point cloud points with numbers
        # for i, point in enumerate(umbrella):
        #     vec1 = point[0] - data[0,:]
        #     vec2 = point[1] - data[0,:]
        #     dot_product = np.dot(vec1, vec2)
        #     norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        #     angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        #     angle = np.rad2deg(angle)
        #     angle_sum += angle
        #     fig.add_trace(go.Scatter3d(x=[point[0][0]], y=[point[0][1]], z=[point[0][2]],
        #                                mode='markers+text', text=[f'{i + 1}, {angle:.2f}'],
        #                                marker=dict(size=5, color='blue'), name='Point Cloud'))
        #
        # # Plot centroid
        # fig.add_trace(go.Scatter3d(x=[data[0,:][0]], y=[data[0,:][1]], z=[data[0,:][2]],
        #                            mode='markers+text', text=['Centroid'],
        #                            marker=dict(size=10, color='red'), name='Centroid'))
        #
        # fig.update_layout(scene=dict(aspectmode="data"))
        # fig.show()
        info = np.load(info_path, allow_pickle=True)
        info = np.array(info, dtype=np.float32)
        return {'data': torch.tensor(data, dtype=torch.float32),
                'lpe': torch.tensor(lpe, dtype=torch.float32),
                'info': torch.tensor(info, dtype=torch.float32)}


# Custom Loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        loss = nn.MSELoss()(predicted, target)
        return torch.sqrt(loss)

import torch

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
    with torch.no_grad():
        for batch in dataloader:
            data, lpe, info = batch['data'].to(device), batch['lpe'].to(device), batch['info'].to(device)
            [batch_size, patch_size, dim] = data.shape
            lidar_coord = info[:, 4:7]
            center_point = torch.tensor([0.0, 0.0, 1.0]).to(device)
            # center points
            data = data - center_point
            if args.use_lpe == 1:
                data = torch.cat([data, lpe], dim=2)

            data = data.permute(0, 2, 1)
            predictions = model(data)
            # predictions =torch.cat((k1, k2), dim=1)

            # Calculate the loss
            if args.loss == 0:
                target = info[:, 2:4]
            if args.loss == 1:
                target = info[:, 0:2]
            if args.loss == 2:
                target = info[:, 7:]
            loss = loss_function(predictions, target)

            total_loss += loss.item()
            count = count + 1

    average_loss = total_loss / count
    return average_loss


def train_and_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if args.use_wandb:
        wandb.login(key="ed8e8f26d1ee503cda463f300a605cb35e75ad23")
        wandb.init(project="Curvature-transformer-POC", name=args.exp_name)

    num_epochs = args.epochs
    learning_rate = args.lr

    # Create instances for training and testing datasets
    train_dataset = CustomDataset(data_folder='./train_curve_transformer')
    test_dataset = CustomDataset(data_folder='./test_curve_transformer')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # if args.use_lpe==1:
    #     model = TransformerNetwork(input_dim=18, hidden_dim=128, output_dim=2, num_layers=4, num_heads=18,
    #                                positional_encoding_type='laplacian').to(device)
    # else:
    #     model = TransformerNetwork(input_dim=3, hidden_dim=128, output_dim=2, num_layers=4, num_heads=3, positional_encoding_type='sinusoidal').to(device)
    # if args.use_lpe==1:
    #     model = MLP(input_size=(21*(3+15)) , hidden_size1=128,hidden_size2=128,hidden_size3=128).to(device)
    # else:
    #     model = MLP(input_size=(21*(3)) , hidden_size1=128,hidden_size2=128,hidden_size3=128).to(device)
    model = Pct(args, output_channels=2).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    loss_function = CustomLoss().to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        count = 0
        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as tqdm_bar:
            for batch in tqdm_bar:
                data, lpe, info = batch['data'].to(device), batch['lpe'].to(device), batch['info'].to(device)
                [batch_size, patch_size, dim] = data.shape
                lidar_coord = info[:, 4:7]
                center_point = torch.tensor([0.0, 0.0, 1.0]).to(device)
                #center points
                data = data - center_point
                if args.use_lpe==1:
                    data= torch.cat([data, lpe], dim=2).to(device)
                data =  data.permute(0, 2, 1)
                predictions = model(data)
                # Calculate the loss
                if args.loss==0:
                    target = info[:, 2:4]
                if args.loss==1:
                    target = info[:, 0:2]
                if args.loss==2:
                    target = info[:, 7:]
                loss = loss_function(predictions, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']

                total_train_loss += loss.item()
                count = count + 1
                tqdm_bar.set_postfix(train_loss=f'{(loss.item() / args.batch_size):.4f}')

        test_loss = test(model, test_dataloader, loss_function, device, args)
        test_loss = test_loss / args.batch_size
        train_loss = (total_train_loss / (args.batch_size * count))
        scheduler.step()
        print({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "learning_rate": current_lr})
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss ,"test_loss": test_loss})
    # Save the trained model if needed
    torch.save(model.state_dict(), args.exp_name+'_trained_model.pth')
def configArgsPCT():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
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
def configArgs():
    parser = argparse.ArgumentParser(description='Curvature-transformer-POC')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=512, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--learning_rate', type=float, default=0.1, metavar='N',
                        help='initial learning rate ')
    parser.add_argument('--use_wandb', type=int, default=0, metavar='N',
                        help='use angles in learning ')
    parser.add_argument('--use_lpe', type=int, default=0, metavar='N',
                        help='use laplacian positional encoding')
    parser.add_argument('--loss', type=int, default=0, metavar='N',
                        help='0 is for K1 and K2; 1 for gaussian and mean curvature; 2 for variances ')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = configArgs()
    args = configArgsPCT()
    # for i in range(5):
    #     args.epochs = 5
    #     # lr = 10**(-i)
    #     lr = 10**(i+1)
    #     args.learning_rate = lr
    #     args.exp_name = f'gauss_{lr}_MLP_LPE'
    #     print(f'lr:{lr}')
    #     print(f'args:{args}')
    #     train_and_test(args)
    #     wandb.finish()
    print(args)
    train_and_test(args)
    wandb.finish()
    print("yay")


