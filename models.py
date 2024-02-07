
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
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
