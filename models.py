import torch
import torch.nn.functional as F
import torch.nn as nn

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class shapeClassifier(nn.Module):
    def __init__(self, args):
        super(shapeClassifier, self).__init__()
        self.lpe_dim = args.lpe_dim
        self.lpe_normalize = args.lpe_normalize
        self.use_xyz = args.use_xyz
        self.use_second_deg = args.use_second_deg
        self.lpe_dim = args.lpe_dim
        input_dim = 0
        if self.use_xyz:
            input_dim = 3
        if self.use_second_deg:
            input_dim = 9
        if self.lpe_dim != 0:
            input_dim = input_dim + (self.lpe_dim)
        input_size = input_dim * (args.sampled_points + 1)
        self.classifier =  MLP(input_size= input_size, num_layers=args.num_mlp_layers, num_neurons_per_layer=args.num_neurons_per_layer, output_size=args.output_dim)


    def forward(self, x):
        batch_size, _, num_of_pcl_centroids, k_nearest_neighbors = x.shape

        # Reshape input to (batch_size * num_of_pcl_centroids, k_nearest_neighbors, 3)
        x = x.permute(0, 2, 3, 1).reshape(batch_size * num_of_pcl_centroids, k_nearest_neighbors, 3)

        # Calculate Laplacian
        l = self.createLap(x, self.lpe_normalize)
        # Compute LPE embedding
        eigvecs = self.top_k_smallest_eigenvectors(l, self.lpe_dim)
        indices, fixed_eigs = self.sort_by_first_eigenvector(eigvecs)

        # Gather and reshape fixed point cloud
        data = torch.gather(x, 1, indices.unsqueeze(2).expand(-1, -1, 3))

        if self.use_second_deg:
            x, y, z = data.unbind(dim=2)
            data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
        if self.lpe_dim != 0:
            data = torch.cat([data, fixed_eigs], dim=2)
        data = data.permute(0, 2, 1)
        output = self.classifier(data)
        return output

    def createLap(self, point_cloud, normalized):
        distances = torch.cdist(point_cloud, point_cloud)
        weights = torch.exp(distances)
        column_sums = weights.sum(dim=1)
        diag_matrix = torch.diag_embed(column_sums)
        laplacian = diag_matrix - weights
        if normalized:
            inv_D_sqrt = torch.diag_embed(torch.sqrt(1.0 / (column_sums + 1e-7)))
            identity = torch.eye(weights.shape[1], device=weights.device).unsqueeze(0).expand_as(weights)
            laplacian = identity - (inv_D_sqrt @ weights @ inv_D_sqrt)
        return laplacian

    def top_k_smallest_eigenvectors(self, graph, k):
        eigenvalues, eigenvectors = torch.linalg.eigh(graph)
        return eigenvectors[:,: ,1:k+1]

    def sort_by_first_eigenvector(self, eigenvectors):
        abs_eigenvector = torch.abs(eigenvectors[:, :, 0])
        sorted_indices = torch.argsort(abs_eigenvector[:,1:])
        zeros_tensor = torch.zeros(size=(eigenvectors.shape[0],1),device=eigenvectors.device, dtype=int)
        sorted_indices = torch.cat((zeros_tensor, 1 + sorted_indices), dim=1)

        sorted_eigvecs = torch.gather(eigenvectors, 1,
                                      sorted_indices.unsqueeze(-1).expand(-1, -1, eigenvectors.size(-1)))
        return sorted_indices, sorted_eigvecs
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
