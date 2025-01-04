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
        self.use_lap_reorder = args.use_lap_reorder
        self.lpe_dim = args.lpe_dim
        self.lap_eigenvalues_dim = args.lap_eigenvalues_dim
        self.graph_weight_mode = args.graph_weight_mode
        input_dim = 0
        if self.use_xyz:
            input_dim = 3
        if self.use_second_deg:
            input_dim = 9
        if (self.lpe_dim != 0):
            input_dim = input_dim + (self.lpe_dim)
        input_size = input_dim * (args.sampled_points + 1)
        if (self.lap_eigenvalues_dim !=0):
            input_size = input_size + (self.lap_eigenvalues_dim)
        self.classifier =  MLP(input_size= input_size, num_layers=args.num_mlp_layers, num_neurons_per_layer=args.num_neurons_per_layer, output_size=args.output_dim)


    def forward(self, x):
        batch_size, _, num_of_pcl_centroids, k_nearest_neighbors = x.shape

        # Reshape input to (batch_size * num_of_pcl_centroids, k_nearest_neighbors, 3)
        x = x.permute(0, 2, 3, 1).reshape(batch_size * num_of_pcl_centroids, k_nearest_neighbors, 3)

        if self.use_lap_reorder or self.lpe_dim or self.lap_eigenvalues_dim:
            # Calculate Laplacian
            l = self.createLap(x, self.lpe_normalize)
            # Compute LPE embedding
            eigvecs, eigenvals = self.top_k_smallest_eigenvectors(l, self.lpe_dim)
            indices, fixed_eigs = self.sort_by_first_eigenvector(eigvecs)

        if self.use_lap_reorder:
            # Gather and reshape fixed point cloud
            data = torch.gather(x, 1, indices.unsqueeze(2).expand(-1, -1, 3))
            data = (transform_point_clouds_to_canonical(data))
        else:
            data = x
        if self.use_second_deg:
            x, y, z = data.unbind(dim=2)
            data = torch.stack([x ** 2, x * y, x * z, y ** 2, y * z, z ** 2, x, y, z], dim=2)
        if (self.lpe_dim != 0):
            data = torch.cat([data, fixed_eigs], dim=2)
        data = data.permute(0, 2, 1)
        if (self.lap_eigenvalues_dim == 0):
            output = self.classifier(data)
        else:
            output = self.classifier(data, eigenvals[:, 1 : 1 + self.lap_eigenvalues_dim])
        output = output.view(batch_size, num_of_pcl_centroids, -1)
        return output

    def createLap(self, point_cloud, normalized):
        distances = torch.cdist(point_cloud, point_cloud)
        if self.graph_weight_mode == 0:
            weights = torch.exp(-distances)
        if self.graph_weight_mode == 1:
            weights = torch.exp(-distances**2)
        if self.graph_weight_mode == 2:
            batch_size = point_cloud.shape[0]
            rbf_weight = distances / (torch.max(distances.view(batch_size, -1), dim=1).values).view(batch_size, 1, 1)
            weights = torch.exp(-rbf_weight)
        column_sums = weights.sum(dim=1)
        diag_matrix = torch.diag_embed(column_sums)
        laplacian = diag_matrix - weights
        if normalized:
            inv_D_sqrt = torch.diag_embed(torch.sqrt(1.0 / (column_sums + 1e-7)))
            identity = torch.eye(weights.shape[1], device=weights.device).unsqueeze(0).expand_as(weights)
            laplacian = identity - (inv_D_sqrt @ weights @ inv_D_sqrt)
        return laplacian

    def top_k_smallest_eigenvectors(self, graph, k):
        if k<1:
            k=1
        eigenvalues, eigenvectors = torch.linalg.eigh(graph)
        return eigenvectors[:,: ,1:k+1], eigenvalues

    def sort_by_first_eigenvector(self, eigenvectors):
        first_eig_vec = eigenvectors[:, :, 0]

        abs_tensor = torch.abs(first_eig_vec)  # Absolute values
        max_indices = torch.argmax(abs_tensor, dim=1)  # Indices of max abs value

        # Step 2: Get the sign of the max absolute values
        max_values = first_eig_vec[torch.arange(eigenvectors.shape[0]), max_indices]  # Gather max values
        signs = torch.sign(max_values)  # Compute the sign of the max values

        # Step 3: Multiply each row by its corresponding sign
        result = first_eig_vec * signs.unsqueeze(1)

        sorted_indices = torch.argsort(result[:,1:])
        zeros_tensor = torch.zeros(size=(eigenvectors.shape[0],1),device=eigenvectors.device, dtype=int)
        sorted_indices = torch.cat((zeros_tensor, 1 + sorted_indices), dim=1)

        sorted_eigvecs = torch.gather(eigenvectors, 1,
                                      sorted_indices.unsqueeze(-1).expand(-1, -1, eigenvectors.size(-1)))
        return sorted_indices, sorted_eigvecs
    def sort_by_first_eigenvector_orig(self, eigenvectors):
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

    def forward(self, x, eigenvals=None):
        x = x.reshape(x.size(0), -1)
        if eigenvals is not None:
            x = torch.cat((x, eigenvals), dim=1)
        return self.model(x)

class MLP_Returns_Mid_Layer(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons_per_layer, output_size):
        super(MLP_Returns_Mid_Layer, self).__init__()
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
        activations = []
        layer_32_32 = False
        for layer in self.model:
            x = layer(x)
            #choose only the last 32*32 layer
            if isinstance(layer, nn.Linear) and layer.in_features == 32 and layer.out_features == 32:
                if layer_32_32 == False:
                    layer_32_32 = True
                if layer_32_32 == True:
                    activations.append(x)
        output_layer = x
        layer_before_last = activations[0]
        return output_layer , layer_before_last

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

def transform_point_clouds_to_canonical(point_clouds: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """
    Normalize batched point clouds by rotating them according to specified conditions.
    Args:
        point_clouds: Tensor of shape (batch_size, 21, 3)
        epsilon: Small value for numerical stability
    Returns:
        Normalized point clouds of same shape
    """
    # Calculate center of mass for each point cloud
    m_point = torch.mean(point_clouds, dim=1)  # Shape: (batch_size, 3)

    # Step 1: Rotate to put center of mass at (0, 0, norm(m_point))
    m_norm = torch.norm(m_point, dim=1, keepdim=True)  # Shape: (batch_size, 1)

    # Create rotation matrix to align m_point with (0, 0, norm(m_point))
    v1 = m_point / (m_norm + epsilon)
    v2 = torch.zeros_like(m_point)
    v2[:, 2] = 1.0

    # Get rotation axis and angle using cross product and dot product
    rotation_axis = torch.cross(v1, v2)
    rotation_axis = rotation_axis / (torch.norm(rotation_axis, dim=1, keepdim=True) + epsilon)
    cos_theta = torch.sum(v1 * v2, dim=1)
    sin_theta = torch.sqrt(1 - cos_theta ** 2 + epsilon)

    # Rodriguez rotation formula
    K = torch.zeros((point_clouds.shape[0], 3, 3), device=point_clouds.device)
    K[:, 0, 1] = -rotation_axis[:, 2]
    K[:, 0, 2] = rotation_axis[:, 1]
    K[:, 1, 0] = rotation_axis[:, 2]
    K[:, 1, 2] = -rotation_axis[:, 0]
    K[:, 2, 0] = -rotation_axis[:, 1]
    K[:, 2, 1] = rotation_axis[:, 0]

    R1 = torch.eye(3, device=point_clouds.device).unsqueeze(0) + \
         sin_theta.unsqueeze(-1).unsqueeze(-1) * K + \
         (1 - cos_theta).unsqueeze(-1).unsqueeze(-1) * (K @ K)

    # Apply first rotation
    rotated_points = torch.bmm(point_clouds, R1.transpose(1, 2))

    # Step 2: Get the rotated last point directly from rotated_points
    rotated_p = rotated_points[:, -1, :]  # Shape: (batch_size, 3)

    # Calculate rotation angle around z-axis to align rotated_p with required conditions
    angle = torch.atan2(rotated_p[:, 1], rotated_p[:, 0])  # Current angle of the last point in xy-plane

    # Adjust the angle to ensure y = 0 and x >= 0
    angle = -angle
    angle = torch.where(rotated_p[:, 0] < 0, angle + torch.pi, angle)

    # Create z-axis rotation matrix using adjusted angle
    cos_phi = torch.cos(angle)
    sin_phi = torch.sin(angle)

    R2 = torch.zeros((point_clouds.shape[0], 3, 3), device=point_clouds.device)
    R2[:, 0, 0] = cos_phi
    R2[:, 0, 1] = -sin_phi
    R2[:, 1, 0] = sin_phi
    R2[:, 1, 1] = cos_phi
    R2[:, 2, 2] = 1.0

    # Apply second rotation
    final_points = torch.bmm(rotated_points, R2.transpose(1, 2))

    # Verify that the last point has y = 0 and x >= 0
    final_rotated_p = final_points[:, -1, :]
    final_rotated_p[:, 1] = 0  # Force y to be 0
    final_rotated_p[:, 0] = torch.abs(final_rotated_p[:, 0])  # Ensure x is non-negative

    # Apply these corrections to the final points
    final_points[:, -1, :] = final_rotated_p

    # Step 3: Adjust the sign of x and y coordinates based on the second point's x-coordinate
    second_point_x = final_points[:, 1, 0]  # Extract the x-coordinate of the second point
    sign = torch.sign(second_point_x).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 1, 1)

    # Multiply all x and y coordinates by the sign to remove ambiguity
    final_points[:, :-1, 0:2] *= sign

    # Zero out small values for stability
    final_points = torch.where(torch.abs(final_points) < epsilon,
                               torch.zeros_like(final_points),
                               final_points)

    return final_points

