"""
Heterogeneous GNN (HeteroGNN) models for MAML training
Handles different node types with type-specific transformations.

Node Types in Circuit Topology:
- Power nodes (VDD, VSS): feature[0] = 1
- Port nodes (Output, Input, Intermediate): feature[1] = 1
- NMOS transistors: feature[2] = 1
- PMOS transistors: feature[2] = -1

This model applies different transformations for each node type,
allowing the network to learn type-specific representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree


# ============================================================================
# Node Type Utilities
# ============================================================================

def get_node_types(x):
    """
    Extract node types from node features.

    Node type encoding in features:
    - feature[0] = 1: Power node (VDD/VSS)
    - feature[1] = 1: Port node (Output/Input/Intermediate)
    - feature[2] = 1: NMOS transistor
    - feature[2] = -1: PMOS transistor

    Args:
        x: Node features [num_nodes, num_features]

    Returns:
        node_types: Tensor of node type indices [num_nodes]
            0 = Power, 1 = Port, 2 = NMOS, 3 = PMOS
    """
    num_nodes = x.size(0)
    node_types = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    # Power nodes: feature[0] == 1
    power_mask = x[:, 0] == 1.0
    node_types[power_mask] = 0

    # Port nodes: feature[1] == 1
    port_mask = x[:, 1] == 1.0
    node_types[port_mask] = 1

    # NMOS transistors: feature[2] == 1
    nmos_mask = x[:, 2] == 1.0
    node_types[nmos_mask] = 2

    # PMOS transistors: feature[2] == -1
    pmos_mask = x[:, 2] == -1.0
    node_types[pmos_mask] = 3

    return node_types


def get_node_type_masks(x):
    """
    Get boolean masks for each node type.

    Args:
        x: Node features [num_nodes, num_features]

    Returns:
        dict of masks: {'power': mask, 'port': mask, 'nmos': mask, 'pmos': mask}
    """
    return {
        'power': x[:, 0] == 1.0,
        'port': x[:, 1] == 1.0,
        'nmos': x[:, 2] == 1.0,
        'pmos': x[:, 2] == -1.0
    }


# ============================================================================
# Heterogeneous GNN Layers
# ============================================================================

class HeteroGCNConv(MessagePassing):
    """
    Heterogeneous GCN Convolution Layer.

    Uses different transformation matrices for different node types:
    - Each node type has its own linear transformation
    - Messages are aggregated considering source node type
    """

    def __init__(self, in_channels, out_channels, num_node_types=4, aggr='add'):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types

        # Type-specific linear transformations
        self.type_linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_node_types)
        ])

        # Shared bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.type_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, node_types):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            node_types: Node type indices [num_nodes] (0=Power, 1=Port, 2=NMOS, 3=PMOS)

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization (degree-based)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Apply type-specific transformations (dtype matches x for mixed precision compatibility)
        x_transformed = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)

        for type_idx in range(self.num_node_types):
            mask = node_types == type_idx
            if mask.any():
                x_transformed[mask] = self.type_linears[type_idx](x[mask]).to(x.dtype)

        # Message passing
        out = self.propagate(edge_index, x=x_transformed, norm=norm)

        return out + self.bias

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class HeteroGATConv(MessagePassing):
    """
    Heterogeneous GAT (Graph Attention) Convolution Layer.

    Uses type-specific attention mechanisms:
    - Different attention weights for different source-target type pairs
    - Multi-head attention supported
    """

    def __init__(self, in_channels, out_channels, num_node_types=4, heads=4,
                 concat=True, dropout=0.0):
        super().__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Output dimension per head
        self.head_dim = out_channels // heads if concat else out_channels

        # Type-specific transformations
        self.type_linears = nn.ModuleList([
            nn.Linear(in_channels, heads * self.head_dim, bias=False)
            for _ in range(num_node_types)
        ])

        # Attention parameters (shared across types for simplicity)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.head_dim))

        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.type_linears:
            nn.init.xavier_uniform_(linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, node_types):
        """
        Forward pass with type-aware attention.
        """
        H, C = self.heads, self.head_dim
        num_nodes = x.size(0)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Apply type-specific transformations (dtype matches x for mixed precision compatibility)
        x_transformed = torch.zeros(num_nodes, H * C, device=x.device, dtype=x.dtype)

        for type_idx in range(self.num_node_types):
            mask = node_types == type_idx
            if mask.any():
                x_transformed[mask] = self.type_linears[type_idx](x[mask]).to(x.dtype)

        # Compute attention coefficients (keep 2D for propagate)
        # Reshape temporarily for attention computation
        x_reshaped = x_transformed.view(num_nodes, H, C)
        alpha_src = (x_reshaped * self.att_src).sum(dim=-1)  # [num_nodes, H]
        alpha_dst = (x_reshaped * self.att_dst).sum(dim=-1)  # [num_nodes, H]

        # Message passing with attention (pass 2D tensors)
        out = self.propagate(edge_index, x=x_transformed,
                            alpha_src=alpha_src, alpha_dst=alpha_dst,
                            size=(num_nodes, num_nodes))

        return out + self.bias

    def message(self, x_j, alpha_src_j, alpha_dst_i, index):
        """
        Compute attention-weighted messages.

        Args:
            x_j: Source node features [num_edges, H*C]
            alpha_src_j: Source attention [num_edges, H]
            alpha_dst_i: Target attention [num_edges, H]
            index: Target node indices
        """
        H, C = self.heads, self.head_dim

        # Compute attention weights per head
        alpha = alpha_src_j + alpha_dst_i  # [num_edges, H]
        alpha = F.leaky_relu(alpha, 0.2)

        # Softmax per target node per head
        from torch_geometric.utils import softmax
        alpha = softmax(alpha, index)  # [num_edges, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Reshape x_j for multi-head and apply attention
        x_j_reshaped = x_j.view(-1, H, C)  # [num_edges, H, C]
        out = alpha.unsqueeze(-1) * x_j_reshaped  # [num_edges, H, C]

        if self.concat:
            return out.view(-1, H * C)  # [num_edges, H*C]
        else:
            return out.mean(dim=1)  # [num_edges, C]


# ============================================================================
# Output Node Pooling
# ============================================================================

def output_node_pool(x, batch, output_node_idx=None, default_idx=2):
    """
    Output-node-only pooling: extract only the output node embedding.
    """
    batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1

    if output_node_idx is None:
        output_node_idx = [default_idx] * batch_size
    elif isinstance(output_node_idx, int):
        output_node_idx = [output_node_idx] * batch_size
    elif isinstance(output_node_idx, torch.Tensor):
        output_node_idx = output_node_idx.tolist()

    output_embeddings = []

    for graph_idx in range(batch_size):
        graph_mask = (batch == graph_idx)
        graph_node_indices = graph_mask.nonzero(as_tuple=True)[0]
        node_idx = output_node_idx[graph_idx] if graph_idx < len(output_node_idx) else default_idx

        if len(graph_node_indices) > node_idx:
            output_node_global_idx = graph_node_indices[node_idx]
            output_embeddings.append(x[output_node_global_idx])
        else:
            output_embeddings.append(x[graph_node_indices].mean(dim=0))

    return torch.stack(output_embeddings, dim=0)


# ============================================================================
# MAML Heterogeneous GNN Model
# ============================================================================

class MAML_HeteroGNN(nn.Module):
    """
    Heterogeneous GNN model for MAML.

    Uses different transformations for different node types:
    - Power nodes (VDD/VSS)
    - Port nodes (Output/Input/Intermediate)
    - NMOS transistors
    - PMOS transistors
    """

    def __init__(self, node_features, pooling='mean', output_dim=1, dropout=0.0,
                 conv_hidden_dim=32, num_conv_layers=2, fc_hidden_dim=256, num_fc_layers=2,
                 output_node_idx=2, num_node_types=4, conv_type='gcn', heads=4):
        """
        Args:
            node_features: Number of input node features
            pooling: Pooling method ('mean', 'max', 'add', 'output')
            output_dim: Output dimension (default: 1)
            dropout: Dropout rate
            conv_hidden_dim: Hidden dimension for convolution layers
            num_conv_layers: Number of convolution layers
            fc_hidden_dim: Hidden dimension for FC layers
            num_fc_layers: Number of FC layers (1, 2, 3, or 4)
            output_node_idx: Index of output node for 'output' pooling
            num_node_types: Number of distinct node types (default: 4)
            conv_type: Convolution type ('gcn' or 'gat')
            heads: Number of attention heads (for GAT only)
        """
        super().__init__()

        self.node_features = node_features
        self.pooling_type = pooling
        self.output_dim = output_dim
        self.dropout = dropout
        self.conv_hidden_dim = conv_hidden_dim
        self.num_conv_layers = num_conv_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_fc_layers = num_fc_layers
        self.output_node_idx = output_node_idx
        self.num_node_types = num_node_types
        self.conv_type = conv_type
        self.heads = heads

        # Input projection (type-specific)
        self.input_linears = nn.ModuleList([
            nn.Linear(node_features, conv_hidden_dim)
            for _ in range(num_node_types)
        ])

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()

        if conv_type == 'gat':
            for i in range(num_conv_layers):
                in_dim = conv_hidden_dim
                out_dim = conv_hidden_dim
                self.convs.append(
                    HeteroGATConv(in_dim, out_dim, num_node_types, heads=heads,
                                  concat=(i < num_conv_layers - 1), dropout=dropout)
                )
        else:  # gcn
            for _ in range(num_conv_layers):
                self.convs.append(
                    HeteroGCNConv(conv_hidden_dim, conv_hidden_dim, num_node_types)
                )

        # Output layers
        self.output_layers = nn.ModuleDict()

        if num_fc_layers == 1:
            self.output_layers['fc1'] = nn.Linear(conv_hidden_dim, output_dim)
        elif num_fc_layers == 2:
            self.output_layers['fc1'] = nn.Linear(conv_hidden_dim, fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(fc_hidden_dim, output_dim)
        elif num_fc_layers == 3:
            self.output_layers['fc1'] = nn.Linear(conv_hidden_dim, fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.output_layers['fc3'] = nn.Linear(fc_hidden_dim, output_dim)
        else:  # num_fc_layers == 4
            self.output_layers['fc1'] = nn.Linear(conv_hidden_dim, fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.output_layers['fc3'] = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.output_layers['fc4'] = nn.Linear(fc_hidden_dim, output_dim)

        # Pooling functions
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass.

        Args:
            x: Node features or PyG Data/Batch object
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        # Handle PyG Data/Batch input
        output_node_idx = None
        if hasattr(x, 'x'):
            data = x
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            if hasattr(data, 'output_node_idx'):
                output_node_idx = data.output_node_idx

        # Get node types from features
        node_types = get_node_types(x)

        # Type-specific input projection (dtype matches x for mixed precision compatibility)
        h = torch.zeros(x.size(0), self.conv_hidden_dim, device=x.device, dtype=x.dtype)
        for type_idx in range(self.num_node_types):
            mask = node_types == type_idx
            if mask.any():
                h[mask] = self.input_linears[type_idx](x[mask]).to(x.dtype)

        # Heterogeneous convolution layers
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, node_types)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)

        # Pooling
        if self.pooling_type == 'output':
            h = output_node_pool(h, batch, output_node_idx, default_idx=self.output_node_idx)
        else:
            h = self.pool(h, batch)

        # Output layers
        if self.num_fc_layers == 1:
            h = self.output_layers['fc1'](h)
        elif self.num_fc_layers == 2:
            h = F.relu(self.output_layers['fc1'](h))
            h = self.output_layers['fc2'](h)
        elif self.num_fc_layers == 3:
            h = F.relu(self.output_layers['fc1'](h))
            h = F.relu(self.output_layers['fc2'](h))
            h = self.output_layers['fc3'](h)
        else:
            h = F.relu(self.output_layers['fc1'](h))
            h = F.relu(self.output_layers['fc2'](h))
            h = F.relu(self.output_layers['fc3'](h))
            h = self.output_layers['fc4'](h)

        return h

    def reset_parameters(self):
        """Reset all parameters for MAML."""
        for linear in self.input_linears:
            linear.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.output_layers.values():
            layer.reset_parameters()


# ============================================================================
# Factory Functions
# ============================================================================

def create_maml_hetero_gnn_model(node_features=11, pooling='mean', output_dim=1, dropout=0.0,
                                  conv_hidden_dim=32, num_conv_layers=2, fc_hidden_dim=256,
                                  num_fc_layers=2, output_node_idx=2, num_node_types=4,
                                  conv_type='gcn', heads=4):
    """
    Create a Heterogeneous GNN model for MAML.

    Args:
        node_features: Number of input node features (default: 11 for TSMC)
        pooling: Pooling method ('mean', 'max', 'add', 'output')
        output_dim: Output dimension
        dropout: Dropout rate
        conv_hidden_dim: Hidden dimension for convolution layers
        num_conv_layers: Number of convolution layers
        fc_hidden_dim: Hidden dimension for FC layers
        num_fc_layers: Number of FC layers (1, 2, 3, or 4)
        output_node_idx: Index of output node for 'output' pooling
        num_node_types: Number of distinct node types (default: 4)
        conv_type: Convolution type ('gcn' or 'gat')
        heads: Number of attention heads (for GAT only)

    Returns:
        MAML_HeteroGNN model instance
    """
    return MAML_HeteroGNN(
        node_features=node_features,
        pooling=pooling,
        output_dim=output_dim,
        dropout=dropout,
        conv_hidden_dim=conv_hidden_dim,
        num_conv_layers=num_conv_layers,
        fc_hidden_dim=fc_hidden_dim,
        num_fc_layers=num_fc_layers,
        output_node_idx=output_node_idx,
        num_node_types=num_node_types,
        conv_type=conv_type,
        heads=heads
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing MAML Heterogeneous GNN Model...")
    print("=" * 60)

    from torch_geometric.data import Data, Batch

    # Create sample data mimicking circuit topology
    # Node features: [is_power, is_port, trans_type, width, voltage, input_slew, output_load, ...]
    num_nodes = 10
    num_features = 11

    # Create realistic node features
    node_features = torch.zeros(num_nodes, num_features)

    # Power nodes (VDD, VSS)
    node_features[0] = torch.tensor([1, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0])  # VDD
    node_features[1] = torch.tensor([1, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0])  # VSS

    # Output port
    node_features[2] = torch.tensor([0, 1, 0, 0, 0.7, 0, 0.5, 0, 0, 0, 0])

    # Intermediate port
    node_features[3] = torch.tensor([0, 1, 0, 0.3, 0.7, 0, 0, 0, 0, 0, 0])

    # Input ports
    node_features[4] = torch.tensor([0, 1, 0, 0.2, 0.7, 0.01, 0, 0, 0, 0, 0])
    node_features[5] = torch.tensor([0, 1, 0, 0.15, 0.7, 0.01, 0, 0, 0, 0, 0])

    # NMOS transistors (trans_type = 1)
    node_features[6] = torch.tensor([0, 0, 1, 0.28, 0.7, 0.01, 0, 0, 0, 0, 0])
    node_features[7] = torch.tensor([0, 0, 1, 0.14, 0.7, 0, 0, 0, 0, 0, 0])

    # PMOS transistors (trans_type = -1)
    node_features[8] = torch.tensor([0, 0, -1, 0.34, 0.7, 0.01, 0, 0, 0, 0, 0])
    node_features[9] = torch.tensor([0, 0, -1, 0.32, 0.7, 0, 0, 0, 0, 0, 0])

    # Create edges
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [8, 6, 3, 2, 6, 7, 2, 3, 0, 1, 9, 7, 8, 9]
    ], dtype=torch.long)

    # Create PyG Data
    data = Data(x=node_features, edge_index=edge_index)
    batch = Batch.from_data_list([data, data])

    print(f"Node features shape: {batch.x.shape}")
    print(f"Edge index shape: {batch.edge_index.shape}")

    # Test node type detection
    print("\n1. Testing node type detection...")
    node_types = get_node_types(batch.x)
    print(f"   Node types: {node_types[:10].tolist()}")
    print(f"   (0=Power, 1=Port, 2=NMOS, 3=PMOS)")

    # Test HeteroGNN with GCN conv
    print("\n2. Testing HeteroGNN with GCN convolution...")
    model_gcn = create_maml_hetero_gnn_model(
        node_features=num_features,
        pooling='mean',
        conv_hidden_dim=32,
        num_conv_layers=2,
        fc_hidden_dim=64,
        num_fc_layers=2,
        conv_type='gcn'
    )
    output_gcn = model_gcn(batch)
    print(f"   Output shape: {output_gcn.shape}")
    print(f"   Output values: {output_gcn.squeeze().tolist()}")

    # Test HeteroGNN with GAT conv
    print("\n3. Testing HeteroGNN with GAT convolution...")
    model_gat = create_maml_hetero_gnn_model(
        node_features=num_features,
        pooling='mean',
        conv_hidden_dim=32,
        num_conv_layers=2,
        fc_hidden_dim=64,
        num_fc_layers=2,
        conv_type='gat',
        heads=4
    )
    output_gat = model_gat(batch)
    print(f"   Output shape: {output_gat.shape}")
    print(f"   Output values: {output_gat.squeeze().tolist()}")

    # Test with output pooling
    print("\n4. Testing with output pooling...")
    model_output = create_maml_hetero_gnn_model(
        node_features=num_features,
        pooling='output',
        conv_hidden_dim=32,
        num_conv_layers=2,
        output_node_idx=2
    )
    output_out = model_output(batch)
    print(f"   Output shape: {output_out.shape}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    output_gcn.sum().backward()
    grad_exists = all(p.grad is not None for p in model_gcn.parameters() if p.requires_grad)
    print(f"   Gradients computed: {grad_exists}")

    # Model statistics
    print("\n6. Model architecture (GCN):")
    total_params = sum(p.numel() for p in model_gcn.parameters())
    trainable_params = sum(p.numel() for p in model_gcn.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print("\n7. State dict keys:")
    for key in list(model_gcn.state_dict().keys())[:10]:
        print(f"   {key}")
    print("   ...")

    print("\n" + "=" * 60)
    print("All tests passed!")
