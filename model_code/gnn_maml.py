"""
GNN models for MAML training - Optimized Version 2
Includes GCN, GraphSAGE, and GAT models with MAML compatibility
Matches the original trained model structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class MAML_GNN_Model(nn.Module):
    """Base class for MAML-compatible GNN models"""
    
    def __init__(self, node_features, hidden_dim, num_layers, pooling='mean', output_dim=1):
        super(MAML_GNN_Model, self).__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling_type = pooling
        self.output_dim = output_dim
        
        # Select pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool
            
    def reset_parameters(self):
        """Reset all parameters for MAML"""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def output_node_pool(x, batch, output_node_idx=None, default_idx=2):
    """
    Output-node-only pooling: extract only the output node embedding from each graph.

    In stage-aware topology, nodes are ordered as:
    [VDD(0), VSS(1), output(2), inputs..., transistors...]

    Args:
        x: Node embeddings [num_nodes, hidden_dim]
        batch: Batch assignment for nodes [num_nodes]
        output_node_idx: Tensor of output node indices per graph [batch_size],
                        or single int for all graphs, or None to use default
        default_idx: Default index if output_node_idx is None (default: 2)

    Returns:
        Graph-level embeddings [batch_size, hidden_dim]
    """
    # Get unique batch indices (number of graphs)
    batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1

    # Handle output_node_idx types
    if output_node_idx is None:
        output_node_idx = [default_idx] * batch_size
    elif isinstance(output_node_idx, int):
        output_node_idx = [output_node_idx] * batch_size
    elif isinstance(output_node_idx, torch.Tensor):
        output_node_idx = output_node_idx.tolist()

    # Find the starting index of each graph in the batch
    # batch tensor looks like [0,0,0,...,1,1,1,...,2,2,2,...]
    output_embeddings = []

    for graph_idx in range(batch_size):
        # Find all node indices belonging to this graph
        graph_mask = (batch == graph_idx)
        graph_node_indices = graph_mask.nonzero(as_tuple=True)[0]

        # Get the output node index for this graph
        node_idx = output_node_idx[graph_idx] if graph_idx < len(output_node_idx) else default_idx

        # Get the output node embedding
        if len(graph_node_indices) > node_idx:
            output_node_global_idx = graph_node_indices[node_idx]
            output_embeddings.append(x[output_node_global_idx])
        else:
            # Fallback: if graph has fewer nodes, use mean pooling for this graph
            output_embeddings.append(x[graph_node_indices].mean(dim=0))

    return torch.stack(output_embeddings, dim=0)


class MAML_GCN(MAML_GNN_Model):
    """GCN model for MAML - Flexible architecture with separate conv and FC configuration"""

    def __init__(self, node_features, pooling='mean', output_dim=1, dropout=0.0,
                 conv_hidden_dim=32, num_conv_layers=2, fc_hidden_dim=256, num_fc_layers=2,
                 output_node_idx=2):
        """
        Args:
            node_features: Number of input node features
            pooling: Pooling method ('mean', 'max', 'add', 'output')
                     'output' mode uses only the output node embedding
            output_dim: Output dimension (default: 1)
            dropout: Dropout rate
            conv_hidden_dim: Hidden dimension for convolution layers (default: 128)
            num_conv_layers: Number of GCN convolutional layers (default: 3)
            fc_hidden_dim: Hidden dimension for FC layers (default: 128)
            num_fc_layers: Number of FC layers (1, 2, 3, or 4) (default: 2)
            output_node_idx: Index of output node in node list (default: 2)
                            Used when pooling='output'
        """
        # Use conv_hidden_dim as the base hidden_dim for parent class
        super(MAML_GCN, self).__init__(node_features, conv_hidden_dim, num_conv_layers, pooling, output_dim)

        self.output_node_idx = output_node_idx

        self.dropout = dropout
        self.conv_hidden_dim = conv_hidden_dim
        self.num_conv_layers = num_conv_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_fc_layers = num_fc_layers

        # Create GCN layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(node_features, self.conv_hidden_dim))

        # Hidden layers
        for _ in range(num_conv_layers - 2):
            self.convs.append(GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))

        # Last conv layer
        if num_conv_layers > 1:
            self.convs.append(GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))

        # Output layers as separate modules to match saved model
        self.output_layers = nn.ModuleDict()

        if num_fc_layers == 1:
            # Single FC layer: conv_hidden -> output
            self.output_layers['fc1'] = nn.Linear(self.conv_hidden_dim, output_dim)
        elif num_fc_layers == 2:
            # Two FC layers: conv_hidden -> fc_hidden -> output
            self.output_layers['fc1'] = nn.Linear(self.conv_hidden_dim, self.fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(self.fc_hidden_dim, output_dim)
        elif num_fc_layers == 3:
            # Three FC layers: conv_hidden -> fc_hidden -> fc_hidden -> output
            self.output_layers['fc1'] = nn.Linear(self.conv_hidden_dim, self.fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim)
            self.output_layers['fc3'] = nn.Linear(self.fc_hidden_dim, output_dim)
        else:  # num_fc_layers == 4
            # Four FC layers: conv_hidden -> fc_hidden -> fc_hidden -> fc_hidden -> output
            self.output_layers['fc1'] = nn.Linear(self.conv_hidden_dim, self.fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim)
            self.output_layers['fc3'] = nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim)
            self.output_layers['fc4'] = nn.Linear(self.fc_hidden_dim, output_dim)
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass
        x: Node features [num_nodes, node_features] or Data/Batch object
        edge_index: Edge connectivity [2, num_edges]
        batch: Batch assignment for nodes
        """
        # Handle PyG Data/Batch input
        output_node_idx = None
        if hasattr(x, 'x'):
            data = x
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            # Get dynamic output_node_idx from Data object if available
            if hasattr(data, 'output_node_idx'):
                output_node_idx = data.output_node_idx

        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling: global pooling or output-node-only pooling
        if self.pooling_type == 'output':
            # Output-node-only pooling: use dynamic index from data or fallback to default
            x = output_node_pool(x, batch, output_node_idx, default_idx=self.output_node_idx)
        else:
            # Global pooling (mean, max, add)
            x = self.pool(x, batch)

        # Output layers - different paths based on num_fc_layers
        if self.num_fc_layers == 1:
            x = self.output_layers['fc1'](x)
        elif self.num_fc_layers == 2:
            x = F.relu(self.output_layers['fc1'](x))
            x = self.output_layers['fc2'](x)
        elif self.num_fc_layers == 3:
            x = F.relu(self.output_layers['fc1'](x))
            x = F.relu(self.output_layers['fc2'](x))
            x = self.output_layers['fc3'](x)
        else:  # num_fc_layers == 4
            x = F.relu(self.output_layers['fc1'](x))
            x = F.relu(self.output_layers['fc2'](x))
            x = F.relu(self.output_layers['fc3'](x))
            x = self.output_layers['fc4'](x)

        return x

class MAML_GraphSAGE(MAML_GNN_Model):
    """GraphSAGE model for MAML - Original structure"""
    
    def __init__(self, node_features, hidden_dim, num_layers, pooling='mean', output_dim=1, dropout=0.0):
        super(MAML_GraphSAGE, self).__init__(node_features, hidden_dim, num_layers, pooling, output_dim)
        
        self.dropout = dropout
        
        # Create GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Last conv layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layers as separate modules to match saved model
        self.output_layers = nn.ModuleDict()
        self.output_layers['fc1'] = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers['fc2'] = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layers['fc3'] = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass
        x: Node features [num_nodes, node_features] or Data/Batch object
        edge_index: Edge connectivity [2, num_edges]
        batch: Batch assignment for nodes
        """
        # Handle PyG Data/Batch input
        if hasattr(x, 'x'):
            data = x
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Output layers with ReLU activations
        x = F.relu(self.output_layers['fc1'](x))
        x = F.relu(self.output_layers['fc2'](x))
        x = self.output_layers['fc3'](x)
        
        return x

class MAML_GAT(MAML_GNN_Model):
    """GAT model for MAML - Flexible architecture with separate conv and FC configuration"""

    def __init__(self, node_features, pooling='mean', output_dim=1, dropout=0.0,
                 conv_hidden_dim=128, num_conv_layers=3, fc_hidden_dim=128, num_fc_layers=2,
                 heads=4, output_node_idx=2):
        """
        Args:
            node_features: Number of input node features
            pooling: Pooling method ('mean', 'max', 'add', 'output')
                     'output' mode uses only the output node embedding
            output_dim: Output dimension (default: 1)
            dropout: Dropout rate
            conv_hidden_dim: Hidden dimension for GAT layers (default: 128)
                            Note: actual per-head dim = conv_hidden_dim // heads
            num_conv_layers: Number of GAT convolutional layers (default: 3)
            fc_hidden_dim: Hidden dimension for FC layers (default: 128)
            num_fc_layers: Number of FC layers (2 or 3) (default: 2)
            heads: Number of attention heads (default: 4)
            output_node_idx: Index of output node in node list (default: 2)
                            Used when pooling='output'
        """
        super(MAML_GAT, self).__init__(node_features, conv_hidden_dim, num_conv_layers, pooling, output_dim)

        self.output_node_idx = output_node_idx
        self.dropout = dropout
        self.heads = heads
        self.conv_hidden_dim = conv_hidden_dim
        self.num_conv_layers = num_conv_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_fc_layers = num_fc_layers

        # Create GAT layers
        self.convs = nn.ModuleList()

        # First layer: node_features -> conv_hidden_dim (with multi-head)
        self.convs.append(GATConv(node_features, conv_hidden_dim // heads, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_conv_layers - 2):
            self.convs.append(GATConv(conv_hidden_dim, conv_hidden_dim // heads, heads=heads, dropout=dropout))

        # Last conv layer (single head to get conv_hidden_dim output)
        if num_conv_layers > 1:
            self.convs.append(GATConv(conv_hidden_dim, conv_hidden_dim, heads=1, dropout=dropout))

        # Output layers as separate modules
        self.output_layers = nn.ModuleDict()

        if num_fc_layers == 2:
            # Two FC layers: conv_hidden -> fc_hidden -> output
            self.output_layers['fc1'] = nn.Linear(conv_hidden_dim, fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(fc_hidden_dim, output_dim)
        elif num_fc_layers == 3:
            # Three FC layers: conv_hidden -> fc_hidden -> fc_hidden -> output
            self.output_layers['fc1'] = nn.Linear(conv_hidden_dim, fc_hidden_dim)
            self.output_layers['fc2'] = nn.Linear(fc_hidden_dim, fc_hidden_dim)
            self.output_layers['fc3'] = nn.Linear(fc_hidden_dim, output_dim)
        else:
            raise ValueError(f"num_fc_layers must be 2 or 3, got {num_fc_layers}")

    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass
        x: Node features [num_nodes, node_features] or Data/Batch object
        edge_index: Edge connectivity [2, num_edges]
        batch: Batch assignment for nodes
        """
        # Handle PyG Data/Batch input
        output_node_idx = None
        if hasattr(x, 'x'):
            data = x
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            # Get dynamic output_node_idx from Data object if available
            if hasattr(data, 'output_node_idx'):
                output_node_idx = data.output_node_idx

        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling: global pooling or output-node-only pooling
        if self.pooling_type == 'output':
            x = output_node_pool(x, batch, output_node_idx, default_idx=self.output_node_idx)
        else:
            x = self.pool(x, batch)

        # Output layers - different paths based on num_fc_layers
        if self.num_fc_layers == 2:
            x = F.relu(self.output_layers['fc1'](x))
            x = self.output_layers['fc2'](x)
        elif self.num_fc_layers == 3:
            x = F.relu(self.output_layers['fc1'](x))
            x = F.relu(self.output_layers['fc2'](x))
            x = self.output_layers['fc3'](x)

        return x

def create_maml_gcn_model(node_features=7, pooling='mean', output_dim=1, dropout=0.0,
                         conv_hidden_dim=32, num_conv_layers=2, fc_hidden_dim=256, num_fc_layers=2,
                         output_node_idx=2):
    """
    Create a GCN model for MAML

    Args:
        node_features: Number of input node features
        pooling: Pooling method ('mean', 'max', 'add', 'output')
                 'output' mode uses only the output node embedding
        output_dim: Output dimension
        dropout: Dropout rate
        conv_hidden_dim: Hidden dimension for convolution layers (default: 128)
        num_conv_layers: Number of GCN convolutional layers (default: 3)
        fc_hidden_dim: Hidden dimension for FC layers (default: 128)
        num_fc_layers: Number of FC layers (1, 2, 3, or 4) (default: 2)
        output_node_idx: Index of output node in node list (default: 2)
                        Used when pooling='output'
    """
    return MAML_GCN(node_features, pooling, output_dim, dropout,
                   conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers,
                   output_node_idx)

def create_maml_graphsage_model(node_features=7, hidden_dim=40, num_layers=3, pooling='mean', 
                               output_dim=1, dropout=0.0):
    """Create a GraphSAGE model for MAML"""
    return MAML_GraphSAGE(node_features, hidden_dim, num_layers, pooling, output_dim, dropout)

def create_maml_gat_model(node_features=7, pooling='mean', output_dim=1, dropout=0.0,
                         conv_hidden_dim=128, num_conv_layers=3, fc_hidden_dim=128, num_fc_layers=2,
                         heads=4, output_node_idx=2):
    """
    Create a GAT model for MAML

    Args:
        node_features: Number of input node features
        pooling: Pooling method ('mean', 'max', 'add', 'output')
                 'output' mode uses only the output node embedding
        output_dim: Output dimension
        dropout: Dropout rate
        conv_hidden_dim: Hidden dimension for GAT layers (default: 128)
        num_conv_layers: Number of GAT convolutional layers (default: 3)
        fc_hidden_dim: Hidden dimension for FC layers (default: 128)
        num_fc_layers: Number of FC layers (2 or 3) (default: 2)
        heads: Number of attention heads (default: 4)
        output_node_idx: Index of output node in node list (default: 2)
                        Used when pooling='output'
    """
    return MAML_GAT(node_features, pooling, output_dim, dropout,
                    conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers,
                    heads, output_node_idx)

# Test the models
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data, Batch
    
    # Create sample data
    num_nodes = 10
    num_edges = 20
    node_features = 7
    
    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data, data])  # Batch of 2 graphs
    
    # Test GCN
    print("Testing MAML GCN...")
    model_gcn = create_maml_gcn_model()
    output_gcn = model_gcn(batch)
    print(f"GCN output shape: {output_gcn.shape}")
    
    # Check state dict keys
    print("\nGCN State dict keys:")
    for key in model_gcn.state_dict().keys():
        print(f"  {key}")
    
    # Test GraphSAGE
    print("\nTesting MAML GraphSAGE...")
    model_sage = create_maml_graphsage_model()
    output_sage = model_sage(batch)
    print(f"GraphSAGE output shape: {output_sage.shape}")
    
    # Test GAT
    print("\nTesting MAML GAT...")
    model_gat = create_maml_gat_model()
    output_gat = model_gat(batch)
    print(f"GAT output shape: {output_gat.shape}")
    
    print("\n✅ All models working correctly!")

# Import HGCN for unified access
try:
    from hgcn_maml import MAML_HGCN, create_maml_hgcn_model
except ImportError:
    pass  # HGCN module not available

# Import Heterogeneous GNN for unified access
try:
    from hetero_gnn_maml import MAML_HeteroGNN, create_maml_hetero_gnn_model
except ImportError:
    pass  # HeteroGNN module not available