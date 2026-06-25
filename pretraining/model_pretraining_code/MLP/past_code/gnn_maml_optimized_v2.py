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

class MAML_GCN(MAML_GNN_Model):
    """GCN model for MAML - Original structure matching trained weights"""
    
    def __init__(self, node_features, hidden_dim, num_layers, pooling='mean', output_dim=1, dropout=0.0):
        super(MAML_GCN, self).__init__(node_features, hidden_dim, num_layers, pooling, output_dim)
        
        self.dropout = dropout
        
        # Create GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Last conv layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layers as separate modules to match saved model
        self.output_layers = nn.ModuleDict()
        self.output_layers['fc1'] = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers['fc2'] = nn.Linear(hidden_dim, hidden_dim)
        self.output_layers['fc3'] = nn.Linear(hidden_dim, output_dim)
        
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
        
        # GCN layers
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
    """GAT model for MAML - Original structure"""
    
    def __init__(self, node_features, hidden_dim, num_layers, pooling='mean', output_dim=1, 
                 dropout=0.0, heads=4):
        super(MAML_GAT, self).__init__(node_features, hidden_dim, num_layers, pooling, output_dim)
        
        self.dropout = dropout
        self.heads = heads
        
        # Create GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(node_features, hidden_dim // heads, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        
        # Last conv layer (single head)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        
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
        
        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Output layers with ReLU activations
        x = F.relu(self.output_layers['fc1'](x))
        x = F.relu(self.output_layers['fc2'](x))
        x = self.output_layers['fc3'](x)
        
        return x

def create_maml_gcn_model(node_features=7, hidden_dim=40, num_layers=3, pooling='mean', 
                         output_dim=1, dropout=0.0):
    """Create a GCN model for MAML"""
    return MAML_GCN(node_features, hidden_dim, num_layers, pooling, output_dim, dropout)

def create_maml_graphsage_model(node_features=7, hidden_dim=40, num_layers=3, pooling='mean', 
                               output_dim=1, dropout=0.0):
    """Create a GraphSAGE model for MAML"""
    return MAML_GraphSAGE(node_features, hidden_dim, num_layers, pooling, output_dim, dropout)

def create_maml_gat_model(node_features=7, hidden_dim=40, num_layers=3, pooling='mean', 
                         output_dim=1, dropout=0.0, heads=4):
    """Create a GAT model for MAML"""
    return MAML_GAT(node_features, hidden_dim, num_layers, pooling, output_dim, dropout, heads)

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