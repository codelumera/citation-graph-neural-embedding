"""
Models: Arsitektur GNN untuk prediksi pengaruh ilmiah.

Mendukung berbagai arsitektur:
- GraphSAGE: Sampling dan agregasi tetangga
- GAT (Graph Attention Network): Attention-based aggregation
- GCN (Graph Convolutional Network): Convolutional layers
- HeterogeneousGNN: Untuk graf multi-tipe node
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GraphSAGE, GATConv, GCNConv, SAGEConv,
    HeteroConv, Linear, BatchNorm
)
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple, List


class GNNBackbone(nn.Module):
    """
    Backbone GNN yang mendukung GraphSAGE, GAT, dan GCN.
    
    Attributes:
        in_channels: Jumlah fitur input
        hidden_channels: Ukuran hidden layer
        num_layers: Jumlah layer GNN
        model_type: Tipe model ('graphsage', 'gat', 'gcn')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 3,
        model_type: str = "graphsage",
        dropout: float = 0.3,
        heads: int = 4,
        aggregation: str = "mean"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Layer pertama
        if self.model_type == "graphsage":
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregation))
        elif self.model_type == "gat":
            self.convs.append(GATConv(
                in_channels, 
                hidden_channels // heads, 
                heads=heads,
                aggr=aggregation
            ))
        elif self.model_type == "gcn":
            self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.bns.append(BatchNorm(hidden_channels))
        
        # Layer berikutnya
        for _ in range(num_layers - 1):
            if self.model_type == "graphsage":
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation))
            elif self.model_type == "gat":
                self.convs.append(GATConv(
                    hidden_channels * heads if i == 0 else hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    aggr=aggregation
                ))
            elif self.model_type == "gcn":
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                
            self.bns.append(BatchNorm(hidden_channels))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset parameter weights."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features of shape (num_nodes, in_channels)
            edge_index: Edge indices of shape (2, num_edges)
            
        Returns:
            Node embeddings of shape (num_nodes, hidden_channels)
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # No activation/BatchNorm after last layer
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous GNN untuk graf dengan multi-tipe node.
    
    Menggunakan HeteroConv dari PyTorch Geometric untuk menangani
    berbagai tipe node dan edge dalam satu model terpadu.
    
    Attributes:
        metadata: Tuple of (node_types, edge_types)
        in_channels: Dictionary mapping node types to input channels
        hidden_channels: Size of hidden layers
    """
    
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_channels: Dict[str, int],
        hidden_channels: int = 256,
        num_layers: int = 2,
        model_type: str = "sage",
        dropout: float = 0.3,
        aggregation: str = "sum"
    ):
        super().__init__()
        
        self.metadata = metadata
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        # First layer
        conv_dict = {}
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            src_dim = in_channels.get(src_type, hidden_channels)
            
            if model_type == "sage":
                conv_dict[edge_type] = SAGEConv((src_dim, hidden_channels), hidden_channels)
            elif model_type == "gat":
                conv_dict[edge_type] = GATConv((src_dim, hidden_channels), hidden_channels // 4, heads=4)
            else:
                conv_dict[edge_type] = SAGEConv((src_dim, hidden_channels), hidden_channels)
                
        self.convs.append(HeteroConv(conv_dict, aggr=aggregation))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            conv_dict = {}
            for edge_type in metadata[1]:
                src_type, _, dst_type = edge_type
                if model_type == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_channels, hidden_channels), 
                        hidden_channels
                    )
                elif model_type == "gat":
                    conv_dict[edge_type] = GATConv(
                        (hidden_channels, hidden_channels), 
                        hidden_channels // 4, 
                        heads=4
                    )
                else:
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_channels, hidden_channels), 
                        hidden_channels
                    )
                    
            self.convs.append(HeteroConv(conv_dict, aggr=aggregation))
            
        # Output projection for each node type
        self.out_projs = nn.ModuleDict()
        for node_type in metadata[0]:
            self.out_projs[node_type] = Linear(hidden_channels, hidden_channels)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset all parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for proj in self.out_projs.values():
            proj.reset_parameters()
            
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass untuk heterogeneous graph.
        
        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge indices
            
        Returns:
            Dictionary mapping node types to embedding tensors
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: F.relu(x)
                for key, x in x_dict.items()
            }
            
        # Apply output projections
        x_dict = {
            key: self.out_projs[key](x)
            for key, x in x_dict.items()
        }
        
        return x_dict


class InfluencePredictor(nn.Module):
    """
    Model utama untuk memprediksi kedalaman pengaruh (influence depth).
    
    Menggabungkan GNN backbone dengan lapisan prediksi untuk
    menghasilkan skor pengaruh antara paper.
    
    Attributes:
        gnn: GNN backbone untuk encoding node
        predictor: MLP untuk prediksi skor
        loss_fn: Loss function untuk training
    """
    
    def __init__(
        self,
        graph_data: HeteroData,
        hidden_channels: int = 256,
        num_layers: int = 3,
        model_type: str = "hetero",
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.graph_data = graph_data
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine input channels per node type
        in_channels = {}
        for node_type in graph_data.node_types:
            if hasattr(graph_data[node_type], 'x'):
                in_channels[node_type] = graph_data[node_type].x.shape[1]
            else:
                in_channels[node_type] = hidden_channels  # Default
                
        # Initialize GNN
        if model_type == "hetero":
            metadata = (
                list(graph_data.node_types),
                list(graph_data.edge_types)
            )
            self.gnn = HeterogeneousGNN(
                metadata=metadata,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            # Use homogeneous GNN for paper nodes only
            paper_channels = in_channels.get('paper', hidden_channels)
            self.gnn = GNNBackbone(
                in_channels=paper_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                model_type=model_type,
                dropout=dropout
            )
            
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.loss_fn = nn.BCELoss()
        self.to(self.device)
        
    def forward(
        self,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        x_dict: Optional[Dict[str, torch.Tensor]] = None,
        edge_index_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict influence score between source and target papers.
        
        Args:
            source_idx: Indices of source papers
            target_idx: Indices of target papers
            x_dict: Node features dictionary (for heterogeneous GNN)
            edge_index_dict: Edge indices dictionary (for heterogeneous GNN)
            
        Returns:
            Influence scores between 0 and 1
        """
        # Get embeddings
        if isinstance(self.gnn, HeterogeneousGNN):
            if x_dict is None or edge_index_dict is None:
                x_dict = {nt: self.graph_data[nt].x for nt in self.graph_data.node_types}
                edge_index_dict = {
                    et: self.graph_data[et].edge_index 
                    for et in self.graph_data.edge_types
                }
            embeddings = self.gnn(x_dict, edge_index_dict)
            source_emb = embeddings['paper'][source_idx]
            target_emb = embeddings['paper'][target_idx]
        else:
            x = self.graph_data['paper'].x
            edge_index = self.graph_data['paper', 'cites', 'paper'].edge_index
            embeddings = self.gnn(x, edge_index)
            source_emb = embeddings[source_idx]
            target_emb = embeddings[target_idx]
            
        # Concatenate and predict
        combined = torch.cat([source_emb, target_emb], dim=-1)
        scores = self.predictor(combined)
        
        return scores.squeeze(-1)
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        positive_edges: torch.Tensor,
        negative_edges: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            optimizer: Optimizer
            positive_edges: Tensor of shape (num_pos, 2) with positive edges
            negative_edges: Tensor of shape (num_neg, 2) with negative edges
            batch_size: Training batch size
            
        Returns:
            Average loss for the epoch
        """
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        # Combine positive and negative samples
        pos_labels = torch.ones(positive_edges.shape[0], device=self.device)
        neg_labels = torch.zeros(negative_edges.shape[0], device=self.device)
        
        all_edges = torch.cat([positive_edges, negative_edges], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Shuffle
        perm = torch.randperm(all_edges.shape[0])
        all_edges = all_edges[perm]
        all_labels = all_labels[perm]
        
        for i in range(0, all_edges.shape[0], batch_size):
            batch_edges = all_edges[i:i + batch_size]
            batch_labels = all_labels[i:i + batch_size]
            
            source_idx = batch_edges[:, 0].to(self.device)
            target_idx = batch_edges[:, 1].to(self.device)
            
            optimizer.zero_grad()
            predictions = self.forward(source_idx, target_idx)
            loss = self.loss_fn(predictions, batch_labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(
        self,
        positive_edges: torch.Tensor,
        negative_edges: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            positive_edges: Positive edge indices
            negative_edges: Negative edge indices
            batch_size: Evaluation batch size
            
        Returns:
            Dictionary with metrics (loss, accuracy, etc.)
        """
        self.eval()
        
        all_preds = []
        all_labels = []
        
        for edges, label in [(positive_edges, 1), (negative_edges, 0)]:
            for i in range(0, edges.shape[0], batch_size):
                batch_edges = edges[i:i + batch_size]
                source_idx = batch_edges[:, 0].to(self.device)
                target_idx = batch_edges[:, 1].to(self.device)
                
                preds = self.forward(source_idx, target_idx)
                all_preds.append(preds.cpu())
                all_labels.extend([label] * len(batch_edges))
                
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.float)
        
        loss = self.loss_fn(all_preds, all_labels)
        accuracy = ((all_preds > 0.5).float() == all_labels).float().mean()
        
        return {"loss": loss.item(), "accuracy": accuracy.item()}
