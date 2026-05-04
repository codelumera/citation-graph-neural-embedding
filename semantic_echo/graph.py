"""
CitationGraph: Konstruksi graf heterogen untuk analisis sitasi.

Mendukung berbagai tipe node (Paper, Author, Journal, Institution, Keyword)
dan edge (cites, co-authors, published_in, affiliated_with, similar_to).
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Node:
    """Representasi node dalam graf akademik."""
    id: str
    type: str  # paper, author, journal, institution, keyword
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Representasi edge dalam graf akademik."""
    src: str
    dst: str
    relation: str  # cites, co_authors, published_in, affiliated_with, similar_to
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CitationGraph:
    """
    Graf heterogen untuk merepresentasikan jaringan sitasi akademik.
    
    Attributes:
        hetero_data: PyTorch Geometric HeteroData object
        node_map: Mapping dari ID node ke index
        edge_index_map: Mapping untuk edge indices
    """
    
    NODE_TYPES = ['paper', 'author', 'journal', 'institution', 'keyword']
    EDGE_TYPES = [
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
        ('paper', 'published_in', 'journal'),
        ('author', 'affiliated_with', 'institution'),
        ('paper', 'has_keyword', 'keyword'),
        ('author', 'co_authors', 'author'),
        ('paper', 'similar_to', 'paper'),
    ]
    
    def __init__(self):
        self.hetero_data = HeteroData()
        self.node_map: Dict[str, Dict[str, int]] = {nt: {} for nt in self.NODE_TYPES}
        self.node_counter: Dict[str, int] = {nt: 0 for nt in self.NODE_TYPES}
        self.edges: List[Edge] = []
        
    @classmethod
    def from_openalex(cls, doi_list: List[str], **kwargs) -> 'CitationGraph':
        """
        Membuat graf dari daftar DOI menggunakan OpenAlex API.
        
        Args:
            doi_list: List of DOI strings
            **kwargs: Additional arguments for data loading
            
        Returns:
            CitationGraph instance
        """
        from .data_loader import OpenAlexLoader
        loader = OpenAlexLoader(**kwargs)
        return loader.build_graph(doi_list)
    
    @classmethod
    def from_arxiv(cls, arxiv_ids: List[str], **kwargs) -> 'CitationGraph':
        """
        Membuat graf dari daftar arXiv IDs.
        
        Args:
            arxiv_ids: List of arXiv ID strings
            
        Returns:
            CitationGraph instance
        """
        from .data_loader import ArxivLoader
        loader = ArxivLoader(**kwargs)
        return loader.build_graph(arxiv_ids)
    
    def add_node(self, node: Node) -> int:
        """
        Menambahkan node ke graf.
        
        Args:
            node: Node object to add
            
        Returns:
            Index of the added node
        """
        if node.type not in self.NODE_TYPES:
            raise ValueError(f"Unknown node type: {node.type}")
            
        if node.id in self.node_map[node.type]:
            return self.node_map[node.type][node.id]
            
        idx = self.node_counter[node.type]
        self.node_map[node.type][node.id] = idx
        self.node_counter[node.type] += 1
        
        # Initialize or extend feature matrix
        if node.features is not None:
            if f'{node.type}.x' not in self.hetero_data:
                self.hetero_data[node.type].x = []
            self.hetero_data[node.type].x.append(node.features)
            
        return idx
    
    def add_edge(self, edge: Edge) -> None:
        """
        Menambahkan edge ke graf.
        
        Args:
            edge: Edge object to add
        """
        self.edges.append(edge)
        
    def _build_edge_index(self, src_type: str, relation: str, dst_type: str) -> torch.Tensor:
        """
        Membangun edge index tensor untuk tipe edge tertentu.
        
        Args:
            src_type: Source node type
            relation: Edge relation type
            dst_type: Destination node type
            
        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        edge_indices = []
        for edge in self.edges:
            if (edge.relation == relation and 
                self._get_node_type(edge.src) == src_type and
                self._get_node_type(edge.dst) == dst_type):
                
                src_idx = self.node_map[src_type].get(edge.src)
                dst_idx = self.node_map[dst_type].get(edge.dst)
                
                if src_idx is not None and dst_idx is not None:
                    edge_indices.append([src_idx, dst_idx])
                    
        if len(edge_indices) == 0:
            return torch.tensor([[], []], dtype=torch.long)
            
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    def _get_node_type(self, node_id: str) -> Optional[str]:
        """Menentukan tipe node dari ID."""
        for node_type, node_map in self.node_map.items():
            if node_id in node_map:
                return node_type
        return None
    
    def finalize(self) -> HeteroData:
        """
        Finalisasi graf dan kembalikan HeteroData.
        
        Returns:
            PyTorch Geometric HeteroData object
        """
        # Build edge indices for all edge types
        for src_type, relation, dst_type in self.EDGE_TYPES:
            edge_index = self._build_edge_index(src_type, relation, dst_type)
            if edge_index.numel() > 0:
                self.hetero_data[src_type, relation, dst_type].edge_index = edge_index
                
        # Convert feature lists to tensors
        for node_type in self.NODE_TYPES:
            if f'{node_type}.x' in self.hetero_data:
                features = self.hetero_data[node_type].x
                if len(features) > 0:
                    self.hetero_data[node_type].x = torch.stack(
                        [torch.tensor(f, dtype=torch.float) for f in features]
                    )
                    
        return self.hetero_data
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Konversi ke NetworkX graph untuk visualisasi.
        
        Returns:
            NetworkX DiGraph
        """
        return to_networkx(self.hetero_data, to_undirected=False)
    
    def num_nodes(self, node_type: Optional[str] = None) -> int:
        """
        Menghitung jumlah node.
        
        Args:
            node_type: Specific node type, or None for total count
            
        Returns:
            Number of nodes
        """
        if node_type:
            return self.node_counter.get(node_type, 0)
        return sum(self.node_counter.values())
    
    def num_edges(self, relation: Optional[str] = None) -> int:
        """
        Menghitung jumlah edge.
        
        Args:
            relation: Specific relation type, or None for total count
            
        Returns:
            Number of edges
        """
        if relation:
            return sum(1 for e in self.edges if e.relation == relation)
        return len(self.edges)
    
    def get_node_ids(self, node_type: str) -> List[str]:
        """
        Mendapatkan semua ID node untuk tipe tertentu.
        
        Args:
            node_type: Node type
            
        Returns:
            List of node IDs
        """
        return list(self.node_map.get(node_type, {}).keys())
