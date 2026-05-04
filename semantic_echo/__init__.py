"""
Semantic Echo: Representasi Vektor Dampak Karya Ilmiah

Package utama untuk konstruksi graf sitasi, model GNN, dan metrik pengaruh semantik.
"""

from .graph import CitationGraph
from .models import GNNBackbone, HeterogeneousGNN, InfluencePredictor
from .embeddings import SciBERTEmbedding
from .metrics import echo_score, disruption_index, influence_depth
from .data_loader import OpenAlexLoader, ArxivLoader

__version__ = "0.1.0"
__author__ = "Semantic Echo Team"

__all__ = [
    "CitationGraph",
    "GNNBackbone",
    "HeterogeneousGNN",
    "InfluencePredictor",
    "SciBERTEmbedding",
    "echo_score",
    "disruption_index",
    "influence_depth",
    "OpenAlexLoader",
    "ArxivLoader",
]
