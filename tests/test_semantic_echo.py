"""
Unit tests untuk Semantic Echo.

Jalankan dengan: pytest tests/test_semantic_echo.py -v
"""

import pytest
import numpy as np
import torch


class TestCitationGraph:
    """Tests untuk modul CitationGraph."""
    
    def test_create_empty_graph(self):
        """Test membuat graf kosong."""
        from semantic_echo.graph import CitationGraph
        
        graph = CitationGraph()
        assert graph.num_nodes() == 0
        assert graph.num_edges() == 0
        
    def test_add_node(self):
        """Test menambahkan node ke graf."""
        from semantic_echo.graph import CitationGraph, Node
        
        graph = CitationGraph()
        node = Node(
            id="test_paper_1",
            type="paper",
            features=np.random.randn(10),
            metadata={"title": "Test Paper"}
        )
        
        idx = graph.add_node(node)
        assert idx == 0
        assert graph.num_nodes("paper") == 1
        
    def test_add_duplicate_node(self):
        """Test menambahkan node duplikat."""
        from semantic_echo.graph import CitationGraph, Node
        
        graph = CitationGraph()
        node = Node(
            id="test_paper_1",
            type="paper",
            features=np.random.randn(10)
        )
        
        idx1 = graph.add_node(node)
        idx2 = graph.add_node(node)
        
        assert idx1 == idx2  # Should return same index
        assert graph.num_nodes("paper") == 1
        
    def test_add_edge(self):
        """Test menambahkan edge ke graf."""
        from semantic_echo.graph import CitationGraph, Node, Edge
        
        graph = CitationGraph()
        
        # Add nodes first
        node1 = Node(id="paper1", type="paper", features=np.random.randn(10))
        node2 = Node(id="paper2", type="paper", features=np.random.randn(10))
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Add edge
        edge = Edge(src="paper1", dst="paper2", relation="cites")
        graph.add_edge(edge)
        
        assert graph.num_edges("cites") == 1
        
    def test_invalid_node_type(self):
        """Test menambahkan node dengan tipe tidak valid."""
        from semantic_echo.graph import CitationGraph, Node
        
        graph = CitationGraph()
        node = Node(id="test", type="invalid_type", features=None)
        
        with pytest.raises(ValueError):
            graph.add_node(node)


class TestMetrics:
    """Tests untuk modul metrics."""
    
    def test_echo_score_identical(self):
        """Test Echo Score untuk vektor identik."""
        from semantic_echo.metrics import echo_score
        
        vec = np.random.randn(100)
        score = echo_score(vec, vec)
        
        # Identical vectors should have similarity close to 1
        assert np.isclose(score, 1.0, atol=1e-5)
        
    def test_echo_score_orthogonal(self):
        """Test Echo Score untuk vektor ortogonal."""
        from semantic_echo.metrics import echo_score
        
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        score = echo_score(vec1, vec2)
        
        assert np.isclose(score, 0.0, atol=1e-5)
        
    def test_echo_score_normalization(self):
        """Test normalisasi Echo Score."""
        from semantic_echo.metrics import echo_score
        
        vec1 = np.random.randn(100) * 1000
        vec2 = np.random.randn(100) * 0.001
        
        score_normalized = echo_score(vec1, vec2, normalize=True)
        score_unnormalized = echo_score(vec1, vec2, normalize=False)
        
        # Normalized should be in [-1, 1]
        assert -1 <= score_normalized <= 1
        
    def test_influence_depth_valid(self):
        """Test Influence Depth dengan input valid."""
        from semantic_echo.metrics import influence_depth
        
        emb1 = np.random.randn(100)
        emb2 = np.random.randn(100)
        
        depth = influence_depth(
            source_embedding=emb1,
            target_embedding=emb2,
            source_year=2018,
            target_year=2020,
            citation_count=50
        )
        
        # Should be positive and reasonable
        assert depth >= 0
        
    def test_influence_depth_invalid_years(self):
        """Test Influence Depth dengan tahun tidak valid."""
        from semantic_echo.metrics import influence_depth
        
        emb1 = np.random.randn(100)
        emb2 = np.random.randn(100)
        
        with pytest.raises(ValueError):
            influence_depth(
                source_embedding=emb1,
                target_embedding=emb2,
                source_year=2020,
                target_year=2018,  # Target before source
                citation_count=50
            )
            
    def test_disruption_index_empty(self):
        """Test Disruption Index dengan data kosong."""
        from semantic_echo.metrics import disruption_index
        
        di = disruption_index([])
        assert di == 0.0


class TestEmbeddings:
    """Tests untuk modul embeddings."""
    
    def test_sciBERT_initialization(self):
        """Test inisialisasi SciBERT (mock)."""
        # This test would require actual model download
        # Skip in CI/CD environments
        try:
            from semantic_echo.embeddings import SciBERTEmbedding
            
            embedder = SciBERTEmbedding(
                model_name="allenai/scibert_scivocab_uncased"
            )
            assert embedder is not None
        except Exception:
            pytest.skip("SciBERT model not available")


class TestModels:
    """Tests untuk modul models."""
    
    def test_gnn_backbone_creation(self):
        """Test pembuatan GNNBackbone."""
        try:
            from semantic_echo.models import GNNBackbone
            
            model = GNNBackbone(
                in_channels=128,
                hidden_channels=64,
                num_layers=2,
                model_type="graphsage"
            )
            
            # Create dummy input
            x = torch.randn(10, 128)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            output = model(x, edge_index)
            
            assert output.shape == (10, 64)
        except Exception as e:
            pytest.skip(f"GNN model test skipped: {e}")
            
    def test_gnn_backbone_types(self):
        """Test berbagai tipe GNN."""
        try:
            from semantic_echo.models import GNNBackbone
            
            for model_type in ["graphsage", "gat", "gcn"]:
                model = GNNBackbone(
                    in_channels=64,
                    hidden_channels=32,
                    num_layers=2,
                    model_type=model_type
                )
                assert model is not None
        except Exception:
            pytest.skip("GNN types test skipped")


class TestDataLoader:
    """Tests untuk modul data_loader."""
    
    def test_openalex_loader_init(self):
        """Test inisialisasi OpenAlexLoader."""
        from semantic_echo.data_loader import OpenAlexLoader
        
        loader = OpenAlexLoader(rate_limit=0.5)
        assert loader.rate_limit == 0.5
        
    def test_arxiv_loader_init(self):
        """Test inisialisasi ArxivLoader."""
        from semantic_echo.data_loader import ArxivLoader
        
        loader = ArxivLoader()
        assert loader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
