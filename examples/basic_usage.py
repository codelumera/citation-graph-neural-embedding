"""
Contoh penggunaan dasar Semantic Echo.

Script ini mendemonstrasikan cara menggunakan library untuk:
1. Membuat graf sitasi dari data
2. Melatih model GNN
3. Menghitung Echo Score dan Influence Depth
"""

import numpy as np
from semantic_echo import (
    CitationGraph,
    InfluencePredictor,
    SciBERTEmbedding,
    echo_score,
    disruption_index,
    influence_depth
)
from semantic_echo.graph import Node, Edge


def example_basic_usage():
    """Contoh penggunaan dasar dengan data dummy."""
    print("=" * 60)
    print("CONTOH PENGGUNAAN DASAR SEMANTIC ECHO")
    print("=" * 60)
    
    # 1. Buat graf secara manual dengan data dummy
    print("\n1. Membuat graf sitasi...")
    graph = CitationGraph()
    
    embedding_dim = 768
    
    # Tambahkan beberapa paper nodes
    papers = [
        ("10.1000/paper1", "Deep Learning Fundamentals", 2018),
        ("10.1000/paper2", "Advanced Neural Networks", 2019),
        ("10.1000/paper3", "Transformers in NLP", 2020),
        ("10.1000/paper4", "Graph Neural Networks Survey", 2021),
    ]
    
    for doi, title, year in papers:
        # Generate random embedding (in practice, use SciBERT)
        features = np.random.randn(embedding_dim).astype(np.float32)
        
        node = Node(
            id=doi,
            type='paper',
            features=features,
            metadata={'title': title, 'year': year}
        )
        graph.add_node(node)
    
    # Tambahkan edge sitasi
    edges = [
        ("10.1000/paper2", "10.1000/paper1"),  # Paper 2 cites Paper 1
        ("10.1000/paper3", "10.1000/paper1"),  # Paper 3 cites Paper 1
        ("10.1000/paper4", "10.1000/paper2"),  # Paper 4 cites Paper 2
        ("10.1000/paper4", "10.1000/paper3"),  # Paper 4 cites Paper 3
    ]
    
    for src, dst in edges:
        graph.add_edge(Edge(src=src, dst=dst, relation='cites'))
    
    # Finalisasi graf
    hetero_data = graph.finalize()
    
    print(f"   - Jumlah paper: {graph.num_nodes('paper')}")
    print(f"   - Jumlah edge sitasi: {graph.num_edges('cites')}")
    
    # 2. Hitung Echo Score antara dua paper
    print("\n2. Menghitung Echo Score...")
    # Akses embedding dari hetero_data setelah finalize
    paper_features = hetero_data['paper'].x  # Shape: [num_papers, embedding_dim]
    emb1 = paper_features[0].numpy() if hasattr(paper_features[0], 'numpy') else paper_features[0]
    emb2 = paper_features[1].numpy() if hasattr(paper_features[1], 'numpy') else paper_features[1]
    
    score = echo_score(emb1, emb2)
    print(f"   - Echo Score antara Paper 1 dan 2: {score:.4f}")
    
    # 3. Hitung Influence Depth
    print("\n3. Menghitung Influence Depth...")
    depth = influence_depth(
        source_embedding=emb1,
        target_embedding=emb2,
        source_year=2018,
        target_year=2019,
        citation_count=50
    )
    print(f"   - Influence Depth: {depth:.4f}")
    
    # 4. Latih model predictor (contoh sederhana)
    print("\n4. Melatih Influence Predictor...")
    try:
        predictor = InfluencePredictor(
            graph_data=hetero_data,
            hidden_channels=128,
            num_layers=2,
            model_type='hetero'
        )
        
        # Create dummy training data
        positive_edges = np.array([[0, 1], [1, 2], [2, 3]])
        negative_edges = np.array([[0, 3], [1, 3], [0, 2]])
        
        import torch
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
        
        # Train for a few epochs
        for epoch in range(5):
            loss = predictor.train_epoch(
                optimizer=optimizer,
                positive_edges=torch.tensor(positive_edges),
                negative_edges=torch.tensor(negative_edges),
                batch_size=2
            )
            print(f"   - Epoch {epoch + 1}/5, Loss: {loss:.4f}")
            
        print("   ✓ Training selesai!")
        
    except Exception as e:
        print(f"   ⚠ Training dilewati (memerlukan PyTorch Geometric): {e}")
    
    print("\n" + "=" * 60)
    print("CONTOH SELESAI")
    print("=" * 60)


def example_with_sciibert():
    """Contoh menggunakan SciBERT embeddings (jika tersedia)."""
    print("\n" + "=" * 60)
    print("CONTOH DENGAN SCIBERT (Opsional)")
    print("=" * 60)
    
    try:
        # Initialize SciBERT
        embedder = SciBERTEmbedding(
            model_name="allenai/scibert_scivocab_uncased",
            pooling="cls"
        )
        
        # Encode some paper titles/abstracts
        texts = [
            "Deep Learning for Natural Language Processing",
            "Graph Neural Networks: A Comprehensive Survey",
            "Transformer-based Models for Scientific Text Mining"
        ]
        
        print("\nEncoding teks ilmiah dengan SciBERT...")
        embeddings = embedder.encode(texts, show_progress=True)
        
        print(f"\nShape embeddings: {embeddings.shape}")
        
        # Calculate similarity matrix
        sim_matrix = embedder.similarity(texts, texts)
        
        print("\nSimilarity Matrix:")
        for i, t1 in enumerate(texts):
            for j, t2 in enumerate(texts):
                if i <= j:
                    print(f"  {t1[:30]:<30} vs {t2[:30]:<30}: {sim_matrix[i][j]:.4f}")
        
        print("\n✓ SciBERT contoh selesai!")
        
    except Exception as e:
        print(f"\n⚠ SciBERT tidak tersedia (install transformers): {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    example_basic_usage()
    example_with_sciibert()
    
    print("\n💡 Untuk contoh lebih lanjut, lihat dokumentasi di README.md")
