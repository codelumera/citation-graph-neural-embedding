buat desain kode lengkap untuk repo Semantic Echo ini. Berikut adalah ringkasan struktur yang telah dibuat:
📁 Struktur Proyek

/workspace/
├── README.md                 # Dokumentasi utama (sudah ada)
├── CONTRIBUTING.md           # Panduan kontribusi (baru)
├── requirements.txt          # Dependensi Python (baru)
├── configs/
│   └── config.yaml          # Konfigurasi hyperparameter (baru)
├── semantic_echo/           # Package utama
│   ├── __init__.py          # Export public API
│   ├── graph.py             # Konstruksi graf heterogen
│   ├── models.py            # GNN architectures (GraphSAGE, GAT, HeteroGNN)
│   ├── embeddings.py        # SciBERT & text embeddings
│   ├── metrics.py           # Echo Score, Disruption Index
│   └── data_loader.py       # OpenAlex & arXiv loaders
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Utility functions
├── examples/
│   └── basic_usage.py       # Contoh penggunaan
├── tests/
│   └── test_semantic_echo.py # Unit tests
├── data/                    # Direktori untuk dataset
└── models/                  # Direktori untuk saved models

🎯 Fitur Utama yang Diimplementasikan
CitationGraph - Graf heterogen dengan node (Paper, Author, Journal, Institution, Keyword) dan edge (cites, co-authors, dll.)

GNN Models:
GNNBackbone - Support GraphSAGE, GAT, GCN
HeterogeneousGNN - Untuk graf multi-tipe node
InfluencePredictor - Model utama untuk prediksi influence
SciBERT Embeddings - Integrasi transformer untuk representasi teks ilmiah

Metrics:
echo_score() - Mengukur kesamaan semantik antar paper
disruption_index() - Mengukur tingkat disruption karya ilmiah
influence_depth() - Metrik komprehensif dengan temporal decay
Data Loaders - OpenAlex dan arXiv API integration
Testing - Unit tests lengkap untuk semua modul

Desain ini mengikuti roadmap di README.md dan siap untuk dikembangkan lebih lanjut!