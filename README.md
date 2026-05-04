# 🧬 Semantic Echo: Representasi Vektor Dampak Karya Ilmiah

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3b78b8.svg)](https://www.pyg.org/)

> *"Sitasi adalah percakapan. AI seharusnya membaca nada dan jedanya, bukan hanya menghitung volume suaranya."*

## 📖 Deskripsi

**Semantic Echo** adalah framework analisis sitasi berbasis **Graph Neural Networks (GNN)** yang mengukur **kedalaman pengaruh konseptual** (`influence depth`) antar karya ilmiah. Berbeda dengan metrik tradisional seperti H-index atau jumlah sitasi yang hanya menghitung kuantitas, Semantic Echo menganalisis **kualitas dan kedalaman pengaruh** dengan cara membandingkan perubahan vektor representasi makalah sebelum dan setelah sitasi terjadi.

### 🎯 Masalah yang Diselesaikan

Metrik sitasi konvensional tidak dapat membedakan antara:
- **Kutipan perfunctory** (sekilas/formalitas) 
- **Kutipan transformatif** (yang mengubah arah penelitian fundamental)

Semantic Echo menjawab pertanyaan: *"Seberapa banyak 'DNA konseptual' Paper A mengubah vektor penelitian Paper B?"*

### 🧠 Pendekatan Teknis

1. **Konstruksi Graf Heterogen Dinamis**:
   - **Node**: Makalah, Penulis, Jurnal, Institusi, Kata Kunci (Keyphrase)
   - **Edge**: Sitasi, Co-Authorship, Publikasi di, Afiliasi, Kesamaan Semantik
   - Mengadopsi pendekatan **Heterogeneous Dynamical Graph Neural Network (SI-HDGNN)** untuk graf akademik berbobot, terarah, dan teratribusi

2. **Arsitektur GNN**:
   - Menggunakan **GraphSAGE** atau **GAT (Graph Attention Networks)** untuk agregasi tetangga
   - **Node Features**: Embedding teks dari abstrak (menggunakan **SciBERT**), fitur struktural (degree centrality), dan fitur temporal (tahun publikasi)
   - Menghasilkan *vectorized representations* untuk setiap node yang dapat di-train

3. **Metrik "Semantic Echo"**:
   - Mengukur **cosine similarity** antara vektor Paper A pada waktu `t` dengan vektor Paper B pada waktu `t+n`
   - Bukan sekadar prediksi link, tetapi **perubahan embedding space** yang disebabkan oleh kemunculan suatu karya

## 🚀 Instalasi Cepat

```bash
# 1. Clone repositori
git clone https://github.com/stipwunaraha/citation-graph-neural-embedding.git
cd citation-graph-neural-embedding

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instal semua dependensi
pip install -r requirements.txt

# 4. Instal package untuk pengembangan (opsional)
pip install -e .
```

## 💻 Penggunaan Dasar

```python
from semantic_echo import CitationGraph, InfluencePredictor

# Muat dataset (misal: subset DBLP atau OpenAlex)
graph = CitationGraph.from_openalex(doi_list=["10.xxx/paper1", "10.xxx/paper2"])

# Latih model GNN
predictor = InfluencePredictor(graph)
predictor.train(epochs=100)

# Dapatkan "Echo Score" antara dua makalah
score = predictor.echo_score(source_doi="10.xxx/paper1", target_doi="10.xxx/paper2")
print(f"Kedalaman pengaruh: {score:.4f}")
```

## 📦 Struktur Repositori

```
citation-graph-neural-embedding/
├── semantic_echo/       # Source code utama
│   ├── __init__.py      # Inisialisasi package
│   ├── data_loader.py   # Loader dataset (OpenAlex, arXiv, DBLP)
│   ├── embeddings.py    # Pembangkitan embedding (SciBERT, dll.)
│   ├── graph.py         # Konstruksi dan manipulasi graf
│   ├── models.py        # Model GNN (GraphSAGE, GAT, dll.)
│   └── metrics.py       # Metrik evaluasi (Echo Score, dll.)
├── tests/               # Unit test dan integration test
├── configs/             # File konfigurasi YAML
├── examples/            # Contoh penggunaan dan notebook
├── utils/               # Fungsi utilitas tambahan
├── requirements.txt     # Daftar dependensi Python
├── setup.py            # Setup script untuk instalasi package
└── docs/               # Dokumentasi lengkap
```

## 📊 Dataset Target

- **OpenAlex**: API terbuka dengan data sitasi lengkap
- **arXiv + Semantic Scholar**: Untuk data teks dan graf sitasi
- **DBLP**: Data publikasi ilmu komputer

## 🛠️ Teknologi Utama

| Kategori | Library/Framework |
|----------|------------------|
| Deep Learning | PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.3 |
| NLP | Transformers (SciBERT), Sentence Transformers |
| Scientific Computing | NumPy, SciPy, Pandas |
| Visualisasi | Matplotlib, NetworkX |
| Testing | pytest, pytest-cov |
| Development | black, flake8 |

## 🗺️ Roadmap

- [x] ✅ Implementasi dasar GCN/GAT dengan PyTorch Geometric
- [x] ✅ Integrasi SciBERT untuk node features
- [ ] 🔄 Skrip scraper untuk OpenAlex
- [ ] 🔄 Evaluasi terhadap metrik "Disruption Index" sebagai baseline
- [ ] 📋 Dashboard visualisasi interaktif
- [ ] 📋 API REST untuk akses programatik

## 🤝 Kontribusi

Kami sangat terbuka untuk kontribusi! Cara memulai:
1. Baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan detail
2. Cek [Issues](https://github.com/stipwunaraha/citation-graph-neural-embedding/issues) untuk tugas yang tersedia
3. Fork repositori dan buat pull request

## 📄 Lisensi

Proyek ini dilisensikan di bawah **Lisensi MIT** - lihat berkas [LICENSE](LICENSE) untuk detail.

---

**Penulis & Kontributor**: [@stipwunaraha](https://github.com/stipwunaraha) dan komunitas

**Kata Kunci**: Graph Neural Networks, Citation Analysis, Academic Impact, Semantic Similarity, PyTorch, Scientometrics
