# 🧬 Semantic Echo: Representasi Vektor Dampak Karya Ilmiah

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3b78b8.svg)](https://www.pyg.org/)

> *"Sitasi adalah percakapan. AI seharusnya membaca nada dan jedanya, bukan hanya menghitung volume suaranya."*

**Semantic Echo** melampaui analisis sitasi tradisional yang hanya menghitung *edge* (siapa mengutip siapa). Proyek ini menggunakan **Graph Neural Networks (GNN)** untuk mengukur **kedalaman pengaruh** (`influence depth`): Seberapa banyak "DNA konseptual" *Paper A* mengubah vektor penelitian *Paper B*.

### 🎯 Mengapa Ini Penting?
Metrik seperti H-index atau jumlah sitasi tidak bisa membedakan antara kutipan "sekilas" (*perfunctory citation*) dengan kutipan yang menunjukkan perubahan arah riset fundamental. Dengan merepresentasikan makalah sebagai *node* dalam graf heterogen yang kaya fitur, kita bisa mengukur *semantic influence* yang sebenarnya.

### 🧠 Pendekatan Teknis
1.  **Konstruksi Graf Heterogen Dinamis**:
    - Node: Makalah, Penulis, Jurnal, Institusi, Kata Kunci (Keyphrase).
    - Edge: Sitasi, Co-Authorship, Publikasi di, Afiliasi, Kesamaan Semantik.
    - Model ini mengadopsi pendekatan serupa dengan **Heterogeneous Dynamical Graph Neural Network (SI-HDGNN)** untuk memodelkan graf akademik yang berbobot, terarah, dan teratribusi.

2.  **Arsitektur GNN**:
    - Menggunakan **GraphSAGE** atau **GAT (Graph Attention Networks)** untuk agregasi tetangga.
    - **Node Features**: *Embedding* teks dari abstrak (menggunakan **SciBERT**), fitur struktural (degree centrality), dan fitur temporal (tahun publikasi).
    - Tujuannya adalah menghasilkan *vectorized representations* untuk setiap node yang dapat di-train untuk memprediksi pengaruh.

3.  **Metrik "Semantic Echo"**:
    - Bukan hanya skor prediksi link, tetapi **cosine similarity** antara vektor *Paper A* pada waktu `t` dengan vektor *Paper B* pada waktu `t+n`.
    - Mengukur perubahan *embedding space* yang disebabkan oleh kemunculan suatu karya.

### 📦 Instalasi & Penggunaan Cepat

```bash
# 1. Clone repositori
git clone https://github.com/stipwunaraha/citation-graph-neural-embedding.git
cd citation-graph-neural-embedding

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instal dependensi
pip install -r requirements.txt
```

**Contoh penggunaan dasar:**
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

### 📚 Dataset Target
- **OpenAlex**: API terbuka dengan data sitasi lengkap.
- **arXiv + Semantic Scholar**: Untuk data teks dan graf.
- **DBLP**: Data publikasi ilmu komputer.

### 🚧 Roadmap
- [ ] Implementasi dasar GCN/GAT dengan PyTorch Geometric.
- [ ] Integrasi SciBERT untuk *node features*.
- [ ] Skrip *scraper* untuk OpenAlex.
- [ ] Evaluasi terhadap metrik "Disruption Index" sebagai baseline.

### 🤝 Kontribusi
Kami sangat terbuka untuk kontribusi! Lihat [CONTRIBUTING.md](CONTRIBUTING.md) dan [Issues](https://github.com/stipwunaraha/citation-graph-neural-embedding/issues) untuk memulai.

### 📄 Lisensi
Proyek ini dilisensikan di bawah Lisensi MIT - lihat berkas [LICENSE](LICENSE) untuk detail.
```
