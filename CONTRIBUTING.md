# Panduan Kontribusi untuk Semantic Echo

Terima kasih atas minat Anda untuk berkontribusi pada **Semantic Echo**! Proyek ini bertujuan untuk mengukur pengaruh ilmiah menggunakan Graph Neural Networks dan representasi semantik.

## 🚀 Cara Berkontribusi

### 1. Fork dan Clone

```bash
# Fork repositori ini di GitHub, lalu:
git clone https://github.com/USERNAME_ANDA/citation-graph-neural-embedding.git
cd citation-graph-neural-embedding
git remote add upstream https://github.com/stipwunaraha/citation-graph-neural-embedding.git
```

### 2. Setup Development Environment

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (termasuk dev dependencies)
pip install -r requirements.txt

# Install dalam mode development
pip install -e .
```

### 3. Buat Branch Baru

```bash
git checkout -b feature/nama-fitur-anda
# atau
git checkout -b fix/nama-bug-fix
```

## 📝 Standar Kode

### Style Guide
- Gunakan **PEP 8** untuk style guide Python
- Gunakan **Black** untuk auto-formatting:
  ```bash
  black semantic_echo/ tests/ examples/
  ```
- Gunakan **flake8** untuk linting:
  ```bash
  flake8 semantic_echo/ tests/
  ```

### Dokumentasi
- Setiap fungsi publik harus memiliki docstring dengan format Google Style
- Sertakan contoh penggunaan dalam docstring
- Update README.md jika menambahkan fitur baru

### Testing
- Tulis unit test untuk setiap fitur baru
- Pastikan semua test lulus sebelum submit:
  ```bash
  pytest tests/ -v --cov=semantic_echo
  ```
- Coverage minimal: 70%

## 🔧 Area yang Membutuhkan Kontribusi

### Prioritas Tinggi
1. **Integrasi Data**: 
   - Parser XML arXiv yang lengkap
   - Support untuk dataset DBLP
   - Caching mechanism untuk API calls

2. **Model Enhancement**:
   - Implementasi SI-HDGNN (Heterogeneous Dynamical GNN)
   - Temporal graph networks untuk evolusi sitasi
   - Multi-task learning untuk prediksi berbagai metrik

3. **Evaluasi**:
   - Benchmark terhadap Disruption Index
   - Comparison dengan metrik tradisional (H-index, Impact Factor)
   - Visualisasi hasil embedding (t-SNE, UMAP)

### Prioritas Sedang
4. **Skalabilitas**:
   - Mini-batch training untuk graf besar
   - Distributed training support
   - Graph sampling strategies

5. **API & CLI**:
   - Command-line interface untuk training
   - REST API untuk inference
   - Interactive notebook tutorials

6. **Dokumentasi**:
   - Tutorial Jupyter Notebook
   - API documentation (Sphinx)
   - Blog posts dan case studies

## 📬 Submit Pull Request

1. **Commit Message**: Gunakan format yang jelas
   ```
   feat: tambahkan support untuk DBLP dataset
   
   - Implementasi DBLPLoader class
   - Unit tests untuk parser XML
   - Update dokumentasi
   ```

2. **PR Description**: Jelaskan perubahan Anda
   - Apa yang diubah?
   - Mengapa perubahan ini diperlukan?
   - Bagaimana cara menguji?

3. **Review Process**:
   - Minimal 1 approval dari maintainer
   - Semua CI checks harus lulus
   - Resolve semua komentar review

## 🐛 Melaporkan Bug

Gunakan [GitHub Issues](https://github.com/stipwunaraha/citation-graph-neural-embedding/issues) dengan template:

```markdown
**Deskripsi Bug**
Jelaskan bug secara singkat dan jelas.

**Langkah Reproduksi**
1. Import '...'
2. Jalankan fungsi '...'
3. Lihat error

**Expected Behavior**
Apa yang seharusnya terjadi?

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.9]
- PyTorch: [e.g. 2.0]
- PyG: [e.g. 2.3]

**Screenshots**
Jika ada, tambahkan screenshot untuk membantu penjelasan.
```

## 💡 Ide Fitur Baru

Kami sangat terbuka untuk ide baru! Silakan buat issue dengan label `enhancement` dan jelaskan:
- Masalah apa yang ingin diselesaikan?
- Bagaimana solusi yang diusulkan?
- Apakah ada referensi paper/implementasi serupa?

## 👥 Tim Maintainers

- [@stipwunaraha](https://github.com/stipwunaraha) - Lead Maintainer

## 📜 Lisensi

Dengan berkontribusi, Anda menyetujui bahwa kontribusi Anda dilisensikan di bawah [MIT License](LICENSE).

---

Terima kasih telah membantu membuat analisis dampak ilmiah lebih bermakna! 🎉
