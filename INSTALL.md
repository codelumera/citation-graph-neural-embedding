# 🚀 Panduan Instalasi dan Pengembangan - Semantic Echo

Panduan ini akan membantu Anda menyiapkan lingkungan pengembangan localhost untuk proyek **Semantic Echo**.

## Prasyarat

Pastikan Anda telah menginstal:
- **Python 3.9** atau lebih baru ([Download Python](https://www.python.org/downloads/))
- **pip** (Python package installer)
- **git** (untuk clone repository)

## Langkah-langkah Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/stipwunaraha/citation-graph-neural-embedding.git
cd citation-graph-neural-embedding
```

### 2. Buat Virtual Environment (Disarankan)

Virtual environment memastikan dependensi proyek tidak mengganggu instalasi Python global Anda.

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

Setelah aktivasi berhasil, Anda akan melihat `(venv)` di awal prompt terminal Anda.

### 3. Instal Semua Dependensi

Instal semua library yang diperlukan dengan satu perintah:

```bash
pip install -r requirements.txt
```

**Atau**, instal sebagai package yang dapat dikembangkan (recommended untuk development):

```bash
pip install -e .
```

Dengan opsi `-e` (editable), perubahan kode di folder `semantic_echo/` akan langsung terlihat tanpa perlu reinstall.

### 4. Verifikasi Instalasi

Jalankan test untuk memastikan semua dependensi terinstal dengan benar:

```bash
pytest tests/
```

Atau jalankan contoh dasar:

```bash
python examples/basic_usage.py
```

## Struktur Dependensi

Proyek ini menggunakan library berikut:

### Core ML & Deep Learning
- **PyTorch** (>=2.0.0) - Framework deep learning
- **PyTorch Geometric** (>=2.3.0) - Graph Neural Networks
- **Transformers** (>=4.30.0) - Model transformer dari Hugging Face

### Scientific Computing
- **NumPy** (>=1.21.0) - Komputasi numerik
- **SciPy** (>=1.7.0) - Algoritma ilmiah
- **Pandas** (>=1.3.0) - Manipulasi data

### NLP & Text Processing
- **Sentence Transformers** (>=2.2.0) - Embedding teks

### API & Data Loading
- **Requests** (>=2.28.0) - HTTP requests
- **tqdm** (>=4.65.0) - Progress bars

### Configuration
- **PyYAML** (>=6.0) - Parsing YAML

### Visualization
- **Matplotlib** (>=3.5.0) - Plotting
- **NetworkX** (>=2.8.0) - Analisis jaringan

### Testing & Development
- **pytest** (>=7.0.0) - Testing framework
- **pytest-cov** (>=4.0.0) - Coverage reporting
- **black** (>=23.0.0) - Code formatter
- **flake8** (>=6.0.0) - Code linting

## Instalasi Opsional

### Mode Development

Untuk kontribusi dan development aktif, instal dengan extras:

```bash
pip install -e ".[dev]"
```

### GPU Support (Opsional)

Jika Anda memiliki GPU NVIDIA dan ingin mempercepat training:

```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install GPU version (sesuaikan dengan CUDA version Anda)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Untuk CUDA 11.8, atau gunakan CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### Masalah Umum

**1. Error saat instalasi PyTorch Geometric:**
```bash
# Pastikan PyTorch terinstal dulu
pip install torch torch-geometric
```

**2. Konflik versi:**
```bash
# Upgrade pip terlebih dahulu
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Memory error saat instalasi:**
Beberapa package seperti PyTorch cukup besar. Pastikan Anda memiliki ruang disk yang cukup (minimal 5GB).

### Verifikasi Instalasi

Cek apakah semua package terinstal:

```bash
pip list | grep -E "torch|transformers|numpy|pandas|sentence-transformers"
```

## Langkah Selanjutnya

Setelah instalasi berhasil:

1. Baca [DESIGN.md](DESIGN.md) untuk memahami arsitektur
2. Lihat [examples/basic_usage.py](examples/basic_usage.py) untuk contoh penggunaan
3. Jalankan test suite: `pytest tests/ -v`
4. Mulai berkontribusi! Lihat [CONTRIBUTING.md](CONTRIBUTING.md)

## Update Dependensi

Untuk update semua dependensi ke versi terbaru yang kompatibel:

```bash
pip install --upgrade -r requirements.txt
```

## Uninstall

Jika ingin menghapus instalasi:

```bash
# Deaktivasi virtual environment
deactivate

# Hapus folder virtual environment
rm -rf venv  # Linux/macOS
rmdir /s /q venv  # Windows
```

---

**Butuh bantuan?** Buka [Issues](https://github.com/stipwunaraha/citation-graph-neural-embedding/issues) atau baca dokumentasi lainnya.
