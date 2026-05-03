# 🪙 Bitcoin Close Price Forecasting — LSTM & Seq2Seq (Multi-Step)

> **Multivariate Multi-Horizon Time Series Forecasting** menggunakan arsitektur LSTM Baseline dan Seq2Seq LSTM dengan Teacher Forcing untuk memprediksi harga penutupan Bitcoin 24 jam ke depan.

---

## 📌 Deskripsi Proyek

Proyek ini membangun dan membandingkan dua model **Deep Learning** berbasis LSTM untuk **multi-step time series forecasting** harga Bitcoin:

1. **Baseline LSTM** — Arsitektur LSTM dengan Custom Multi-Head Attention dan Global Average Pooling
2. **Seq2Seq LSTM** — Encoder-Decoder LSTM dengan Teacher Forcing dan Bahdanau Attention

Model dilatih menggunakan data historis Bitcoin per jam dari tahun **2017–2023** (lebih dari 53.000 data points) dengan **8 fitur** (setelah feature engineering), dan berhasil memprediksi harga Close 24 langkah ke depan dengan akurasi MAE (scaled) di bawah threshold **0.015**.

---

## 👩‍💻 Informasi Penulis

| | |
|---|---|
| **Nama** | Nadine Riskia Windi Paramita |
| **Dataset** | Multivariate Crypto Data Hourly (2017–2023) — Bitcoin |
| **Target** | Prediksi harga Close Bitcoin 24 langkah ke depan |

---

## 📂 Struktur Repository

```
📦 bitcoin-forecasting-lstm-seq2seq/
├── 📓 Bitcoin_Close_Price_Forecasting.ipynb   # Notebook utama (EDA, modeling, evaluasi)
├── 🧠 model_baseline_LSTM.keras               # Saved model Baseline LSTM
├── 🧠 model_seq2seq_LSTM.keras                # Saved model Seq2Seq LSTM
├── 🧠 best_model_seq2seq_LSTM.keras           # Best weights Seq2Seq LSTM
├── 🧠 encoder_model.keras                     # Encoder model (untuk Seq2Seq)
├── 📋 requirements.txt                        # Daftar dependensi Python
└── 📖 README.md                               # Dokumentasi proyek
```

---

## 🗂️ Dataset

- **Sumber:** [Multivariate Crypto Data Hourly (Google Drive)](https://drive.google.com/uc?export=download&id=1hpsqSpfjdqIZWqwd259klQSeaNSe5Trr)
- **Periode:** 2017–2023 (data per jam)
- **Fitur input:** `Close`, `Volume USDT`, `RSI`, `MACD_Hist`, `ATR`
- **Fitur tambahan (feature engineering):** `rolling_mean_24`, `rolling_std_24`, `rolling_max_24`
- **Total fitur:** 8 fitur setelah feature engineering
- **Target prediksi:** `Close` (harga penutupan Bitcoin dalam USD)

---

## ⚙️ Konfigurasi Model

| Parameter | Nilai |
|---|---|
| Window Size | 48 jam (2 siklus harian, berdasarkan analisis ACF/PACF) |
| Horizon | 24 langkah ke depan |
| Batch Size | 64 |
| Epochs (max) | 70 |
| Split Data | Train 70% / Val 15% / Test 15% |
| Normalisasi | MinMaxScaler per kolom (fit hanya pada train — no data leakage) |

---

## 🏗️ Arsitektur Model

### 1. Baseline LSTM

```
Input (48, 8)
  └── LSTM(128, return_sequences=True)
  └── CustomDropout(0.2)
  └── LSTM(64, return_sequences=True)
  └── CustomDropout(0.2)
  └── CustomMultiHeadAttention(num_heads=4, key_dim=64)
  └── CustomLayerNorm (Residual Connection)
  └── GlobalAveragePooling1D
  └── CustomDense(128, relu)
  └── CustomDropout(0.2)
  └── CustomDense(24)  →  Output: prediksi 24 horizon
```

### 2. Seq2Seq LSTM

```
Encoder:
  Input (48, 8)
    └── Bidirectional LSTM(128)
    └── CustomDropout(0.2)
    └── LSTM(128, return_state=True)  →  (enc_h, enc_c)

Decoder (dijalankan 24× — satu step per horizon):
  Tiap step:
    └── LSTM Cell (inisialisasi dari encoder state)
    └── CustomMultiHeadAttention (cross-attention ke encoder output)
    └── CustomLayerNorm
    └── CustomDense(64, relu)
    └── CustomDense(1)  →  prediksi 1 step

Mode Training: Teacher Forcing (input = nilai target asli)
Mode Inference: Autoregressive (input = prediksi step sebelumnya)
```

### Custom Layers yang Diimplementasikan

| Custom Layer | Deskripsi |
|---|---|
| `CustomDense` | Implementasi Dense layer dari nol (W, b, aktivasi) |
| `CustomMultiHeadAttention` | Multi-Head Self/Cross Attention dari nol |
| `CustomDropout` | Dropout layer dengan mode training/inference |
| `CustomLayerNorm` | Layer Normalization dengan gamma dan beta trainable |

---

## 🔁 Custom Training Loop

Training dilakukan **sepenuhnya dengan `tf.GradientTape`** tanpa menggunakan `model.fit()` untuk Seq2Seq. Callback kustom juga diimplementasikan dari nol:

- **`CustomEarlyStopping`** — Berhenti jika val_loss tidak membaik selama N epoch; best weights otomatis di-restore
- **`CustomReduceLROnPlateau`** — Mengurangi learning rate sebesar faktor 0.5 jika val_loss stagnan

---

## 📊 Hasil Evaluasi

| Model | MAE (scaled) | MAE (skala asli USD) | Memenuhi Target (< 0.015) |
|---|---|---|---|
| **Baseline LSTM** | **0.006259** | **$409.74** | ✅ |
| Seq2Seq LSTM | 0.014520 | $950.48 | ✅ |

### Ringkasan Training

| | Baseline LSTM | Seq2Seq LSTM |
|---|---|---|
| Epoch hingga berhenti | 30 | 38 |
| Best val_loss | 0.011219 | 0.020502 |
| Epoch terbaik | 22 | 23 |

> **Pemenang: Baseline LSTM** — MAE scaled 0.006259 ($409.74), lebih dari 2× lebih akurat dibanding Seq2Seq LSTM. Hal ini disebabkan arsitektur Baseline yang lebih ringkas (tidak ada exposure bias) dan prediksi seluruh 24 horizon dalam satu forward pass.

---

## 🔍 Tahapan Analisis

### 1. Exploratory Data Analysis (EDA)
- Visualisasi harga Close Bitcoin (2017–2023)
- **Heatmap Korelasi Antar Fitur** — `Close ↔ ATR` (korelasi tinggi), `RSI` (korelasi rendah)
- **Dekomposisi Time Series** (trend, seasonal, residual) — periode 24 jam
- **ACF & PACF** — lag signifikan hingga ~48, mendukung `WINDOW_SIZE = 48`

### 2. Feature Engineering
- Rolling Mean 24 jam
- Rolling Std 24 jam
- Rolling Max 24 jam

### 3. Preprocessing
- Train/Val/Test split (70/15/15%)
- Normalisasi `MinMaxScaler` per kolom (no data leakage)
- Pembuatan `tf.data.Dataset` dengan batching dan prefetching

### 4. Modeling & Training
- Implementasi Custom Layers dari nol
- Custom training loop dengan `tf.GradientTape`
- Custom callbacks: EarlyStopping & ReduceLROnPlateau

### 5. Evaluasi & Visualisasi
- Perbandingan prediksi vs aktual (200 sampel)
- Detail prediksi 24-step per sampel
- Tabel perbandingan Aktual vs Prediksi Baseline vs Seq2Seq

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/NadineParamita-14/bitcoin-forecasting-lstm-seq2seq.git
cd bitcoin-forecasting-lstm-seq2seq
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Notebook

Buka dan jalankan notebook secara berurutan di **Google Colab** atau **Jupyter Notebook**:

```
Bitcoin_Close_Price_Forecasting.ipynb
```

> Dataset diunduh otomatis dari Google Drive melalui URL yang sudah tersedia di notebook.

### 4. Load Model Tersimpan (Opsional)

```python
import tensorflow as tf
from tensorflow import keras

# Load Baseline LSTM
model_baseline = keras.models.load_model(
    'model_baseline_LSTM.keras',
    custom_objects={
        'CustomDense': CustomDense,
        'CustomMultiHeadAttention': CustomMultiHeadAttention,
        'CustomDropout': CustomDropout,
        'CustomLayerNorm': CustomLayerNorm,
    }
)

# Load Seq2Seq LSTM
model_seq2seq = keras.models.load_model(
    'best_model_seq2seq_LSTM.keras',
    custom_objects={ ... }  # definisikan custom objects yang relevan
)
```

---

## 🛠️ Requirements

```
tensorflow==2.19.0
numpy>=1.26.0,<2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
tqdm>=4.65.0
```

Install sekaligus:

```bash
pip install -r requirements.txt
```

---

## 📝 Kesimpulan

Proyek ini berhasil membuktikan bahwa:

1. **Baseline LSTM** dengan Custom Multi-Head Attention mampu memprediksi harga Close Bitcoin 24 jam ke depan dengan MAE scaled **0.006259** — jauh di bawah threshold 0.015.
2. **Seq2Seq LSTM** dengan Teacher Forcing juga memenuhi target (MAE scaled 0.014520), namun lebih rendah akurasinya akibat *exposure bias* pada mode inference autoregressive.
3. **Feature engineering** (rolling statistics 24 jam) dan **window size 48 jam** (berdasarkan ACF/PACF) berkontribusi signifikan pada performa model.
4. **Custom training loop** dengan `tf.GradientTape` memberikan kontrol penuh atas proses training, termasuk implementasi Teacher Forcing pada Seq2Seq.

---

## 📜 Lisensi

Proyek ini dibuat untuk keperluan **submission tugas akhir** kelas *Membangun Proyek Deep Learning Tingkat Mahir*. Bebas digunakan untuk keperluan edukasi dan pembelajaran.
