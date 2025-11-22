
# Proyek Klasifikasi Penyakit Tanaman menggunakan CNN

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.10%2B-red?style=for-the-badge)
![Classification](https://img.shields.io/badge/Task-Image%20Classification-green?style=for-the-badge)

Ini adalah proyek implementasi model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan 15 jenis penyakit pada tanaman dari dataset `PlantVillage`. Model ini menggunakan teknik *data augmentation* dan *early stopping* untuk mencapai akurasi tinggi dan mencegah *overfitting*.

## Ringkasan Model

Model ini adalah arsitektur CNN sekuensial yang dirancang untuk pengenalan fitur visual pada citra daun tanaman.

| Komponen | Tujuan |
| :--- | :--- |
| **Dataset** | `PlantVillage` (20.638 citra), 15 kelas penyakit/sehat. |
| **Input** | Citra warna 180x180 piksel (RGB). |
| **Preprocessing** | Normalisasi piksel (`1./255`), *Caching*, dan *Prefetching*. |
| **Augmentation** | `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.2)`. |
| **Arsitektur** | 3 Blok Konvolusi (16, 32, 64 filter), `MaxPooling`, `Dropout(0.2)`. |
| **Optimizer** | Adam. |
| **Loss** | `SparseCategoricalCrossentropy(from_logits=True)`. |
| **Callback** | `EarlyStopping` (monitor `val_loss`, `patience=3`). |

---

## Metodologi

### 1. Penyiapan Data

Data diunduh dari Kaggle dan dibagi menjadi tiga set untuk memastikan evaluasi yang tidak bias:
* **Training Set:** 80% data untuk melatih model.
* **Validation Set:** 10% data untuk memonitor kinerja selama pelatihan.
* **Test Set:** 10% data untuk evaluasi akhir model yang tidak terlihat.

```python
# Pembagian Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123, ...
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123, ...
)

# Pemisahan Val/Test
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)
````

### 2\. Augmentasi Data (Untuk Mencegah Overfitting)

Lapisan augmentasi diterapkan langsung pada model untuk melatihnya agar lebih tangguh terhadap variasi citra.

```python
data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
  ]
)
```

### 3\. Arsitektur CNN

Model ini menggunakan tiga blok konvolusi yang diikuti dengan `MaxPooling` dan lapisan *dropout* untuk ekstraksi fitur yang efisien.

```python
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

  # Blok 1
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Blok 2
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Blok 3
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # Regularization

  # Klasifikasi
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(class_names))
])
```

-----

## Kinerja dan Hasil

Model dilatih selama 20 *epoch* dengan *callback* **Early Stopping** untuk menemukan titik pelatihan optimal.

### 1\. Metrik Pelatihan

Model menunjukkan peningkatan kinerja yang stabil pada data pelatihan dan validasi:

| Metrik | Nilai Awal (Epoch 1) | Nilai Akhir (Epoch 20) |
| :--- | :--- | :--- |
| **Training Accuracy** | 37.56% | 94.82% |
| **Validation Accuracy** | 57.34% | 89.71% |
| **Training Loss** | 1.9107 | 0.1567 |
| **Validation Loss** | 1.7094 | 0.3953 |

### 2\. Visualisasi Pelatihan

Grafik di bawah menunjukkan bahwa model belajar secara konsisten dan tidak menunjukkan *overfitting* yang parah, karena akurasi validasi mengikuti tren akurasi pelatihan.

### 3\. Evaluasi Akhir pada Test Set

Setelah pelatihan, model dievaluasi pada **Test Set** yang benar-benar baru, menghasilkan metrik kinerja akhir:

| Metrik | Nilai |
| :--- | :--- |
| **Test Loss** | *Tergantung pada hasil akhir `model.evaluate(test_ds)`* |
| **Test Accuracy** | *Tergantung pada hasil akhir `model.evaluate(test_ds)`* |

#### Laporan Klasifikasi Rinci (Contoh Output)

```text
# Laporan ini memberikan metrik per kelas (precision, recall, f1-score)
# (Output dari classification_report)

# ... Contoh Baris dari Output (Akan diisi dengan hasil aktual) ...
#                       precision    recall  f1-score   support
#     Apple___Black_rot       0.93      0.82      0.87       116
# Apple___healthy           0.97      0.98      0.97       115
# ...
#                accuracy                           0.87      2048
#               macro avg       0.85      0.86      0.85      2048
#            weighted avg       0.87      0.87      0.87      2048
```

-----

## Cara Menggunakan Model

Anda dapat memuat citra baru dan menjalankan inferensi untuk mendapatkan prediksi kelas dan tingkat kepercayaan.

### 1\. Inferensi Satu Citra

Langkah ini menunjukkan cara memuat citra, memprosesnya, dan mendapatkan prediksi kelas:

```python
# 1. Tentukan path citra
image_path = "Image/PlantVillage/Pepper__bell___Bacterial_spot/..."

# 2. Muat dan Persiapkan Citra
image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
img_arr = tf.keras.utils.img_to_array(image)
img_bat = tf.expand_dims(img_arr, 0) # Konversi ke format batch (1, 180, 180, 3)

# 3. Prediksi dan Konversi ke Probabilitas
predict = model.predict(img_bat)
score = tf.nn.softmax(predict)

# 4. Cetak Hasil
predicted_class = class_names[np.argmax(score)].replace('_', ' ')
confidence = 100 * np.max(score)

print("Jenis daun adalah {} dengan akurasi {:0.2f}".format(predicted_class, confidence))
```

```
