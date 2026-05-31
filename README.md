# Bitki Hastalığı Tespit Sistemi

Derin öğrenme tabanlı bitki yaprak hastalığı tespit ve sınıflandırma projesi.

---
<img width="1280" height="760" alt="Ekran görüntüsü 2026-05-31 100712" src="https://github.com/user-attachments/assets/da96a60d-a9bf-4932-aca1-b1092539ad36" />

<img width="1280" height="720" alt="Ekran görüntüsü 2026-05-31 100834" src="https://github.com/user-attachments/assets/dd94fcb4-691c-4503-81e2-4ca56baaa1b3" />

<img width="1272" height="712" alt="Ekran görüntüsü 2026-05-31 100730" src="https://github.com/user-attachments/assets/1843c1f1-3293-49c1-8fa8-e20962e603c7" />

<img width="1280" height="678" alt="Ekran görüntüsü 2026-05-31 100847" src="https://github.com/user-attachments/assets/d99abd2e-e34c-4a83-83dd-536df44d674e" />

## Projenin Amacı

Bu proje, tarım alanında bitki yapraklarının fotoğraflarını analiz ederek:

1. **Bitki türünü** tahmin etmeyi
2. **Yaprağın sağlıklı mı hastalıklı mı** olduğunu tespit etmeyi
3. Hastalık varsa **hastalık adını ve kısa açıklamasını** göstermeyi
4. Kullanıcıya **anlaşılır bir arayüz** ile sonuç sunmayı

amaçlamaktadır. Çiftçiler ve tarım uzmanları, mobil cihazlarından yükledikleri yaprak fotoğrafları ile hızlı bir ön teşhis alabilir.

---

## Kullanılan Yapay Zeka Yöntemi

- **Derin Öğrenme** (Deep Learning)
- **Evrişimli Sinir Ağı** (CNN - Convolutional Neural Network)
- **Transfer Learning** (Aktarım Öğrenmesi): ImageNet üzerinde önceden eğitilmiş **MobileNetV2** modeli temel alınmıştır
- **Softmax** sınıflandırma katmanı ile 30 farklı sınıf tahmin edilir
- **Veri Artırma** (Data Augmentation): rotation, zoom, flip, brightness

---

## Kullanılan Veri Seti

- **PlantDoc / PlantVillage benzeri veri seti** (Roboflow export)
- 13 bitki türü, **30 sınıf** (sağlıklı + hastalıklı yapraklar)
- Veri kümesi `archive/` klasöründe bulunur
- Yapı: `archive/train`, `archive/valid`, `archive/test`
- Eğitim sırasında veriler `dataset/` klasörüne sınıf bazlı organize edilir

---

## Proje Dosya Yapısı

```
plant_ai_project/
│
├── app.py                  # Streamlit web arayüzü
├── train.py                # Model eğitim scripti
├── predict.py              # Tahmin modülü
├── config.py               # Proje ayarları
├── utils.py                # Yardımcı fonksiyonlar
├── requirements.txt        # Python bağımlılıkları
├── README.md               # Bu dosya
├── models/                 # Eğitilmiş model (.h5)
├── dataset/                # Hazırlanmış veri seti
└── results/                # Eğitim grafikleri ve metrikler
    ├── accuracy_loss_graph.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## Kurulum

### 1. Sanal ortam oluştur (önerilir)

```bash
cd plant_ai_project
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 3. Veri setini kontrol et

`archive/` klasörünün `plant_ai_project` ile aynı dizinde olduğundan emin olun:

```
derin öğrenme ödevi/
├── archive/          ← veri kümesi
└── plant_ai_project/ ← proje
```

---

## Eğitim Süreci

Model eğitimini başlatmak için:

```bash
cd plant_ai_project
python train.py
```

Eğitim sırasında yapılanlar:

1. `archive/` verisinden `dataset/train`, `dataset/val`, `dataset/test` oluşturulur
2. Görseller 224x224 boyutuna yeniden boyutlandırılır
3. Veri artırma uygulanır (rotation, zoom, flip, brightness)
4. MobileNetV2 tabanlı CNN modeli eğitilir (15 epoch)
5. Model `models/plant_disease_model.h5` olarak kaydedilir
6. Grafikler ve metrikler `results/` klasörüne yazılır

---

## Performans Metrikleri

Eğitim sonunda aşağıdaki metrikler hesaplanır:

| Metrik | Açıklama |
|--------|----------|
| **Accuracy** | Doğru tahmin oranı |
| **Precision** | Pozitif tahminlerin doğruluk oranı |
| **Recall** | Gerçek pozitiflerin yakalanma oranı |
| **F1-Score** | Precision ve Recall harmonik ortalaması |
| **Confusion Matrix** | Sınıflar arası karışım matrisi |
| **Classification Report** | Sınıf bazlı detaylı rapor |

Sonuçlar `results/` klasöründe saklanır.

---

## Web Arayüzünü Çalıştırma

Model eğitildikten sonra Streamlit arayüzünü başlatın:

```bash
cd plant_ai_project
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresi açılır.

---

## Sonuç Ekranı Açıklaması

Kullanıcı bir yaprak fotoğrafı yüklediğinde sistem şunları gösterir:

| Alan | Açıklama |
|------|----------|
| **Bitki Türü** | Tahmin edilen bitki (Elma, Domates, Patates vb.) |
| **Durum** | Sağlıklı ✅ veya Hastalıklı ⚠️ |
| **Hastalık Adı** | Tespit edilen hastalık (varsa) |
| **Güven Skoru** | Modelin tahmin güveni (% olarak) |
| **Hastalık Açıklaması** | Hastalık hakkında kısa bilgi |
| **Öneri** | Bitki bakımı veya müdahale önerisi |

---

## Tek Görsel Tahmin (Komut Satırı)

```bash
python predict.py ../archive/test/images/ornek_gorsel.jpg
```

---

## Kaynakça

1. **PlantVillage Dataset**
   - Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*
   - https://github.com/spMohanty/PlantVillage-Dataset

2. **PlantDoc Dataset**
   - Singh, D., et al. (2019). PlantDoc: A Dataset for Visual Plant Disease Detection. *arXiv:1911.10317*
   - https://github.com/pratikkayal/PlantDoc-Dataset

3. **TensorFlow Dokümantasyonu**
   - https://www.tensorflow.org/tutorials

4. **MobileNetV2 Transfer Learning**
   - Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*
   - https://keras.io/api/applications/mobilenet/

5. **EfficientNet Transfer Learning**
   - Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*
   - https://keras.io/api/applications/efficientnet/

6. **ResNet Transfer Learning**
   - He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*
   - https://keras.io/api/applications/resnet/

7. **Bitki Hastalığı Tespiti Akademik Çalışmalar**
   - Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311-318.
   - Too, E. C., et al. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272-279.
   - Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, 147, 70-90.

8. **Streamlit Dokümantasyonu**
   - https://docs.streamlit.io/

---

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Veri seti Creative Commons CC BY 4.0 lisansı altındadır.
