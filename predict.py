# -*- coding: utf-8 -*-
"""Yüklenen görsel için tahmin yapan modül."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from config import CLASS_NAMES, CLASS_NAMES_PATH, HEALTHY_CLASSES, IMG_SIZE, MODEL_PATH
from utils import load_class_names, parse_plant_and_disease


# Hastalık açıklamaları ve öneriler (Türkçe)
DISEASE_INFO = {
    "Apple Scab Leaf": {
        "description": "Elma kabuklu biti (Venturia inaequalis) yapraklarda kahverengi lekeler oluşturur.",
        "recommendation": "Enfekteli yaprakları toplayın, fungisit uygulayın ve ağaçları havalandırın.",
    },
    "Apple rust leaf": {
        "description": "Elma pas hastalığı yaprak altında turuncu-kahverengi kabarcıklar oluşturur.",
        "recommendation": "Hastalıklı yaprakları uzaklaştırın, çevredeki ardıç ağaçlarını kontrol edin.",
    },
    "Bell_pepper leaf spot": {
        "description": "Biber yaprak lekesi, yapraklarda dairesel kahverengi lekeler oluşturur.",
        "recommendation": "Sulama yaparken yaprakları ıslatmaktan kaçının, fungisit kullanın.",
    },
    "Corn Gray leaf spot": {
        "description": "Mısır gri yaprak lekesi, yapraklarda dikdörtgen gri lekeler oluşturur.",
        "recommendation": "Dayanıklı çeşitler kullanın, crop rotation uygulayın.",
    },
    "Corn leaf blight": {
        "description": "Mısır yaprak yanıklığı, uzun kahverengi lekeler ve kuruma yapar.",
        "recommendation": "Enfekteli bitki artıklarını temizleyin, fungisit uygulayın.",
    },
    "Corn rust leaf": {
        "description": "Mısır pas hastalığı, yapraklarda turuncu-kahverengi püstüller oluşturur.",
        "recommendation": "Dayanıklı hibrit tohumlar kullanın, erken müdahale edin.",
    },
    "Potato leaf early blight": {
        "description": "Patates erken yanıklığı, yapraklarda hedef tahtası şeklinde lekeler oluşturur.",
        "recommendation": "Fungisit uygulayın, bitkiler arası mesafeyi artırın.",
    },
    "Potato leaf late blight": {
        "description": "Patates geç yanıklığı, yapraklarda koyu lekeler ve beyaz küf oluşturur.",
        "recommendation": "Acil fungisit uygulaması yapın, enfekteli bitkileri sökün.",
    },
    "Squash Powdery mildew leaf": {
        "description": "Kabak külleme hastalığı, yapraklarda beyaz unlu tabaka oluşturur.",
        "recommendation": "Havalandırmayı artırın, kükürt bazlı fungisit kullanın.",
    },
    "Tomato Early blight leaf": {
        "description": "Domates erken yanıklığı, alt yapraklarda kahverengi halkalı lekeler oluşturur.",
        "recommendation": "Alt yaprakları temizleyin, fungisit uygulayın.",
    },
    "Tomato Septoria leaf spot": {
        "description": "Domates septoria lekesi, küçük dairesel lekeler ve sarı hale oluşturur.",
        "recommendation": "Enfekteli yaprakları koparın, bakır bazlı fungisit kullanın.",
    },
    "Tomato leaf bacterial spot": {
        "description": "Domates bakteriyel leke, yapraklarda küçük siyah lekeler oluşturur.",
        "recommendation": "Tohum dezenfeksiyonu yapın, bakır bazlı ilaç kullanın.",
    },
    "Tomato leaf late blight": {
        "description": "Domates geç yanıklığı, koyu lekeler ve hızlı yaprak kuruması yapar.",
        "recommendation": "Acil müdahale gerektirir, fungisit uygulayın ve bitkileri izole edin.",
    },
    "Tomato leaf mosaic virus": {
        "description": "Domates mozaik virüsü, yapraklarda sarı-yeşil mozaik desen oluşturur.",
        "recommendation": "Enfekteli bitkileri sökün, yaprak bitlerini kontrol edin.",
    },
    "Tomato leaf yellow virus": {
        "description": "Domates sarı yaprak virüsü, yapraklarda sararma ve kıvrılma yapar.",
        "recommendation": "Beyaz sinek popülasyonunu kontrol edin, dayanıklı çeşitler kullanın.",
    },
    "Tomato mold leaf": {
        "description": "Domates küf hastalığı, yapraklarda gri-kahverengi küf oluşturur.",
        "recommendation": "Sera havalandırmasını artırın, nem seviyesini düşürün.",
    },
    "Tomato two spotted spider mites leaf": {
        "description": "İki noktalı kırmızı örümcek, yapraklarda sarı noktalar ve ağ oluşturur.",
        "recommendation": "Akarsisit uygulayın, yaprak altlarını kontrol edin.",
    },
    "grape leaf black rot": {
        "description": "Üzüm siyah çürüklüğü, yapraklarda dairesel kahverengi-siyah lekeler oluşturur.",
        "recommendation": "Enfekteli yaprakları toplayın, fungisit uygulayın.",
    },
    "healthy": {
        "description": "Yaprak sağlıklı görünüyor, belirgin bir hastalık belirtisi tespit edilmedi.",
        "recommendation": "Düzenli sulama ve gübreleme ile bitkinin sağlığını koruyun.",
    },
}


def get_class_names():
    """Eğitilmiş modelin sınıf adlarını yükler."""
    if CLASS_NAMES_PATH.exists():
        return load_class_names(CLASS_NAMES_PATH)
    return CLASS_NAMES


def load_trained_model():
    """Kaydedilmiş .h5 modelini yükler."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}\n"
            "Önce 'python train.py' komutu ile modeli eğitin."
        )
    return load_model(str(MODEL_PATH))


def preprocess_image(image):
    """PIL görüntüsünü model giriş formatına dönüştürür."""
    # RGB'ye çevir ve yeniden boyutlandır
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # NumPy dizisine çevir ve batch boyutu ekle
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 ön işleme
    img_array = preprocess_input(img_array)
    return img_array


def get_disease_info(class_name, is_healthy):
    """Hastalık açıklaması ve öneri metnini döndürür."""
    if is_healthy:
        return DISEASE_INFO["healthy"]["description"], DISEASE_INFO["healthy"]["recommendation"]

    info = DISEASE_INFO.get(class_name)
    if info:
        return info["description"], info["recommendation"]

    return (
        f"{class_name} hastalığı tespit edildi.",
        "Uzman bir ziraat mühendisine danışmanız önerilir.",
    )


def predict_image(image, model=None):
    """
    Görsel analiz eder ve sonuç sözlüğü döndürür.

    Dönüş:
        dict: bitki_adi, durum, hastalik_adi, guven_skoru, aciklama, oneri, sinif_adi
    """
    if model is None:
        model = load_trained_model()

    class_names = get_class_names()
    img_array = preprocess_image(image)

    # Model tahmini
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx]) * 100

    # Sınıf adını al (klasör adlarındaki _ karakterlerini düzelt)
    raw_class = class_names[predicted_idx] if predicted_idx < len(class_names) else "Bilinmeyen"
    class_name = raw_class.replace("_", " ")

    # Bitki ve hastalık bilgisi
    plant_name, disease_name, is_healthy = parse_plant_and_disease(raw_class)
    status = "Sağlıklı" if is_healthy else "Hastalıklı"

    # Açıklama ve öneri
    description, recommendation = get_disease_info(raw_class, is_healthy)

    return {
        "bitki_adi": plant_name,
        "durum": status,
        "hastalik_adi": disease_name if not is_healthy else "Yok",
        "guven_skoru": round(confidence, 2),
        "aciklama": description,
        "oneri": recommendation,
        "sinif_adi": class_name,
        "saglikli": is_healthy,
    }


if __name__ == "__main__":
    # Komut satırından test
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <gorsel_yolu>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = Image.open(img_path)
    result = predict_image(img)

    print("\n--- Tahmin Sonucu ---")
    for key, value in result.items():
        print(f"{key}: {value}")
