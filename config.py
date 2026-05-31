# -*- coding: utf-8 -*-
"""Proje ayarları ve sabit değerler."""

from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent

# Archive veri kümesi yolu (bir üst klasörde)
ARCHIVE_DIR = BASE_DIR.parent / "archive"

# Eğitim sonrası oluşacak klasörler
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model dosya yolları
MODEL_PATH = MODELS_DIR / "plant_disease_model.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.txt"

# Görüntü boyutu (MobileNetV2 giriş boyutu)
IMG_SIZE = 224

# Eğitim parametreleri
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2  # Eğitim verisinin %20'si doğrulama için ayrılır
RANDOM_SEED = 42

# PlantDoc / PlantVillage benzeri 30 sınıf adı
CLASS_NAMES = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Soybean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf black rot",
    "grape leaf",
]

# Sağlıklı sınıflar (hastalık içermeyen yapraklar)
HEALTHY_CLASSES = {
    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Peach leaf",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Soybean leaf",
    "Strawberry leaf",
    "Tomato leaf",
    "grape leaf",
}
