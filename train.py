# -*- coding: utf-8 -*-
"""Bitki hastalığı modeli eğitim scripti."""

import os

# TensorFlow log seviyesini azalt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import seaborn as sns

from config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    CLASS_NAMES_PATH,
)
from utils import prepare_dataset, save_class_names


def create_data_generators(train_dir, val_dir, test_dir):
    """Veri artırma (augmentation) ile ImageDataGenerator oluşturur."""

    # Tüm splitlerde aynı sınıf listesini kullan (boyut uyumu için)
    class_names = sorted([
        d.name for d in train_dir.iterdir() if d.is_dir()
    ])

    # Eğitim için veri artırma: döndürme, zoom, çevirme, parlaklık
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,           # 30 dereceye kadar döndür
        zoom_range=0.2,              # %20 yakınlaştır/uzaklaştır
        horizontal_flip=True,        # yatay çevir
        vertical_flip=True,          # dikey çevir
        brightness_range=[0.7, 1.3], # parlaklık değişimi
    )

    # Doğrulama ve test için sadece ön işleme
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Eğitim verisi akışı
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
        shuffle=True,
    )

    # Doğrulama verisi akışı
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
        shuffle=False,
    )

    # Test verisi akışı
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def build_model(num_classes):
    """MobileNetV2 transfer learning ile CNN modeli oluşturur."""

    # ImageNet üzerinde önceden eğitilmiş MobileNetV2 tabanı
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Taban model katmanlarını dondur (transfer learning)
    base_model.trainable = False

    # Sınıflandırma katmanlarını ekle
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # özellik haritasını vektöre çevir
        layers.Dropout(0.3),                # aşırı öğrenmeyi azalt
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),  # sınıf olasılıkları
    ])

    # Modeli derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history, save_path):
    """Accuracy ve Loss grafiklerini çizer ve kaydeder."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy grafiği
    axes[0].plot(history.history["accuracy"], label="Eğitim Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss grafiği
    axes[1].plot(history.history["loss"], label="Eğitim Loss")
    axes[1].plot(history.history["val_loss"], label="Doğrulama Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Grafik kaydedildi: {save_path}")


def evaluate_model(model, test_generator, class_names):
    """Test seti üzerinde performans metriklerini hesaplar."""

    # Test tahminleri
    test_generator.reset()
    y_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes

    # Metrikler
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("PERFORMANS METRİKLERİ")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    # Raporu dosyaya kaydet
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PERFORMANS METRİKLERİ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Rapor kaydedildi: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=[n[:15] for n in class_names],
        yticklabels=[n[:15] for n in class_names],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix kaydedildi: {cm_path}")

    return accuracy, precision, recall, f1


def main():
    """Ana eğitim fonksiyonu."""
    print("=" * 50)
    print("Bitki Hastalığı Tespiti - Model Eğitimi")
    print("=" * 50)

    # Klasörleri oluştur
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Archive verisinden dataset hazırla
    print("\n[1/5] Veri seti hazırlanıyor...")
    train_dir, val_dir, test_dir, class_names = prepare_dataset()

    # Veri yükleyicileri
    print("\n[2/5] Veri yükleyiciler oluşturuluyor...")
    train_gen, val_gen, test_gen = create_data_generators(train_dir, val_dir, test_dir)
    num_classes = train_gen.num_classes

    # Generator sırasına göre sınıf adlarını kaydet (tahmin ile uyumlu olsun)
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    ordered_class_names = [idx_to_class[i] for i in range(num_classes)]
    save_class_names(ordered_class_names, CLASS_NAMES_PATH)

    # Model oluştur
    print("\n[3/5] MobileNetV2 modeli oluşturuluyor...")
    model = build_model(num_classes)
    model.summary()

    # Callback'ler
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Eğitim
    print("\n[4/5] Model eğitiliyor...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Grafikleri kaydet
    graph_path = RESULTS_DIR / "accuracy_loss_graph.png"
    plot_training_history(history, graph_path)

    # Test değerlendirmesi
    print("\n[5/5] Test seti değerlendiriliyor...")
    evaluate_model(model, test_gen, ordered_class_names)

    print("\n" + "=" * 50)
    print(f"Eğitim tamamlandı! Model: {MODEL_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
