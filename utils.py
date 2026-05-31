# -*- coding: utf-8 -*-
"""Veri hazırlama ve yardımcı fonksiyonlar."""

import shutil
from collections import Counter
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from config import ARCHIVE_DIR, CLASS_NAMES, DATASET_DIR, RANDOM_SEED


def load_class_names_from_yaml():
    """archive/data.yaml dosyasından sınıf adlarını okur."""
    yaml_path = ARCHIVE_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("names", CLASS_NAMES)
    return CLASS_NAMES


def get_image_class_from_label(label_path):
    """YOLO etiket dosyasından baskın sınıf numarasını bulur."""
    if not label_path.exists():
        return None

    class_ids = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))

    if not class_ids:
        return None

    # Bir görselde birden fazla etiket varsa en sık görülen sınıfı al
    return Counter(class_ids).most_common(1)[0][0]


def collect_labeled_images(split_name, class_names):
    """Belirtilen split (train/test) için görsel ve sınıf eşleşmelerini toplar."""
    images_dir = ARCHIVE_DIR / split_name / "images"
    labels_dir = ARCHIVE_DIR / split_name / "labels"

    samples = []
    if not images_dir.exists():
        return samples

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        class_id = get_image_class_from_label(label_path)
        if class_id is None:
            continue

        class_name = class_names[class_id]
        samples.append((image_path, class_name))

    return samples


def copy_samples_to_folder(samples, target_dir):
    """Görselleri sınıf klasörlerine kopyalar."""
    for image_path, class_name in samples:
        # Klasör adında geçersiz karakterleri temizle
        safe_name = class_name.replace("/", "-")
        class_folder = target_dir / safe_name
        class_folder.mkdir(parents=True, exist_ok=True)

        target_path = class_folder / image_path.name
        if not target_path.exists():
            shutil.copy2(image_path, target_path)


def prepare_dataset():
    """
    Archive verisinden dataset/train, dataset/val, dataset/test oluşturur.
    - train: eğitim görüntülerinin %80'i
    - val: eğitim görüntülerinin %20'si
    - test: archive/test klasörü
    """
    class_names = load_class_names_from_yaml()

    # Önceki dataset klasörünü temizle
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    # Archive train ve test görsellerini topla
    train_samples = collect_labeled_images("train", class_names)
    test_samples = collect_labeled_images("test", class_names)

    if not train_samples:
        raise FileNotFoundError(
            "Eğitim verisi bulunamadı. archive/train/images klasörünü kontrol edin."
        )

    # Eğitim verisini train/val olarak ayır (sınıf dağılımını korumaya çalış)
    labels = [s[1] for s in train_samples]
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    if min_count >= 2:
        # Her sınıfta en az 2 örnek varsa stratified split kullan
        train_part, val_part = train_test_split(
            train_samples,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    else:
        # Çok az örneği olan sınıflar varsa normal split kullan
        print("Uyarı: Bazı sınıflarda az örnek var, stratified split kullanılamadı.")
        train_part, val_part = train_test_split(
            train_samples,
            test_size=0.2,
            random_state=RANDOM_SEED,
        )

    # Klasörlere kopyala
    copy_samples_to_folder(train_part, train_dir)
    copy_samples_to_folder(val_part, val_dir)
    copy_samples_to_folder(test_samples, test_dir)

    # Tüm splitlerde ortak olan sınıfları bul (model uyumu için)
    train_classes = {p.parent.name for p in train_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    val_classes = {p.parent.name for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    test_classes = {p.parent.name for p in test_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    common_classes = sorted(train_classes & val_classes & test_classes)

    # Ortak olmayan sınıfları kaldır
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir() and class_dir.name not in common_classes:
                shutil.rmtree(class_dir)

    print(f"Train: {len(train_part)} görsel")
    print(f"Val:   {len(val_part)} görsel")
    print(f"Test:  {len(test_samples)} görsel")
    print(f"Ortak sınıf sayısı: {len(common_classes)}")

    return train_dir, val_dir, test_dir, common_classes


def save_class_names(class_names, save_path):
    """Sınıf adlarını dosyaya kaydeder."""
    with open(save_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")


def load_class_names(load_path):
    """Kaydedilmiş sınıf adlarını okur."""
    with open(load_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_plant_and_disease(class_name):
    """Sınıf adından bitki türü ve hastalık bilgisini ayırır."""
    name = class_name.replace("_", " ")

    # Sağlıklı yaprak kontrolü
    healthy_keywords = ["leaf"]  # sadece "X leaf" formatı sağlıklı sayılır
    disease_keywords = [
        "scab", "rust", "spot", "blight", "mildew", "rot",
        "mosaic", "virus", "mold", "mites", "bacterial", "gray",
    ]

    is_healthy = not any(kw in name.lower() for kw in disease_keywords)

    # Bitki adını bul (ilk kelime veya bileşik isim)
    plant_map = {
        "Apple": "Elma",
        "Bell_pepper": "Dolmalık Biber",
        "Bell pepper": "Dolmalık Biber",
        "Blueberry": "Yaban Mersini",
        "Cherry": "Kiraz",
        "Corn": "Mısır",
        "Peach": "Şeftali",
        "Potato": "Patates",
        "Raspberry": "Ahududu",
        "Soyabean": "Soya",
        "Soybean": "Soya",
        "Squash": "Kabak",
        "Strawberry": "Çilek",
        "Tomato": "Domates",
        "grape": "Üzüm",
        "Grape": "Üzüm",
    }

    plant_tr = "Bilinmeyen Bitki"
    for eng, tr in plant_map.items():
        if name.lower().startswith(eng.lower()):
            plant_tr = tr
            break

    if is_healthy:
        disease_tr = "Sağlıklı"
    else:
        disease_tr = name  # Hastalık adı olarak orijinal sınıf adı

    return plant_tr, disease_tr, is_healthy
