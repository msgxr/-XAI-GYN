"""
XAI-GYN | src/preprocess.py
Görüntü ön işleme ve veri augmentation pipeline'ı
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


# ─────────────────────────────────────
# Sabitler
# ─────────────────────────────────────
IMAGE_SIZE = 224        # EfficientNet girdi boyutu
MEAN = [0.485, 0.456, 0.406]   # ImageNet ortalaması
STD  = [0.229, 0.224, 0.225]   # ImageNet std


# ─────────────────────────────────────
# CLAHE Kontrast Artırma
# ─────────────────────────────────────
def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular.
    Kolposkopi görüntülerinde doku detaylarını öne çıkarır.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────
# Gürültü Giderme
# ─────────────────────────────────────
def apply_denoising(image_bgr: np.ndarray) -> np.ndarray:
    """
    Gaussian blur ile hafif gürültü giderir.
    """
    return cv2.GaussianBlur(image_bgr, (3, 3), 0)


# ─────────────────────────────────────
# Ham Görüntüyü İşle
# ─────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Tek bir görüntüyü okur, CLAHE + gürültü giderme uygular, yeniden boyutlandırır.
    Döndürür: RGB numpy array [H, W, 3]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

    # Kontrast artır
    img = apply_clahe(img)

    # Gürültü gider
    img = apply_denoising(img)

    # Yeniden boyutlandır
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ─────────────────────────────────────
# Eğitim için Augmentation Pipeline
# ─────────────────────────────────────
def get_train_transform():
    """
    Eğitim sırasında rastgele augmentation uygular.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(translate_percent=(0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transform():
    """
    Doğrulama/test için sadece normalize et, augmentation yapma.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_inference_transform():
    """
    Tek görüntü inference için transform (web arayüzü servisi).
    """
    return get_val_transform()


# ─────────────────────────────────────
# Yardımcı: PIL Image → Tensor
# ─────────────────────────────────────
def pil_to_tensor(pil_image: Image.Image):
    """
    Web upload'dan gelen PIL görüntüsünü model girişine dönüştürür.
    """
    import torch
    img_np = np.array(pil_image.convert("RGB"))
    transform = get_inference_transform()
    augmented = transform(image=img_np)
    tensor = augmented["image"].unsqueeze(0)  # [1, 3, H, W]
    return tensor


if __name__ == "__main__":
    # Hızlı test
    print("Preprocess modülü yüklendi ✓")
    print(f"  Hedef boyut : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Normalizasyon - Mean: {MEAN}")
    print(f"  Normalizasyon - Std : {STD}")
