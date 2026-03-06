"""
XAI-GYN | src/preprocess.py
Görüntü ön işleme ve veri augmentation pipeline'ı
Desteklenen modaliteler: kolposkopi | ultrason | laparoskopi
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
    Gaussian blur ile hafif gürültü giderir (kolposkopi).
    """
    return cv2.GaussianBlur(image_bgr, (3, 3), 0)


def apply_ultrasound_denoising(image_bgr: np.ndarray) -> np.ndarray:
    """
    Ultrason için bilateral filter ile speckle gürültüsü azaltma.
    Kenar bilgisini korurken speckle'ı baskırır.
    """
    return cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)


def apply_laparoscopy_preprocessing(image_bgr: np.ndarray) -> np.ndarray:
    """
    Laparoskopi için parlaklık normalizasyonu + hafif aydınlatma düzeltmesi.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.merge((l_norm, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────
# Ham Görüntüyü İşle
# ─────────────────────────────────────
def preprocess_image(image_path: str, modality: str = "kolposkopi") -> np.ndarray:
    """
    Tek bir görüntüyü okur, modaliteye özgü ön işleme uygular.

    Args:
        image_path: Görüntü dosyası yolu
        modality  : 'kolposkopi' | 'ultrason' | 'laparoskopi'

    Döndürür: RGB numpy array [H, W, 3]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

    if modality == "ultrason":
        img = apply_ultrasound_denoising(img)
        img = apply_clahe(img, clip_limit=1.5)  # Daha hafif CLAHE
    elif modality == "laparoskopi":
        img = apply_laparoscopy_preprocessing(img)
        img = apply_clahe(img)
    else:  # kolposkopi (varsayılan)
        img = apply_clahe(img)
        img = apply_denoising(img)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_preprocess_fn(modality: str = "kolposkopi"):
    """
    Modaliteye göre uygun ön işleme fonksiyonunu döndürür.
    Kullanım: fn = get_preprocess_fn('ultrason'); img = fn(image_bgr)
    """
    if modality == "ultrason":
        def _fn(img):
            img = apply_ultrasound_denoising(img)
            return apply_clahe(img, clip_limit=1.5)
        return _fn
    elif modality == "laparoskopi":
        def _fn(img):
            img = apply_laparoscopy_preprocessing(img)
            return apply_clahe(img)
        return _fn
    else:
        def _fn(img):
            img = apply_clahe(img)
            return apply_denoising(img)
        return _fn


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
    print("Preprocess modülü yüklendi ✓")
    print(f"  Hedef boyut    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Normalizasyon - Mean: {MEAN}")
    print(f"  Normalizasyon - Std : {STD}")
    print("  Desteklenen modaliteler: kolposkopi | ultrason | laparoskopi")
