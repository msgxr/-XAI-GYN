"""
XAI-GYN | src/dataset.py
PyTorch Dataset sınıfı — kolposkopi görüntü veri seti
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch


# ─────────────────────────────────────
# Sınıf etiketleri
# ─────────────────────────────────────
CLASS_NAMES = ["benign", "malign"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class ColposcopyDataset(Dataset):
    """
    Kolposkopi görüntü veri seti.

    Beklenen dizin yapısı:
        root_dir/
            benign/
                img1.jpg
                img2.jpg
                ...
            malign/
                img1.jpg
                ...

    Args:
        root_dir : Veri kök dizini (benign/ ve malign/ klasörlerini içerir)
        transform: Albumentations transform pipeline
        augment  : Eğitim modu augmentation açık mı?
    """

    def __init__(self, root_dir: str, transform=None, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.samples: List[Tuple[Path, int]] = []

        # Dizindeki görüntüleri tara
        self._scan_directory()
        print(f"[Dataset] {len(self.samples)} görüntü yüklendi → {self.root_dir}")
        self._print_class_distribution()

    def _scan_directory(self):
        """Klas klasörlerini tara ve (path, label) listesi oluştur."""
        supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[Uyarı] {class_dir} bulunamadı, atlanıyor...")
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in supported_ext:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(
                f"Veri seti boş! Dizini kontrol et: {self.root_dir}\n"
                "Beklenen yapı: root_dir/benign/ ve root_dir/malign/"
            )

    def _print_class_distribution(self):
        """Sınıf dağılımını yazdır."""
        from collections import Counter
        dist = Counter(label for _, label in self.samples)
        print("[Dataset] Sınıf dağılımı:")
        for idx, count in dist.items():
            print(f"  {IDX_TO_CLASS[idx]:10s}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Görüntüyü yükle
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Transform uygula
        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return img_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Sınıf dengesizliği için ağırlık hesapla."""
        from collections import Counter
        dist = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = []
        for idx in range(len(CLASS_NAMES)):
            count = dist.get(idx, 1)
            weights.append(total / (len(CLASS_NAMES) * count))
        return torch.tensor(weights, dtype=torch.float)


# ─────────────────────────────────────
# DataLoader oluşturucu
# ─────────────────────────────────────
def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Veri setini train/val/test olarak böler ve DataLoader'lar döndürür.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    from src.preprocess import get_train_transform, get_val_transform

    # Tam veri seti (transform olmadan) → split için
    full_dataset = ColposcopyDataset(data_dir, transform=None)
    total = len(full_dataset)

    n_test = int(total * test_split)
    n_val  = int(total * val_split)
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Her split için kendi transform'unu uygulayan wrapper
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            if self.transform:
                augmented = self.transform(image=img_np)
                img_tensor = augmented["image"]
            else:
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return img_tensor, label

    train_ds = TransformedSubset(train_subset, get_train_transform())
    val_ds   = TransformedSubset(val_subset,   get_val_transform())
    test_ds  = TransformedSubset(test_subset,  get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_weights = full_dataset.get_class_weights()

    print(f"\n[DataLoader] Bölme tamamlandı:")
    print(f"  Train : {len(train_ds)}")
    print(f"  Val   : {len(val_ds)}")
    print(f"  Test  : {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("Dataset modülü yüklendi ✓")
    print(f"  Sınıflar : {CLASS_NAMES}")
    print(f"  Etiketler: {CLASS_TO_IDX}")
