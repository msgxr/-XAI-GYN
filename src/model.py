"""
XAI-GYN | src/model.py
EfficientNet-B0 tabanlı kolposkopi sınıflandırma modeli
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────
# Model Sınıfı
# ─────────────────────────────────────
class XAIGynModel(nn.Module):
    """
    EfficientNet-B0 tabanlı kolposkopi sınıflandırma modeli.

    Mimarisi:
        EfficientNet-B0 (pretrained) → Global Avg Pool → Dropout → FC(2)

    Args:
        num_classes : Sınıf sayısı (varsayılan 2: benign/malign)
        dropout_rate: Dropout oranı
        pretrained  : ImageNet ağırlıkları kullanılsın mı?
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # EfficientNet-B0 backbone yükle
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        # Özellik çıkarıcı (son FC hariç her şey)
        self.features = base.features
        self.avgpool  = base.avgpool  # [B, 1280, 1, 1]

        # Sınıflandırma başlığı
        in_features = base.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş.
        x: [B, 3, 224, 224]
        döndürür: [B, num_classes] logit
        """
        x = self.features(x)     # [B, 1280, 7, 7]
        x = self.avgpool(x)      # [B, 1280, 1, 1]
        x = torch.flatten(x, 1) # [B, 1280]
        x = self.classifier(x)  # [B, 2]
        return x

    def get_gradcam_target_layers(self):
        """
        Grad-CAM için hedef katmanları döndürür (son konvolüsyon bloğu).
        """
        return [self.features[-1]]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Softmax olasılıkları döndürür.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# ─────────────────────────────────────
# Model Kaydetme / Yükleme
# ─────────────────────────────────────
def save_checkpoint(model: XAIGynModel, path: str, epoch: int, val_acc: float, optimizer_state=None):
    """Model checkpoint kaydeder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch"      : epoch,
        "val_acc"    : val_acc,
        "model_state": model.state_dict(),
        "num_classes": model.num_classes,
    }
    if optimizer_state:
        checkpoint["optimizer_state"] = optimizer_state
    torch.save(checkpoint, path)
    print(f"[Model] Checkpoint kaydedildi → {path} (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(path: str, device: str = "cpu") -> XAIGynModel:
    """Kaydedilmiş checkpoint'ten model yükler."""
    checkpoint = torch.load(path, map_location=device)
    num_classes = checkpoint.get("num_classes", 2)
    model = XAIGynModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[Model] Checkpoint yüklendi ← {path}")
    print(f"  Epoch    : {checkpoint.get('epoch', '?')}")
    print(f"  Val Acc  : {checkpoint.get('val_acc', 0):.4f}")
    return model


def get_model(device: str = "cpu", pretrained: bool = True) -> XAIGynModel:
    """Yeni model oluştur ve belirtilen cihaza taşı."""
    model = XAIGynModel(pretrained=pretrained)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] XAIGynModel oluşturuldu — {n_params:,} eğitilebilir parametre")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    model = get_model(device=device, pretrained=False)

    # Dummy forward pass testi
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"Çıktı şekli: {out.shape}")  # [2, 2]

    probs = model.predict_proba(dummy)
    print(f"Olasılıklar: {probs}")
