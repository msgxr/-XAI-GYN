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
    EfficientNet-B0 veya ResNet50 tabanlı kolposkopi sınıflandırma modeli.

    Args:
        num_classes : Sınıf sayısı (varsayılan 2: benign/malign)
        dropout_rate: Dropout oranı
        pretrained  : ImageNet ağırlıkları kullanılsın mı?
        model_type  : "efficientnet" veya "resnet50"
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        model_type: str = "efficientnet"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type.lower()

        if self.model_type == "resnet50":
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
            
            # ResNet özellik çıkarıcı
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            self.avgpool = base.avgpool
            in_features = base.fc.in_features
        else:
            # EfficientNet-B0 (Varsayılan)
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            
            self.features = base.features
            self.avgpool  = base.avgpool
            in_features = base.classifier[1].in_features

        # Sınıflandırma başlığı
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_gradcam_target_layers(self):
        """Grad-CAM için hedef katmanları döndürür."""
        if self.model_type == "resnet50":
            return [self.features[-1][-1]] # layer4'ün son bloğu
        else:
            return [self.features[-1]] # efficientnet son blok

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax olasılıkları döndürür."""
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
        "model_type" : model.model_type,
    }
    if optimizer_state:
        checkpoint["optimizer_state"] = optimizer_state
    torch.save(checkpoint, path)
    print(f"[Model] Checkpoint kaydedildi → {path} (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(path: str, device: str = "cpu", fallback_type: str = "efficientnet") -> XAIGynModel:
    """Kaydedilmiş checkpoint'ten model yükler."""
    checkpoint = torch.load(path, map_location=device)
    num_classes = checkpoint.get("num_classes", 2)
    model_type = checkpoint.get("model_type", fallback_type)
    
    model = XAIGynModel(num_classes=num_classes, pretrained=False, model_type=model_type)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    print(f"[Model] Checkpoint yüklendi ← {path} ({model_type})")
    print(f"  Epoch    : {checkpoint.get('epoch', '?')}")
    print(f"  Val Acc  : {checkpoint.get('val_acc', 0):.4f}")
    return model


def get_model(device: str = "cpu", pretrained: bool = True, model_type: str = "efficientnet") -> XAIGynModel:
    """Yeni model oluştur ve belirtilen cihaza taşı."""
    model = XAIGynModel(pretrained=pretrained, model_type=model_type)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] XAIGynModel ({model_type}) oluşturuldu — {n_params:,} eğitilebilir parametre")
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
