import os
import torch
from pathlib import Path
from src.models.unet import UNet

def create_dummy():
    models_dir = Path(__file__).parent / "models" / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "unet_best.pth"
    
    if out_path.exists():
        print(f"U-Net ağırlıkları zaten mevcut: {out_path}")
        return
        
    print("Dummy U-Net ağırlıkları oluşturuluyor...")
    model = UNet(n_channels=3, n_classes=1)
    torch.save(model.state_dict(), out_path)
    print(f"Oluşturuldu: {out_path}")

if __name__ == "__main__":
    create_dummy()
