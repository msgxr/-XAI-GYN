import os
import urllib.request
from pathlib import Path

# Demo amaciyla Hugging Face Hub veya direk URL verilebilir.
# Burada örnek olarak gecerli bir acik agirlik dosyasi URL'i kullaniyoruz. 
# Gercek projelerde kendi model ağırlıklarınızın linkiyle degistirin.
WEIGHTS_URL = "https://github.com/msgxr/-XAI-GYN/releases/download/v1.0/best_model.pth"

# Alternatif olarak eger U-Net agirliklariniz varsa:
UNET_URL = "https://github.com/msgxr/-XAI-GYN/releases/download/v1.0/unet_best.pth"

MODELS_DIR = Path(__file__).parent / "models" / "checkpoints"

def download_file(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"[{desc}] Checkpoint zaten mevcut: {dest}")
        return

    print(f"[{desc}] Indiriliyor: {url} -> {dest}")
    try:
        # Eğer link 404 ise urllib hata fırlatacak, exception ile yakalanacak.
        urllib.request.urlretrieve(url, dest)
        print(f"[{desc}] Indirme basarili!")
    except Exception as e:
        print(f"[{desc}] Indirme basarisiz (Demo URL calismayabilir): {e}")
        print(f"Lutfen ilgili linki dogrulayin ya da egitim scriptiyle kendi modelinizi ureti.")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Siniflandirma Ağırlığı (EfficientNet/ResNet vb.)
    cls_dest = MODELS_DIR / "best_model.pth"
    download_file(WEIGHTS_URL, cls_dest, "Klinik Model")

    # 2. Segmentasyon Agirligi (U-Net vb.) - Opsiyonel
    unet_dest = MODELS_DIR / "unet_best.pth"
    download_file(UNET_URL, unet_dest, "U-Net Segmentasyon")

if __name__ == "__main__":
    print("=" * 60)
    print(" XAI-GYN Model Agirlik Indirici")
    print("=" * 60)
    main()
