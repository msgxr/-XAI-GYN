"""
XAI-GYN | demo_inference.py
Komut satırından hızlı görüntü analizi
Kullanım: python demo_inference.py --image data/sample/test.jpg
"""

import argparse
import sys
import os
from pathlib import Path

# Proje kök dizini
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="XAI-GYN Demo — Kolposkopi Görüntü Analizi",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Analiz edilecek görüntü dosyası yolu"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="models/checkpoints/best_model.pth",
        help="Model checkpoint yolu (opsiyonel)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Grad-CAM karşılaştırma görseli kayıt yolu (opsiyonel)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  XAI-GYN Görüntü Analizi")
    print("="*60)

    # Model yükle
    import torch
    from src.model import get_model, load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Checkpoint yükleniyor: {checkpoint_path}")
        model = load_checkpoint(str(checkpoint_path), device=device)
    else:
        print("⚠ Checkpoint bulunamadı — demo modu (rastgele ağırlıklar)")
        model = get_model(device=device, pretrained=False)
        model.eval()

    # Görüntüyü analiz et
    from PIL import Image
    from src.xai.gradcam import analyze_image

    print(f"\nGörüntü analiz ediliyor: {args.image}")
    result = analyze_image(
        model=model,
        image_input=args.image,
        class_names=["Benign", "Malign"],
        save_path=args.save,
    )

    # Sonuçları yazdır
    print("\n" + "─"*60)
    print(f"  TAHMİN     : {result['class_name']} (Sınıf {result['class_idx']})")
    print(f"  GÜVEN      : {result['confidence']*100:.2f}%")
    print(f"  Benign/Malign: {result['probabilities'][0]*100:.2f}% / {result['probabilities'][1]*100:.2f}%")
    print("─"*60)
    print("\nAÇIKLAMA:")
    print(result['explanation'])
    print("─"*60)

    if args.save:
        print(f"\nGrad-CAM görseli kaydedildi: {args.save}")

    print("\n✓ Analiz tamamlandı!")


if __name__ == "__main__":
    main()
