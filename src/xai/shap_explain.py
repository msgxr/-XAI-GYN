"""
XAI-GYN | src/xai/shap_explain.py
SHAP Gradient Explainer — özellik önem analizi
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def shap_explain(model, pil_image: Image.Image, background_images=None, n_evals: int = 50) -> dict:
    """
    SHAP GradientExplainer ile özellik önem değerleri üretir.

    Args:
        model           : Eğitilmiş XAIGynModel
        pil_image       : PIL görüntüsü
        background_images: Baseline tensor [N, 3, H, W] (None ise rastgele)
        n_evals         : SHAP hesaplama tekrarı (Hız için varsayılan 50 yapıldı)

    Returns:
        dict: shap_values, overlay_image, summary
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap kütüphanesi yüklü değil. pip install shap"}

    try:
        import torch
        from src.preprocess import pil_to_tensor, IMAGE_SIZE

        device = next(model.parameters()).device
        model.eval()

        # Girdi tensörü
        input_tensor = pil_to_tensor(pil_image).to(device)

        # Arka plan tensörü (baseline)
        if background_images is None:
            background = torch.zeros(5, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        else:
            background = background_images.to(device)

        # SHAP Gradient Explainer
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(input_tensor, nsamples=n_evals)

        # Sınıf 1 (malign) için önem haritası
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # [C, H, W]
        else:
            sv = shap_values[0]

        # Kanal ortalaması al → [H, W]
        sv_mean = np.abs(sv).mean(axis=0) if sv.ndim == 3 else sv

        # Normalize
        if sv_mean.max() > sv_mean.min():
            sv_norm = (sv_mean - sv_mean.min()) / (sv_mean.max() - sv_mean.min())
        else:
            sv_norm = sv_mean

        # Overlay
        import cv2
        img_np = np.array(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
        hm_uint8 = (sv_norm * 255).astype(np.uint8)
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_PLASMA)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        overlay = np.clip((1 - 0.4) * img_np + 0.4 * hm_colored, 0, 255).astype(np.uint8)

        return {
            "shap_values"   : shap_values,
            "importance_map": sv_norm,
            "overlay_image" : overlay,
            "summary"       : f"SHAP analizi: En yüksek önem değeri {sv_norm.max():.4f}, "
                              f"ortalama {sv_norm.mean():.4f}",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    print("SHAP modülü yüklendi ✓")
