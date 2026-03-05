"""
XAI-GYN | src/xai/lime_explain.py
LIME (Local Interpretable Model-agnostic Explanations) lokal açıklama
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def lime_explain(model, pil_image: Image.Image, class_names: list = None, num_samples: int = 1000) -> dict:
    """
    LIME ile lokal piksel segment açıklaması üretir.

    Args:
        model      : Eğitilmiş XAIGynModel
        pil_image  : PIL görüntüsü
        class_names: ["Benign", "Malign"]
        num_samples: LIME örnekleme sayısı

    Returns:
        dict: explanation, mask, image_with_mask
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        return {"error": "lime kütüphanesi yüklü değil. pip install lime"}

    import torch
    from src.preprocess import get_inference_transform, IMAGE_SIZE

    if class_names is None:
        class_names = ["Benign", "Malign"]

    device = next(model.parameters()).device
    transform = get_inference_transform()

    # LIME için tahmin fonksiyonu (batch)
    def predict_fn(images_np):
        """
        LIME'ın çağırdığı tahmin fonksiyonu.
        images_np: [N, H, W, 3] uint8
        Döndürür: [N, num_classes] float
        """
        batch_tensors = []
        for img in images_np:
            pil = Image.fromarray(img.astype(np.uint8))
            img_arr = np.array(pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
            aug = transform(image=img_arr)
            batch_tensors.append(aug["image"])

        batch = torch.stack(batch_tensors).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # LIME açıklayıcı
    explainer = lime_image.LimeImageExplainer()
    img_np = np.array(pil_image.convert("RGB"))

    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42,
    )

    # En tahmin edilen sınıf için maske
    pred_class = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )

    # Görsel oluştur
    from skimage.segmentation import mark_boundaries
    image_with_mask = mark_boundaries(temp / 255.0, mask)
    image_with_mask = (image_with_mask * 255).astype(np.uint8)

    return {
        "explanation"     : explanation,
        "mask"            : mask,
        "image_with_mask" : image_with_mask,
        "pred_class_idx"  : pred_class,
        "pred_class_name" : class_names[pred_class],
    }


if __name__ == "__main__":
    print("LIME modülü yüklendi ✓")
