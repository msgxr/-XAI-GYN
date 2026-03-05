"""
XAI-GYN | tests/test_gradcam.py
Grad-CAM modülü unit testleri
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import XAIGynModel
from src.xai.gradcam import GradCAM, analyze_image


def make_dummy_pil(size=(224, 224)):
    """Rastgele PIL görüntüsü oluştur."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestGradCAM:
    def setup_method(self):
        self.model = XAIGynModel(num_classes=2, pretrained=False)
        self.model.eval()
        target_layer = self.model.get_gradcam_target_layers()[0]
        self.gradcam = GradCAM(self.model, target_layer)

    def test_heatmap_shape(self):
        """Isı haritası 2D array olmalı."""
        from src.preprocess import pil_to_tensor
        pil_img = make_dummy_pil()
        tensor = pil_to_tensor(pil_img)
        tensor.requires_grad_(True)

        heatmap = self.gradcam.generate(tensor, class_idx=0)
        assert heatmap.ndim == 2, "Isı haritası 2D olmalı"

    def test_heatmap_value_range(self):
        """Normalize edilmiş ısı haritası 0-1 arasında olmalı."""
        from src.preprocess import pil_to_tensor
        pil_img = make_dummy_pil()
        tensor = pil_to_tensor(pil_img)
        tensor.requires_grad_(True)

        heatmap = self.gradcam.generate(tensor, class_idx=1)
        assert heatmap.min() >= -0.01
        assert heatmap.max() <= 1.01

    def test_resize_heatmap(self):
        """Yeniden boyutlandırma doğru çalışmalı."""
        heatmap = np.random.rand(7, 7).astype(np.float32)
        resized = GradCAM.resize_heatmap(heatmap, target_size=(224, 224))
        assert resized.shape == (224, 224)

    def test_overlay_output_dtype(self):
        """Overlay çıktısı uint8 RGB array olmalı."""
        original = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        heatmap  = np.random.rand(7, 7).astype(np.float32)
        overlay = GradCAM.overlay_on_image(original, heatmap)
        assert overlay.dtype == np.uint8
        assert overlay.shape == (224, 224, 3)


class TestAnalyzeImage:
    def test_analyze_returns_required_keys(self):
        """analyze_image gerekli anahtarları döndürmeli."""
        model = XAIGynModel(pretrained=False)
        model.eval()
        pil_img = make_dummy_pil()

        result = analyze_image(model, pil_img)
        required = ["class_idx", "class_name", "confidence", "probabilities",
                    "heatmap", "overlay", "original", "explanation"]
        for key in required:
            assert key in result, f"'{key}' anahtarı eksik"

    def test_class_idx_valid(self):
        """Tahmin sınıfı 0 veya 1 olmalı."""
        model = XAIGynModel(pretrained=False)
        model.eval()
        pil_img = make_dummy_pil()

        result = analyze_image(model, pil_img)
        assert result["class_idx"] in [0, 1]

    def test_confidence_in_range(self):
        """Güven değeri 0-1 arasında olmalı."""
        model = XAIGynModel(pretrained=False)
        model.eval()
        pil_img = make_dummy_pil()

        result = analyze_image(model, pil_img)
        assert 0.0 <= result["confidence"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
