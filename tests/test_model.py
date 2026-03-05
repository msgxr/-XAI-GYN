"""
XAI-GYN | tests/test_model.py
Model modülü unit testleri — forward pass, checkpoint, output shape
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import XAIGynModel, get_model, save_checkpoint, load_checkpoint


class TestXAIGynModel:
    def setup_method(self):
        self.device = "cpu"
        self.model = XAIGynModel(num_classes=2, pretrained=False)
        self.model.eval()

    def test_forward_pass_output_shape(self):
        """Model doğru çıktı boyutunu üretmeli."""
        x = torch.randn(1, 3, 224, 224)
        out = self.model(x)
        assert out.shape == (1, 2), f"Beklenen (1,2), alınan: {out.shape}"

    def test_batch_forward_pass(self):
        """Batch boyutu doğru çalışmalı."""
        x = torch.randn(4, 3, 224, 224)
        out = self.model(x)
        assert out.shape == (4, 2)

    def test_predict_proba_sum_to_one(self):
        """Olasılıklar toplamı 1.0 olmalı."""
        x = torch.randn(3, 3, 224, 224)
        probs = self.model.predict_proba(x)
        probs_sum = probs.sum(dim=1)
        for i, s in enumerate(probs_sum):
            assert abs(s.item() - 1.0) < 1e-5, f"Örnek {i}: prob toplamı {s.item():.6f} (1.0 olmalı)"

    def test_predict_proba_range(self):
        """Olasılıklar 0-1 arasında olmalı."""
        x = torch.randn(2, 3, 224, 224)
        probs = self.model.predict_proba(x)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradcam_target_layers(self):
        """Grad-CAM için hedef katmanlar dönmeli."""
        layers = self.model.get_gradcam_target_layers()
        assert len(layers) > 0
        assert layers[0] is not None

    def test_model_has_correct_num_classes(self):
        """Sınıf sayısı doğru olmalı."""
        assert self.model.num_classes == 2


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        """Checkpoint kaydet ve yükle — ağırlıklar eşleşmeli."""
        model = XAIGynModel(num_classes=2, pretrained=False)
        checkpoint_path = str(tmp_path / "test_checkpoint.pth")

        save_checkpoint(model, checkpoint_path, epoch=5, val_acc=0.87)

        loaded_model = load_checkpoint(checkpoint_path, device="cpu")
        loaded_model.eval()

        # Ağırlıkları karşılaştır
        for (n1, p1), (n2, p2) in zip(
            model.state_dict().items(),
            loaded_model.state_dict().items()
        ):
            assert torch.allclose(p1, p2), f"Katman {n1} yüklemeden sonra değişti"

    def test_loaded_model_produces_valid_output(self, tmp_path):
        model = XAIGynModel(pretrained=False)
        ckpt_path = str(tmp_path / "test.pth")
        save_checkpoint(model, ckpt_path, epoch=1, val_acc=0.5)

        loaded = load_checkpoint(ckpt_path)
        x = torch.randn(1, 3, 224, 224)
        out = loaded(x)
        assert out.shape == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
