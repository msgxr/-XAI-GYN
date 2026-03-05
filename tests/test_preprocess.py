"""
XAI-GYN | tests/test_preprocess.py
Ön işleme modülü unit testleri
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import (
    apply_clahe, apply_denoising, preprocess_image,
    get_train_transform, get_val_transform, IMAGE_SIZE
)


def make_dummy_image(h=256, w=256, c=3):
    """Rastgele test görüntüsü oluştur."""
    return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)


class TestCLAHE:
    def test_output_shape(self):
        img = make_dummy_image()
        out = apply_clahe(img)
        assert out.shape == img.shape, "CLAHE çıktı boyutu değişmemeli"

    def test_output_dtype(self):
        img = make_dummy_image()
        out = apply_clahe(img)
        assert out.dtype == np.uint8


class TestDenoising:
    def test_output_shape(self):
        img = make_dummy_image()
        out = apply_denoising(img)
        assert out.shape == img.shape

    def test_output_dtype(self):
        img = make_dummy_image()
        out = apply_denoising(img)
        assert out.dtype == np.uint8


class TestTransforms:
    def test_train_transform_output_shape(self):
        img = make_dummy_image(300, 300)
        transform = get_train_transform()
        result = transform(image=img)
        tensor = result["image"]
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE), (
            f"Eğitim transform çıktısı [3, {IMAGE_SIZE}, {IMAGE_SIZE}] olmalı"
        )

    def test_val_transform_output_shape(self):
        img = make_dummy_image(400, 300)
        transform = get_val_transform()
        result = transform(image=img)
        tensor = result["image"]
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_train_transform_normalization(self):
        """Çıktı tensör -3 ile 3 arasında float olmalı (normalize edilmiş)."""
        img = make_dummy_image()
        transform = get_train_transform()
        result = transform(image=img)
        tensor = result["image"].float()
        assert tensor.min() >= -5.0
        assert tensor.max() <= 5.0


class TestPreprocessImage:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            preprocess_image("nonexistent_image.jpg")

    def test_output_shape_from_file(self, tmp_path):
        import cv2
        img = make_dummy_image(200, 200)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)
        out = preprocess_image(img_path)
        assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert out.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
