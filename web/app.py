"""
XAI-GYN | web/app.py
Flask REST API — görüntü al, tahmin yap, XAI açıklaması döndür
"""

import sys
import os
import io
import base64
import json
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import cv2

# Proje kök dizini
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────
# Flask uygulaması
# ─────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# Model global değişkeni
_model = None
_device = "cpu"
_model_loaded = False

SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "best_model.pth"


# ─────────────────────────────────────
# Model Yükleme
# ─────────────────────────────────────
def get_model():
    """Singleton model yükleyici — ilk çağrıda yükler."""
    global _model, _device, _model_loaded

    if _model_loaded:
        return _model, _device

    from src.model import get_model as build_model, load_checkpoint

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if CHECKPOINT_PATH.exists():
        print(f"[App] Checkpoint yükleniyor: {CHECKPOINT_PATH}")
        _model = load_checkpoint(str(CHECKPOINT_PATH), device=_device)
    else:
        print("[App] ⚠ Checkpoint bulunamadı — demo modu (rastgele ağırlıklar)")
        _model = build_model(device=_device, pretrained=False)
        _model.eval()

    _model_loaded = True
    return _model, _device


# ─────────────────────────────────────
# Yardımcı: Görüntüyü Base64'e Çevir
# ─────────────────────────────────────
def image_to_base64(np_image: np.ndarray, format: str = "PNG") -> str:
    """NumPy RGB görüntüsünü base64 string'e çevirir."""
    pil = Image.fromarray(np_image.astype(np.uint8))
    buffer = io.BytesIO()
    pil.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def pil_image_to_base64(pil_img: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ─────────────────────────────────────
# Routes
# ─────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Sağlık kontrolü endpoint'i."""
    model_status = "checkpoint" if CHECKPOINT_PATH.exists() else "demo"
    return jsonify({
        "status"      : "ok",
        "model_status": model_status,
        "device"      : _device if _model_loaded else "not_loaded",
        "timestamp"   : time.time(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Ana tahmin endpoint'i.

    Request: multipart/form-data
        - image: görüntü dosyası
        - xai_method: "gradcam" | "lime" | "shap" (opsiyonel, varsayılan: gradcam)

    Response JSON:
        - class_name    : str
        - class_idx     : int
        - confidence    : float
        - probabilities : [benign_prob, malign_prob]
        - explanation   : str
        - original_b64  : base64 orijinal görüntü
        - overlay_b64   : base64 Grad-CAM overlay görüntü
        - heatmap_b64   : base64 ısı haritası
        - processing_ms : int
    """
    start_time = time.time()

    # Dosya kontrolü
    if "image" not in request.files:
        return jsonify({"error": "Görüntü dosyası bulunamadı. 'image' alanı gerekli."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Dosya adı boş."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Desteklenmeyen dosya türü: .{ext}"}), 400

    xai_method = request.form.get("xai_method", "gradcam")

    try:
        # Görüntüyü oku
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Model yükle
        model, device = get_model()

        # Grad-CAM analizi
        from src.xai.gradcam import analyze_image
        result = analyze_image(model, pil_img)

        # Base64 dönüşümleri
        original_b64 = pil_image_to_base64(pil_img.resize((400, 400)))
        overlay_b64  = image_to_base64(result["overlay"])

        # Isı haritasını renklendir
        hm_uint8 = (result["heatmap"] * 255).astype(np.uint8)
        hm_colored = cv2.applyColorMap(
            cv2.resize(hm_uint8, (400, 400)), cv2.COLORMAP_JET
        )
        hm_colored_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        heatmap_b64 = image_to_base64(hm_colored_rgb)

        processing_ms = int((time.time() - start_time) * 1000)

        response = {
            "class_name"    : result["class_name"],
            "class_idx"     : result["class_idx"],
            "confidence"    : round(result["confidence"] * 100, 2),
            "probabilities" : {
                "benign": round(result["probabilities"][0] * 100, 2),
                "malign": round(result["probabilities"][1] * 100, 2),
            },
            "explanation"   : result["explanation"],
            "original_b64"  : original_b64,
            "overlay_b64"   : overlay_b64,
            "heatmap_b64"   : heatmap_b64,
            "processing_ms" : processing_ms,
            "xai_method"    : "Grad-CAM",
            "model_status"  : "checkpoint" if CHECKPOINT_PATH.exists() else "demo",
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"[App] HATA: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Analiz sırasında hata: {str(e)}"}), 500


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  XAI-GYN Web Arayüzü Başlatılıyor")
    print(f"  URL: http://localhost:5000")
    print("="*60 + "\n")

    # Model önceden yükle
    get_model()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
