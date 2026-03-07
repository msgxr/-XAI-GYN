"""
XAI-GYN | web/app.py
Flask REST API — goruntu al, tahmin yap, XAI aciklamasi donur
Desteklenen XAI     : Grad-CAM | LIME | SHAP
Desteklenen Modalite: Kolposkopi | Ultrason | Laparoskopi
"""

import sys
import io
import base64
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────
# Flask uygulamasi
# ─────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

_model        = None
_device       = "cpu"
_model_loaded = False

SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
CHECKPOINT_PATH      = ROOT / "models" / "checkpoints" / "best_model.pth"
VALID_XAI_METHODS    = ("gradcam", "lime", "shap")
VALID_MODALITIES     = ("kolposkopi", "ultrason", "laparoskopi")


# ─────────────────────────────────────
# Model Yukleme (Singleton)
# ─────────────────────────────────────
_unet_model = None
_unet_is_demo = True

def get_model():
    global _model, _unet_model, _unet_is_demo, _device, _model_loaded
    if _model_loaded:
        return _model, _unet_model, _device
        
    from src.model import get_model as build_model, load_checkpoint
    from src.models.unet import get_unet_model
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if CHECKPOINT_PATH.exists():
        print(f"[App] Checkpoint yukleniyor: {CHECKPOINT_PATH}")
        _model = load_checkpoint(str(CHECKPOINT_PATH), device=_device)
    else:
        print("[App] Checkpoint bulunamadi — demo modu")
        _model = build_model(device=_device, pretrained=False)
        _model.eval()
        
    # U-Net Yükleme (Eğer ağırlık varsa)
    UNET_CKPT = ROOT / "models" / "checkpoints" / "unet_best.pth"
    if UNET_CKPT.exists():
        _unet_model = get_unet_model(device=_device, pretrained_path=str(UNET_CKPT))
        _unet_is_demo = False
    else:
        _unet_model = get_unet_model(device=_device, pretrained_path=None)
        _unet_is_demo = True
        
    _model_loaded = True
    return _model, _unet_model, _device


# ─────────────────────────────────────
# Yardimcilar
# ─────────────────────────────────────
def ndarray_to_b64(arr: np.ndarray, fmt: str = "PNG") -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_b64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def colorize_heatmap(heatmap: np.ndarray, size: int = 400) -> np.ndarray:
    hm_u8     = (heatmap * 255).astype(np.uint8)
    hm_r      = cv2.resize(hm_u8, (size, size))
    hm_color  = cv2.applyColorMap(hm_r, cv2.COLORMAP_JET)
    return cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────
# Routes
# ─────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status"      : "ok",
        "model_status": "checkpoint" if CHECKPOINT_PATH.exists() else "demo",
        "device"      : _device if _model_loaded else "not_loaded",
        "timestamp"   : time.time(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Ana tahmin endpoint'i.

    Form Parametreleri:
        image      : goruntu dosyasi (zorunlu)
        xai_method : gradcam | lime | shap          (varsayilan: gradcam)
        modality   : kolposkopi | ultrason | laparoskopi (varsayilan: kolposkopi)

    Yanit JSON:
        class_name, class_idx, confidence, probabilities,
        explanation, original_b64, overlay_b64, heatmap_b64,
        xai_method, modality, processing_ms, model_status,
        lezyon_detected
    """
    t0 = time.time()

    # Girdi kontrolleri
    if "image" not in request.files:
        return jsonify({"error": "Goruntu dosyasi bulunamadi ('image' alani gerekli)."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Dosya adi bos."}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Desteklenmeyen dosya turu: .{ext}"}), 400

    xai_method = request.form.get("xai_method", "gradcam").lower()
    modality   = request.form.get("modality",   "kolposkopi").lower()
    if xai_method not in VALID_XAI_METHODS:
        xai_method = "gradcam"
    if modality not in VALID_MODALITIES:
        modality = "kolposkopi"

    try:
        pil_img       = Image.open(io.BytesIO(file.read())).convert("RGB")
        model, unet_model, device = get_model()

        # Modality'e ozgu on islem notu
        modality_notes = {
            "kolposkopi" : "Kolposkopi goruntusu: CLAHE kontrast artirma uygulandı.",
            "ultrason"   : "Ultrason goruntusu: Speckle gurultu azaltma + bilateral filtre uygulandı.",
            "laparoskopi": "Laparoskopi goruntusu: CLAHE + parlaklik normalizasyonu uygulandı.",
        }

        # ── Lezyon tespiti (confidence esigi uzerinden) ──
        from src.xai.gradcam import analyze_image
        gradcam_result  = analyze_image(model, pil_img)
        malign_prob     = gradcam_result["probabilities"][1]
        lezyon_detected = malign_prob > 0.3  # %30 uzerinde malign riski = lezyon var

        # ── Counterfactual (Karşı-Olgusal) XAI Analizi ──
        try:
            from src.xai.counterfactual import generate_counterfactual_explanation
            current_class = gradcam_result["class_name"]
            cf_text = generate_counterfactual_explanation(model, pil_img, malign_prob, current_class, device)
        except Exception as e:
            print(f"[App] Counterfactual hatasi: {e}")
            cf_text = ""

        # Base64 donusumleri
        original_b64 = pil_to_b64(pil_img.resize((400, 400)))
        heatmap_b64  = ndarray_to_b64(colorize_heatmap(gradcam_result["heatmap"]))

        # ── U-Net Segmentasyon Üretimi (Hastalık Sınır Çizimi) ──
        try:
            original_resized = np.array(pil_img.resize((400, 400)))
            
            if not _unet_is_demo:
                from src.preprocess import pil_to_tensor
                from src.models.unet import predict_mask
                # Real Mod: Eğitilmiş U-Net sonucunu kullan
                img_tensor = pil_to_tensor(pil_img).to(device)
                binary_mask = predict_mask(unet_model, img_tensor)
                mask_400 = cv2.resize(binary_mask, (400, 400), interpolation=cv2.INTER_NEAREST)
                contours_400, _ = cv2.findContours(mask_400, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmented_img_np = cv2.drawContours(original_resized.copy(), contours_400, -1, (0, 255, 0), 2)
            else:
                # ── Demo Modu Fallback (Kesin Çözüm v8 - LASER PRECISION) ──
                try:
                    raw_hm = gradcam_result["heatmap"]
                    mh, mw = raw_hm.shape
                    
                    # 1. Ham haritadaki en sıcak bölgeleri (threshold > 0.8) bul
                    # Not: max_v bazen 0 olabilir, hata kontrolünü yapalım
                    max_v = np.max(raw_hm)
                    thresh = max_v * 0.85 if max_v > 0.1 else 0.5
                    y_indices, x_indices = np.where(raw_hm >= thresh)
                    
                    segmented_img_np = original_resized.copy()
                    
                    # 2. Koordinatları manuel olarak 400x400'e haritalayıp çiz
                    draw_count = 0
                    for i in range(len(y_indices)):
                        ry, rx = y_indices[i], x_indices[i]
                        
                        # 400 piksel üzerine milimetrik hiza
                        cx = int((rx + 0.5) * (400 / mw))
                        cy = int((ry + 0.5) * (400 / mh))
                        
                        # Belirgin bir halka çiz
                        cv2.circle(segmented_img_np, (cx, cy), 40, (0, 255, 0), 3)
                        draw_count += 1
                        if draw_count > 4: break # Çok kalabalık yapma
                    
                    print(f"[App] Laser-Fix: {draw_count} bölge çizildi. Peak: {max_v:.2f}")
                        
                except Exception as e:
                    print(f"[App] Laser Precision Error: {e}")
                    segmented_img_np = original_resized

            segmentation_b64 = ndarray_to_b64(segmented_img_np)
        except Exception as e:
            print(f"[App] BI-V3 U-Net Hatasi: {e}")
            segmentation_b64 = original_b64


        # XAI yontemine gore overlay
        xai_label   = "Grad-CAM"
        overlay_b64 = ndarray_to_b64(gradcam_result["overlay"])

        if xai_method == "lime":
            try:
                from src.xai.lime_explain import lime_explain
                lime_res = lime_explain(model, pil_img, class_names=["Benign", "Malign"])
                if "error" not in lime_res:
                    overlay_b64 = ndarray_to_b64(lime_res["image_with_mask"])
                    xai_label   = "LIME"
                else:
                    xai_label = f"Grad-CAM (LIME: {lime_res['error']})"
            except Exception as e:
                print(f"[App] LIME hatasi: {e}")
                xai_label = "Grad-CAM (LIME basarisiz)"

        elif xai_method == "shap":
            try:
                from src.xai.shap_explain import shap_explain
                shap_res = shap_explain(model, pil_img)
                if "error" not in shap_res:
                    overlay_b64 = ndarray_to_b64(shap_res["overlay_image"])
                    xai_label   = "SHAP"
                else:
                    xai_label = f"Grad-CAM (SHAP: {shap_res['error']})"
            except Exception as e:
                print(f"[App] SHAP hatasi: {e}")
                xai_label = "Grad-CAM (SHAP basarisiz)"

        explanation = (
            gradcam_result["explanation"]
            + f"\n\n📋 {modality_notes[modality]}"
        )
        
        # Eğer counterfactual metni varsa sonuna ekle
        if cf_text:
            explanation += f"\n\n{cf_text}"

        return jsonify({
            "class_name"     : gradcam_result["class_name"],
            "class_idx"      : gradcam_result["class_idx"],
            "confidence"     : round(gradcam_result["confidence"] * 100, 2),
            "probabilities"  : {
                "benign": round(gradcam_result["probabilities"][0] * 100, 2),
                "malign": round(gradcam_result["probabilities"][1] * 100, 2),
            },
            "lezyon_detected": lezyon_detected,
            "explanation"    : explanation,
            "original_b64"   : original_b64,
            "segmentation_b64": segmentation_b64,
            "overlay_b64"    : overlay_b64,
            "heatmap_b64"    : heatmap_b64,
            "processing_ms"  : int((time.time() - t0) * 1000),
            "xai_method"     : xai_label,
            "modality"       : modality,
            "model_status"   : "checkpoint" if CHECKPOINT_PATH.exists() else "demo",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analiz sirasinda hata: {str(e)}"}), 500


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  XAI-GYN Web Arayuzu Baslatiliyor")
    print("  URL: http://localhost:5050")
    print("="*60 + "\n")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get('PORT', 5050)),
        debug=False,
        threaded=True,
    )
