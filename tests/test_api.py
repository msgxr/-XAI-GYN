"""
XAI-GYN | tests/test_api.py
Flask API end-to-end entegrasyon testleri
"""
import sys
import io
from pathlib import Path
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from web.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Health kontrol endpoint'inin calistigini dogrular."""
    rv = client.get("/health")
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data["status"] == "ok"
    assert "model_status" in json_data

def test_predict_endpoint_no_image(client):
    """Gorsel gonderilmediginde 400 hatasi dondurmeli."""
    rv = client.post("/predict")
    assert rv.status_code == 400
    assert "error" in rv.get_json()

def test_predict_endpoint_with_image(client):
    """Geçerli bir görsel ile analiz endpoint'i doğru JSON döndürmeli."""
    # Dummy RGB image
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    data = {
        "image": (img_bytes, "test.jpg"),
        "xai_method": "gradcam"
    }

    rv = client.post("/predict", data=data, content_type="multipart/form-data")
    assert rv.status_code == 200
    json_data = rv.get_json()
    
    # Beklenen JSON anahtarlari
    expected_keys = [
        "class_name", "class_idx", "confidence", "probabilities",
        "explanation", "original_b64", "overlay_b64", "heatmap_b64", "processing_ms"
    ]
    for key in expected_keys:
        assert key in json_data, f"'{key}' yanit JSON'unda eksik"
        
    assert json_data["class_idx"] in [0, 1]
    assert 0 <= json_data["confidence"] <= 100
