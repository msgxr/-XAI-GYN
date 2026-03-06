import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

def simulate_image_change(image_np, modification_type, intensity=0.1):
    """
    Görüntüde belirli bir değişikliği simüle eder (Counterfactual analizi için).
    modification_type: 'blur', 'brightness', 'contrast', 'noise'
    intensity: Değişiklik şiddeti (0.0 - 1.0)
    """
    img = image_np.copy()
    
    if modification_type == 'blur':
        # Lezyon sınır düzensizliğini azaltma simülasyonu (Gaussian Blur)
        k_size = int(intensity * 20)
        if k_size % 2 == 0:
            k_size += 1
        if k_size > 0:
            img = cv2.GaussianBlur(img, (k_size, k_size), 0)
            
    elif modification_type == 'brightness':
        # Aydınlanma / doku gölgelenmesini azaltma 
        img = cv2.convertScaleAbs(img, alpha=1, beta=int(intensity * 100))
        
    elif modification_type == 'contrast':
        # Kontrast azaltma (doku homojenliğini artırma simülasyonu)
        alpha = 1.0 - (intensity * 0.5) # 1.0 ile 0.5 arası
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
    return img

def generate_counterfactual_explanation(model, original_pil_img, current_prob, current_class, device='cpu'):
    """
    Karşı-olgusal (Counterfactual) XAI açıklaması üretir.
    "Eğer X özelliği %Y daha az olsaydı, sonuç Z olurdu" şeklinde metin döndürür.
    """
    if current_class == "Benign" or current_prob < 0.5:
        # Zaten iyi huylu veya düşük riskliyse counterfactual üretmeye gerek yok (veya farklı bir yorum yapılabilir)
        return "Görüntü düşük riskli (Benign) grupta olduğundan karşı-olgusal risk azaltma analizine gerek görülmemiştir."
        
    # Sadece malign (kötü huylu) tahminler için neyin riski düşüreceğini arıyoruz
    original_np = np.array(original_pil_img)
    
    # 1. Deney: Sınır düzensizliğini (kenar keskinliğini) azaltma (Blur simülasyonu)
    # 2. Deney: Doku heterojenliğini (kontrastı) azaltma 
    
    experiments = [
        {'type': 'blur', 'name': 'lezyon sınır düzensizliği', 'intensities': [0.1, 0.2, 0.3]},
        {'type': 'contrast', 'name': 'doku heterojenliği', 'intensities': [0.1, 0.2, 0.3]}
    ]
    
    # Modelin transform fonksiyonunu almamız lazım (preprocess'ten)
    try:
        from src.preprocess import get_inference_transform
        transform = get_inference_transform()
    except ImportError:
        # Fallback transform if not found in current context
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    explanation_found = False
    best_explanation = ""
    
    model.eval()
    
    for exp in experiments:
        for intensity in exp['intensities']:
            # Simüle edilmiş görüntüyü oluştur
            simulated_np = simulate_image_change(original_np, exp['type'], intensity)
            
            # Modele ver
            if hasattr(transform, '__call__') and not isinstance(transform, torch.nn.Module):
                # Albumentations transform
                try:
                    tensor_img = transform(image=simulated_np)["image"].unsqueeze(0).to(device)
                except TypeError:
                     # Torchvision fallback
                     sim_pil = Image.fromarray(simulated_np)
                     tensor_img = transform(sim_pil).unsqueeze(0).to(device)
            else:
                sim_pil = Image.fromarray(simulated_np)
                tensor_img = transform(sim_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor_img)
                probs = F.softmax(outputs, dim=1)[0]
                malign_prob = probs[1].item()
                
            # Eğer malign riski %50'nin altına (veya mevcut durumdan %20 aşağı) düştüyse sebebi bulduk!
            if malign_prob < 0.5:
                reduction_pct = int(intensity * 100)
                best_explanation = f"💡 Karşı-Olgusal (Counterfactual) Analiz:\nEğer görüntüdeki {exp['name']} yaklaşık %{reduction_pct} daha az olsaydı, modelin malign (kötü huylu) tahmini %{int(malign_prob*100)} seviyesine düşecek ve risk düşük sınıfa (Benign) girecekti."
                explanation_found = True
                break
                
        if explanation_found:
            break
            
    if not explanation_found:
        best_explanation = "💡 Karşı-Olgusal (Counterfactual) Analiz:\nGörüntüdeki malign bulgular oldukça belirgin. Sınır düzensizliği veya doku heterojenliğinde yapılacak varsayımsal küçük iyileştirmeler lezyonun risk sınıfını (Malign) değiştirmek için yeterli olmamıştır."
        
    return best_explanation

