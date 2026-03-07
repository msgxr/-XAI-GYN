# 📄 Model Kartı: XAI-GYN (EfficientNet-B0)

Bu belge, XAI-GYN projesi kapsamında eğitilen ve dağıtılan kolposkopi sınıflandırma modelinin yeteneklerini, sınırlamalarını ve kullanım bağlamını özetlemektedir.

## 1. Model Detayları
- **Mimari:** EfficientNet-B0 (PyTorch)
- **Ön Eğitim:** ImageNet-1K (Transfer Learning)
- **Girdi Formatı:** 224x224 piksel RGB Görüntü (Tensör)
- **Çıktı Sınıfları:** İki sınıf (Benign / Malign) logit değerleri üzerinden Softmax olasılıkları.
- **XAI Entegrasyonu:** Grad-CAM ile son konvolüsyon katmanı (features[-1]) üzerinden görsel açıklama sağlama.
- **Lisans:** MIT Lisansı (Araştırma ve Eğitim Amaçlı)

## 2. Kullanım Amacı (Intended Use)
- **Birincil Hedef Kitle:** Tıbbi görüntüleme alanında çalışan araştırmacılar, asistan hekimler, veri bilimciler ve jinekolojik hastalıkların yapay zeka ile tespiti konusunda çalışan akademisyenler.
- **Birincil Kullanım Senaryosu:** Kolposkopi görüntülerinde şüpheli bulguların (malignite potansiyeli bağlamında) hekimlere destek olacak şekilde "ikinci bir görüş" olarak renklendirilmiş (yapay zeka ısı haritasıyla) sunulması.
- **Olası Hatalı Kullanım Senaryoları:** Canlı, üretim ortamında (production) tek başına bir teşhis aracı olarak, klinik bir patolog kararı yerine kullanılması kesinlikle hatalıdır.

## 3. Eğitim Verisi ve Metrikler
*(Not: Sistem, örnek bir küçük veri seti ya da özel yerel veri setinde (ör. celler_150200) çalıştırılabileceği için elde edilecek kesin rakamlar veri setine göre değişiklik gösterir. Aşağıdaki alanlar temsilidir.)*

- **Kullanılabilecek Veri Setleri:** SIPaKMeD, MobileODT, Herlev Dataset.
- **Sınıf Oranları:** Eğitim aşamasında CrossEntropyLoss ile sınıf dengesizliklerine (`class weights`) karşı ağırlıklandırma yapılmaktadır.
- **Metrikler:** Accuracy, AUC-ROC, F1 Skoru sistemce `evaluate.py` ile ölçülmektedir. 

## 4. Etik Çekinceler ve Yanlılık (Bias)
- **Popülasyon Yanlılığı (Demographic Bias):** Model, yalnızca eğitildiği veri setindeki hastaların demografik özelliklerine göre kalibre edilmiştir. Başka bölgesel farklılıklara (ışıklandırma, cihaz modeli vb.) maruziyet, modelin performansını düşürebilir (Domain Shift kavramı).
- **Açıklanabilirlik ve Güvenilirlik:** Modelin ürettiği Grad-CAM ısı haritaları korelasyona dayalıdır; modelin dikkati her zaman tıbbi nedenselliği (causality) yansıtmayabilir.

## 5. Uygulama ve Regülasyon Kapsamı
- **Sorumluluk Reddi:** Bu yazılım hiçbir tıbbi otorite (FDA, CE vb.) tarafından onaylanmamıştır ve in-vitro diyagnostik veya tıbbi cihaz statüsünde değildir. 
- Sağlanan tüm yazılım **olduğu gibi (as is)** sunulmakta olup, geliştirici ekip teşhise dayalı kayıplardan sorumlu değildir.
