import io
import base64
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import datetime

def generate_clinical_report(report_data: dict) -> bytes:
    """
    Rapor verisinden PDF dökümanı üretir (byte olarak).
    """
    buffer = io.BytesIO()
    # A4 boyutu: 210 x 297 mm -> 595.27 x 841.89 points
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # 1. Başlık ve Üst Bilgi Başlangıcı
    c.setFillColorRGB(0.1, 0.2, 0.3)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(2 * cm, height - 2.5 * cm, "XAI-GYN Klinik Analiz Raporu")
    
    # Alt çizgi
    c.setStrokeColorRGB(0.3, 0.6, 0.9)
    c.setLineWidth(2)
    c.line(2 * cm, height - 2.8 * cm, width - 2 * cm, height - 2.8 * cm)

    # 2. Rapor Meta Verileri
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica", 11)
    
    now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    c.drawString(2 * cm, height - 3.5 * cm, f"Tarih/Saat: {now_str}")
    c.drawString(2 * cm, height - 4.2 * cm, f"Risk Skoru (Malignite): %{report_data.get('malign_prob', '?')}")
    c.drawString(2 * cm, height - 4.9 * cm, f"Sınıf: {report_data.get('class_name', '?')} (Güven: %{report_data.get('confidence', '?')})")
    
    modality = report_data.get("modality", "Bilinmiyor").capitalize()
    xai_method = report_data.get("xai_method", "Bilinmiyor")
    c.drawString(2 * cm, height - 5.6 * cm, f"Modalite: {modality} | XAI Metodu: {xai_method}")

    # Lezyon Tespiti
    if report_data.get("lezyon_detected", False):
        c.setFillColorRGB(0.8, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, height - 6.5 * cm, "DIKKAT: Lezyon Tespit Edildi (> %30 Malign Riski)")
    else:
        c.setFillColorRGB(0.1, 0.6, 0.2)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, height - 6.5 * cm, "Düşük Risk: Belirgin Lezyon Bulgusu Yok")

    # 3. Görüntüleri Ekle
    c.setFillColorRGB(0.1, 0.2, 0.3)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, height - 8 * cm, "Görüntü Analizi")

    # Görüntü boyutları (genişlik ~5cm, yükseklik orantılı)
    img_width = 5 * cm
    img_height = 5 * cm
    start_y = height - 13.5 * cm

    def draw_b64_img(b64_str, x_pos, label):
        try:
            img_data = base64.b64decode(b64_str)
            img_io = io.BytesIO(img_data)
            img = Image.open(img_io)
            # Raporlab için ImageReader kullanılır, ancak geçici dosyaya yazma yerine
            # PIL Image objesini destekleyip desteklemediğine bakılır. PIL destekleniyor.
            c.drawImage(io.BytesIO(img_data), x_pos, start_y, width=img_width, height=img_height)
            
            c.setFillColorRGB(0.3, 0.3, 0.3)
            c.setFont("Helvetica", 10)
            # Ortalama için
            c.drawString(x_pos, start_y - 0.5 * cm, label)
        except Exception as e:
            print(f"Resim cizim hatasi: {e}")

    orig_b64 = report_data.get("original_b64")
    heatmap_b64 = report_data.get("heatmap_b64")
    overlay_b64 = report_data.get("overlay_b64")

    if orig_b64:
        draw_b64_img(orig_b64, 2 * cm, "Orijinal Goruntu")
    if heatmap_b64:
        draw_b64_img(heatmap_b64, 8 * cm, "Isi Haritasi")
    if overlay_b64:
        draw_b64_img(overlay_b64, 14 * cm, "Süperpozisyon (XAI)")

    # 4. Açıklama Metni (Counterfactual + XAI notları)
    c.setFillColorRGB(0.1, 0.2, 0.3)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, start_y - 2 * cm, "Klinik Açıklama ve XAI Yorumu")

    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica", 10)
    
    explanation_text = report_data.get("explanation", "")
    # Çok satırlı metni PDF'e yazmak için satırlara bölelim
    text_y = start_y - 2.8 * cm
    
    # Türkçe karakter hatalarını basitçe aşmak için standart encoding kullanabiliriz 
    # veya sadece ingilizce çevirisi yapabiliriz fakat Helvetica basit TR uyar.
    # Replace non-ascii to avoid font issues with reportlab default fonts if needed:
    tr_map = str.maketrans("ğüşöçiĞÜŞÖÇİ", "gusociGUSOCI")
    
    for line in explanation_text.split('\n'):
        if not line.strip():
            text_y -= 0.3 * cm
            continue
        # Çok uzun satırları kes
        words = line.split()
        current_line = ""
        for word in words:
            if c.stringWidth(current_line + " " + word, "Helvetica", 10) < (width - 4 * cm):
                current_line += " " + word
            else:
                c.drawString(2 * cm, text_y, current_line.strip().translate(tr_map))
                text_y -= 0.5 * cm
                current_line = word
        if current_line:
            c.drawString(2 * cm, text_y, current_line.strip().translate(tr_map))
            text_y -= 0.5 * cm

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    footer_text = "UYARI: Bu sistem klinik tanı amacli degildir, arastirma icindir. Uzman doktor gorusune basvurunuz."
    c.drawString(2 * cm, 1 * cm, footer_text)

    # Kaydet
    c.save()
    buffer.seek(0)
    return buffer.read()
