/**
 * XAI-GYN | web/static/js/main.js
 * Premium frontend logic — upload, analyze, animate, display
 */

'use strict';

// ─── SVG Gradient Definitions (injected once) ───
const SVG_DEFS = `
<svg width="0" height="0" style="position:absolute">
  <defs>
    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#4facfe"/>
      <stop offset="50%"  stop-color="#f7931e"/>
      <stop offset="100%" stop-color="#ff4d6d"/>
    </linearGradient>
  </defs>
</svg>`;
document.body.insertAdjacentHTML('afterbegin', SVG_DEFS);

// ─── DOM References ───────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadAnimation = document.getElementById('uploadAnimation');
const previewContainer = document.getElementById('previewContainer');
const previewImg = document.getElementById('previewImg');
const previewInfo = document.getElementById('previewInfo');
const resetBtn = document.getElementById('resetBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingContainer = document.getElementById('loadingContainer');
const loadingStep = document.getElementById('loadingStep');
const resultsSection = document.getElementById('resultsSection');
const errorContainer = document.getElementById('errorContainer');
const errorMsg = document.getElementById('errorMsg');
const errorRetryBtn = document.getElementById('errorRetryBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadBtn = document.getElementById('downloadBtn');

// Result elements
const gaugeFill = document.getElementById('gaugeFill');
const gaugeValue = document.getElementById('gaugeValue');
const gaugeClass = document.getElementById('gaugeClass');
const benignBar = document.getElementById('benignBar');
const malignBar = document.getElementById('malignBar');
const benignVal = document.getElementById('benignVal');
const malignVal = document.getElementById('malignVal');
const explanationText = document.getElementById('explanationText');
const processingTime = document.getElementById('processingTime');
const modelBadge = document.getElementById('modelBadge');
const originalImg = document.getElementById('originalImg');
const heatmapImg = document.getElementById('heatmapImg');
const segmentationImg = document.getElementById('segmentationImg');
const overlayImg = document.getElementById('overlayImg');

// State
let selectedFile = null;
let lastResult = null;
let selectedModality = 'kolposkopi';  // varsayilan
let selectedXai = 'gradcam';     // varsayilan
let selectedModel = 'efficientnet'; // varsayilan

// ─── Opt-Button Gruplari (Modalite + XAI + Model) ──────
function bindOptionGroup(groupId, onChange) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll('.opt-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      group.querySelectorAll('.opt-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      onChange(btn.dataset.value);
    });
  });
}

bindOptionGroup('modalityGroup', (val) => { selectedModality = val; });
bindOptionGroup('xaiGroup', (val) => { selectedXai = val; });
bindOptionGroup('modelTypeGroup', (val) => { selectedModel = val; });

// ─── Gauge Arc Math ───────────────────────────
// The gauge arc path is a half-circle (180°), total arc length ≈ 251px
const GAUGE_ARC_LENGTH = 251;

function setGauge(percentage) {
  const dashValue = (percentage / 100) * GAUGE_ARC_LENGTH;
  gaugeFill.setAttribute(
    'stroke-dasharray',
    `${dashValue.toFixed(2)} ${GAUGE_ARC_LENGTH}`
  );
  gaugeValue.textContent = `${Math.round(percentage)}%`;
}

// ─── Drag & Drop ──────────────────────────────
['dragover', 'dragenter'].forEach(evt => {
  uploadZone.addEventListener(evt, e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach(evt => {
  uploadZone.addEventListener(evt, e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
  });
});

uploadZone.addEventListener('drop', e => {
  const files = e.dataTransfer?.files;
  if (files?.length) handleFile(files[0]);
});

// ─── Click Upload ─────────────────────────────
uploadZone.addEventListener('click', () => {
  if (!previewContainer.style.display || previewContainer.style.display === 'none') {
    fileInput.click();
  }
});

uploadZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

browseBtn.addEventListener('click', e => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener('change', e => {
  if (e.target.files?.length) handleFile(e.target.files[0]);
});

// ─── Handle Selected File ─────────────────────
function handleFile(file) {
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/tif'];
  const MAX_SIZE = 32 * 1024 * 1024;

  if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(/\.(tiff|tif)$/i)) {
    showError('Klinik Uyarı: Desteklenmeyen görüntü formatı tespit edildi. Analiz doğruluğu için lütfen standart kolposkopi formatlarını (JPG, PNG, BMP, TIFF) kullanınız.');
    return;
  }

  if (file.size > MAX_SIZE) {
    showError('Sistem Uyarısı: Görüntü dosyası maksimum işleme sınırını (32 MB) aşmaktadır. Lütfen görüntüyü optimize ederek tekrar deneyiniz.');
    return;
  }

  selectedFile = file;
  hideResults();
  hideError();

  // Önizleme oluştur
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    previewImg.src = dataUrl;

    // Dosya bilgisi
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    previewInfo.textContent = `${file.name}  ·  ${sizeMB} MB  ·  ${file.type || 'image'}`;

    // Küçük gecikmeli boyut bilgisi ekle
    const tempImg = new Image();
    tempImg.onload = () => {
      previewInfo.textContent += `  ·  ${tempImg.naturalWidth}×${tempImg.naturalHeight}px`;
    };
    tempImg.src = dataUrl;

    uploadAnimation.style.display = 'none';
    previewContainer.style.display = 'block';
    previewContainer.classList.add('fade-in');
  };
  reader.readAsDataURL(file);
}

// ─── Reset ────────────────────────────────────
resetBtn.addEventListener('click', resetUpload);
newAnalysisBtn.addEventListener('click', resetUpload);

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewInfo.textContent = '';
  previewContainer.style.display = 'none';
  uploadAnimation.style.display = 'block';
  hideResults();
  hideError();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── Analyze ──────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  hideResults();
  hideError();
  showLoading();

  // Loading adım animasyonu
  const steps = ['step1', 'step2', 'step3', 'step4'];
  const stepMessages = [
    'Görüntü ön işleme yapılıyor...',
    'Model çıkarımı çalışıyor...',
    'Grad-CAM ısı haritası üretiliyor...',
    'Açıklama metni oluşturuluyor...',
  ];
  let stepDelay = 0;
  steps.forEach((sid, idx) => {
    setTimeout(() => {
      steps.forEach(s => {
        const el = document.getElementById(s);
        if (el) el.classList.remove('active');
      });
      // Mark previous as done
      if (idx > 0) {
        const prev = document.getElementById(steps[idx - 1]);
        if (prev) {
          prev.classList.remove('active');
          prev.classList.add('done');
        }
      }
      const current = document.getElementById(sid);
      if (current) current.classList.add('active');
      if (loadingStep) loadingStep.textContent = stepMessages[idx];
    }, stepDelay);
    stepDelay += 1000;
  });

  // FormData
  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('xai_method', selectedXai);
  formData.append('modality', selectedModality);
  formData.append('model_type', selectedModel);


  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Sunucu hatası: ${response.status}`);
    }

    if (data.error) {
      throw new Error(data.error);
    }

    lastResult = data;

    // Mark all steps done
    setTimeout(() => {
      steps.forEach(sid => {
        const el = document.getElementById(sid);
        if (el) {
          el.classList.remove('active');
          el.classList.add('done');
        }
      });

      setTimeout(() => {
        hideLoading();
        displayResults(data);
      }, 500);
    }, stepDelay - 200);

  } catch (err) {
    console.error('Analysis error:', err);
    hideLoading();
    showError(err.message || 'Sistem Uyarısı: Analiz sunucusuna bağlantı kurulamadı. Lütfen ağ bağlantınızı ve teşhis sunucusunun (Flask) durumunu kontrol ediniz.');
  }
}

// ─── Display Results ──────────────────────────
function displayResults(data) {
  // Model badge
  if (data.model_status === 'checkpoint') {
    modelBadge.textContent = '🔬 Egitilmis Model';
    modelBadge.classList.add('live');
  } else {
    modelBadge.textContent = '⚗️ Demo Modu';
  }

  // Processing time
  processingTime.textContent = `Islem suresi: ${data.processing_ms} ms`;

  // Lezyon Tespiti Banner
  const lezyonBanner = document.getElementById('lezyonBanner');
  const lezyonIcon = document.getElementById('lezyonIcon');
  const lezyonText = document.getElementById('lezyonText');
  if (lezyonBanner && lezyonIcon && lezyonText) {
    lezyonBanner.style.display = 'flex';
    if (data.lezyon_detected) {
      lezyonIcon.textContent = '⚠️';
      lezyonText.textContent = 'Lezyon Tespit Edildi — Detayli analiz yapiliyor';
      lezyonBanner.className = 'lezyon-banner lezyon-detected';
    } else {
      lezyonIcon.textContent = '✅';
      lezyonText.textContent = 'Belirgin Lezyon Bulgusuna Rastlanmadi';
      lezyonBanner.className = 'lezyon-banner lezyon-clear';
    }
  }

  // XAI Yontem Etiketi
  const xaiMethodLabel = document.getElementById('xaiMethodLabel');
  if (xaiMethodLabel) xaiMethodLabel.textContent = `${data.xai_method} · Karsi-Olgusal Analiz`;

  // Overlay label
  const overlayLabel = document.getElementById('overlayLabel');
  if (overlayLabel) overlayLabel.textContent = `🎯 ${data.xai_method} Overlay`;

  // Analysis meta (XAI + Modalite)
  const analysisMeta = document.getElementById('analysisMeta');
  if (analysisMeta) {
    const modalityLabels = { kolposkopi: '🔍 Kolposkopi', ultrason: '🟦 Ultrason', laparoskopi: '🔬 Laparoskopi' };
    analysisMeta.innerHTML =
      `<span class="meta-tag">${data.xai_method}</span>` +
      `<span class="meta-tag">${modalityLabels[data.modality] || data.modality}</span>`;
  }

  // Animasyonlu gauge (malign skoru)
  const malignPct = data.probabilities.malign;
  setTimeout(() => setGauge(malignPct), 100);

  // Sinif badge
  const ismalign = data.class_idx === 1;
  gaugeClass.textContent = ismalign ? '🔴 MALiGN' : '🟢 BENiGN';
  gaugeClass.classList.toggle('malign-class', ismalign);
  gaugeClass.classList.toggle('benign-class', !ismalign);

  // Probability bars (animasyonlu)
  setTimeout(() => {
    benignBar.style.width = `${data.probabilities.benign}%`;
    malignBar.style.width = `${data.probabilities.malign}%`;
    benignVal.textContent = `${data.probabilities.benign}%`;
    malignVal.textContent = `${data.probabilities.malign}%`;
  }, 200);

  // Aciklama
  explanationText.textContent = data.explanation;

  // Gorseller
  originalImg.src = `data:image/png;base64,${data.original_b64}`;
  heatmapImg.src = `data:image/png;base64,${data.heatmap_b64}`;
  if (segmentationImg && data.segmentation_b64) {
    segmentationImg.src = `data:image/png;base64,${data.segmentation_b64}`;
  }
  overlayImg.src = `data:image/png;base64,${data.overlay_b64}`;

  // Sonuclari goster
  showResults();

  // Sayfayi asagi kaydır
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

// ─── Download ─────────────────────────────────
downloadBtn.addEventListener('click', downloadResults);

async function downloadResults() {
  if (!lastResult) return;

  // Butonu loading durumuna alalim
  const originalText = downloadBtn.innerHTML;
  downloadBtn.innerHTML = '⏳ Hazırlanıyor...';
  downloadBtn.disabled = true;

  try {
    const response = await fetch('/generate_pdf', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(lastResult)
    });

    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || 'PDF olusturulamadi.');
    }

    // Base64 PDF'i indir
    const linkSource = `data:application/pdf;base64,${data.pdf_b64}`;
    const downloadLink = document.createElement("a");
    const fileName = `xai_gyn_rapor_${Date.now()}.pdf`;

    downloadLink.href = linkSource;
    downloadLink.download = fileName;
    downloadLink.click();

  } catch (err) {
    console.error('PDF indirme hatasi:', err);
    showError('Sistem Uyarısı: Klinik PDF raporu oluşturulamadı. Lütfen tekrar deneyiniz.');
  } finally {
    downloadBtn.innerHTML = originalText;
    downloadBtn.disabled = false;
  }
}

// ─── UI State Helpers ─────────────────────────
function showLoading() {
  loadingContainer.style.display = 'block';
  loadingContainer.classList.add('fade-in');
  analyzeBtn.disabled = true;
  // Reset step indicators
  ['step1', 'step2', 'step3', 'step4'].forEach(s => {
    const el = document.getElementById(s);
    if (el) { el.classList.remove('active', 'done'); }
  });
}

function hideLoading() {
  loadingContainer.style.display = 'none';
  analyzeBtn.disabled = false;
}

function showResults() {
  resultsSection.style.display = 'block';
  resultsSection.classList.add('fade-in');
}

function hideResults() {
  resultsSection.style.display = 'none';
}

function showError(message) {
  errorMsg.textContent = message;
  errorContainer.style.display = 'block';
  errorContainer.classList.add('fade-in');
  errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
  errorContainer.style.display = 'none';
}

// ─── Error retry ──────────────────────────────
errorRetryBtn.addEventListener('click', () => {
  if (selectedFile) {
    hideError();
    runAnalysis();
  } else {
    hideError();
    resetUpload();
  }
});

// ─── Health Check on Load ─────────────────────
window.addEventListener('load', async () => {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      const statusEl = document.querySelector('.nav-status span');
      if (statusEl) {
        statusEl.textContent = data.model_status === 'checkpoint'
          ? 'Model Yüklendi ✓'
          : 'Demo Modu Aktif';
      }
    }
  } catch {
    const statusEl = document.querySelector('.nav-status span');
    if (statusEl) statusEl.textContent = 'Sunucu Bekleniyor...';
    const dot = document.querySelector('.status-dot');
    if (dot) dot.style.background = '#ff4d6d';
  }
});
