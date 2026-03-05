# GitHub'a Yükleme Rehberi

## 📋 Adım Adım GitHub Repository Kurulumu

### 1️⃣ GitHub'da Yeni Repository Oluşturun

1. [GitHub.com](https://github.com)'e giriş yapın
2. **Sağ üstte "+" → "New repository"** tıklayın
3. Repository adını girin: `xai-gyn`
4. Açıklamayı kopyalayın:
   ```
   Açıklanabilir Yapay Zeka ile Kolposkopi Görüntü Analizi
   ```
5. **Public** seçini (halka açık)
6. `.gitignore`: Python seçin
7. License: **MIT License** seçin
8. **"Create repository"** tıklayın

---

### 2️⃣ Lokal Kurulum

Windows PowerShell'de:

```powershell
cd C:\Users\muham\.gemini\antigravity\scratch\xai-gyn

# Git repository'sini başlatın
git init

# Tüm dosyaları ekleyin
git add .

# İlk commit
git commit -m "Initial commit: XAI-GYN kolposkopi analiz sistemi"

# Remote repository'i ekleyin (USERNAME kısmını değiştirin!)
git remote add origin https://github.com/YOUR_USERNAME/xai-gyn.git

# Main branch'a pushleyin
git branch -M main
git push -u origin main
```

> **Not:** `YOUR_USERNAME` yerine GitHub kullanıcı adınızı yazın!

---

### 3️⃣ Repository Bilgileri

```bash
# Remote'u kontrol etmek için:
git remote -v

# Çıktı:
# origin  https://github.com/YOUR_USERNAME/xai-gyn.git (fetch)
# origin  https://github.com/YOUR_USERNAME/xai-gyn.git (push)
```

---

## 📤 Dosya Yapısı (Push Edilecek)

```
xai-gyn/
├── src/                    ✅ Push edilecek
├── web/                    ✅ Push edilecek
├── tests/                  ✅ Push edilecek
├── README.md               ✅ Push edilecek
├── FIXES.md                ✅ Push edilecek (güncellemeler)
├── requirements.txt        ✅ Push edilecek (düzeltilmiş)
├── setup_and_run.bat       ✅ Push edilecek
├── demo_inference.py       ✅ Push edilecek
├── .gitignore              ✅ Push edilecek
├── .github/workflows/      (İsteğe bağlı - CI/CD)
└── data/sample/test.jpg    ❌ .gitignore'da yok
```

---

## 🔄 Sonraki İşlemlerde Push

Kod değişiklikleri yaptıktan sonra:

```bash
# Değişiklikleri ekleyin
git add .

# Commit edin
git commit -m "Açıklayıcı mesaj"

# Push edin
git push origin main
```

---

## 🔗 Faydalı Komutlar

```bash
# Status kontrol
git status

# Son commit'leri görmek
git log --oneline -5

# Geçmişi görmek
git log --oneline

# Belirli bir dosyanın değişikliklerini görmek
git diff README.md

# Değişiklikleri unstage etmek
git reset HEAD <dosya>

# Son commit'i geri almak
git revert HEAD
```

---

## 🎯 İlk Push Kontrolü

Push yaptıktan sonra GitHub sayfanızı yenileyin:
- ✅ Tüm dosyalar görülüyor mu?
- ✅ README.md düzgün render oluyor mu?
- ✅ Code tab'ında proje yapısı doğru mu?

---

## 🌟 Opsiyonel: GitHub Badges Ekleyin (README'ye)

README.md dosyasının en başına ekleyin:

```markdown
# XAI-GYN 🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-27%2F27%20passed-brightgreen.svg)](https://github.com/YOUR_USERNAME/xai-gyn/actions)

> **Açıklanabilir Yapay Zeka ile Kadın Hastalıklarında Akıllı Görüntü Analizi**  
```

---

## ❓ Sorun Giderme

### ❌ "Permission denied (publickey)"
```bash
# SSH key oluşturun:
ssh-keygen -t ed25519 -C "your_email@example.com"

# GitHub Settings → SSH Keys → Yeni key ekleyin
```

### ❌ "fatal: not a git repository"
```bash
# Repository başlat:
git init
```

### ❌ "fatal: 'origin' does not appear to be a git repository"
```bash
# Remote ekleyin:
git remote add origin https://github.com/YOUR_USERNAME/xai-gyn.git
```

---

## 📞 Destek

Sorular için GitHub Issues kullanın!

---

**Tamamlandı! 🎉 Projeniz artık GitHub'da!**
