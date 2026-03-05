@echo off
:: XAI-GYN Kurulum ve Başlatma Scripti (Windows)
:: Çalıştırmak için çift tıklayın veya: .\setup_and_run.bat

echo ==========================================
echo   XAI-GYN Kurulum ve Baslatma
echo ==========================================

:: Python var mı kontrol et
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi! Lutfen Python 3.9+ yukleyin.
    pause
    exit /b 1
)

echo [1/4] Python kontrol edildi ✓

:: Sanal ortam oluştur
if not exist "venv" (
    echo [2/4] Sanal ortam olusturuluyor...
    python -m venv venv
) else (
    echo [2/4] Sanal ortam mevcut ✓
)

:: Aktif et
call venv\Scripts\activate.bat
echo [3/4] Sanal ortam aktif edildi ✓

:: Bağımlılıkları kur
echo [4/4] Kutuphaneler kuruluyor...
pip install -q -r requirements.txt

echo.
echo ==========================================
echo   Sunucu Baslatiliyor: http://localhost:5000
echo ==========================================
echo.

:: Flask uygulamasını başlat
cd web
python app.py

pause
