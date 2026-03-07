"""
XAI-GYN | src/train.py
Model eğitim döngüsü — Early Stopping, LR Scheduler, Checkpoint
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Proje kök dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import get_model, save_checkpoint
from src.dataset import create_dataloaders


# ─────────────────────────────────────
# Eğitim Konfigürasyonu
# ─────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir"        : "data/processed",
    "checkpoint_dir"  : "models/checkpoints",
    "epochs"          : 50,
    "batch_size"      : 32,
    "learning_rate"   : 1e-4,
    "weight_decay"    : 1e-4,
    "patience"        : 8,          # Early stopping sabır sayısı
    "val_split"       : 0.15,
    "test_split"      : 0.10,
    "num_workers"     : 0,
    "seed"            : 42,
}


# ─────────────────────────────────────
# Tek Epoch Eğitimi
# ─────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────
# Validasyon
# ─────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────
# Ana Eğitim Fonksiyonu
# ─────────────────────────────────────
def train(config: dict = None):
    if config is None:
        config = DEFAULT_CONFIG

    # Cihaz
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  XAI-GYN Model Eğitimi Başlıyor")
    print(f"  Cihaz  : {device}")
    print(f"  Epochs : {config['epochs']}")
    print(f"  Batch  : {config['batch_size']}")
    print(f"  LR     : {config['learning_rate']}")
    print(f"{'='*60}\n")

    # DataLoader'lar
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_dir   = config["data_dir"],
        batch_size = config["batch_size"],
        val_split  = config["val_split"],
        test_split = config["test_split"],
        num_workers= config["num_workers"],
        seed       = config["seed"],
    )

    # Model
    model = get_model(device=device, pretrained=True)

    # Kayıp fonksiyonu (sınıf ağırlıklı)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # LR Scheduler
    # ReduceLROnPlateau does not accept verbose in some torch versions
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Early Stopping değişkenleri
    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint = Path(config["checkpoint_dir"]) / "best_model.pth"
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ─── Eğitim Döngüsü ───
    for epoch in range(1, config["epochs"] + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Eğer val_loader boşsa (ör. tüm veri eğitim için kullanıldı) valide atla
        if len(val_loader.dataset) > 0:
            val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        scheduler.step(val_acc)

        elapsed = time.time() - t_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:03d}/{config['epochs']}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Süre: {elapsed:.1f}s"
        )

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, str(best_checkpoint),
                epoch=epoch, val_acc=val_acc,
                optimizer_state=optimizer.state_dict(),
            )
            print(f"  ✓ Yeni en iyi model kaydedildi! (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience']}")

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping! En iyi val_acc: {best_val_acc:.4f}")
            break

    print(f"\nEğitim tamamlandı. En iyi val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {best_checkpoint}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI-GYN Model Eğitimi")
    parser.add_argument("--data_dir",   type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--epochs",     type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["data_dir"]      = args.data_dir
    config["epochs"]        = args.epochs
    config["batch_size"]    = args.batch_size
    config["learning_rate"] = args.lr

    train(config)
