"""
XAI-GYN | src/evaluate.py
Model değerlendirme — Accuracy, AUC-ROC, F1, Confusion Matrix
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_model(model, test_loader, device: str = "cpu") -> dict:
    """
    Test veri seti üzerinde model değerlendirmesi yapar.

    Returns:
        dict: accuracy, auc, f1, confusion_matrix, report
    """
    model.eval()
    model = model.to(device)

    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Malign olasılığı

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # Metrikler
    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average="weighted")
    auc    = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Benign", "Malign"])

    results = {
        "accuracy"        : acc,
        "f1_weighted"     : f1,
        "auc_roc"         : auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    # Ekrana yazdır
    print("\n" + "="*60)
    print("  MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*60)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC    : {auc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"\nKarmaşıklık Matrisi:\n{cm}")
    print(f"\nSınıflandırma Raporu:\n{report}")
    print("="*60)

    return results


def plot_confusion_matrix(cm: list, class_names: list, save_path: str = None):
    """Karmaşıklık matrisini görselleştirir."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Karmaşıklık Matrisi",
        ylabel="Gerçek Etiket",
        xlabel="Tahmin Edilen Etiket",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm_arr.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm_arr[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] Karmaşıklık matrisi kaydedildi: {save_path}")
    return fig


if __name__ == "__main__":
    print("Evaluate modülü yüklendi ✓")
