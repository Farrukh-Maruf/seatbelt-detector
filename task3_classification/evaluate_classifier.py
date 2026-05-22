from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (auc, confusion_matrix, precision_recall_fscore_support,
                                roc_curve)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "classifier"
FIG = REPO / "task3_classification" / "figures"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch: str) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(arch)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--precision-target", type=float, default=0.90,
                    help="precision(kamar_yoq) we want at the precision-first threshold")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.weights, map_location="cpu")
    arch = ckpt.get("arch", "mobilenet_v3_small")
    class_to_idx = ckpt["class_to_idx"]
    bor_idx = class_to_idx["kamar_bor"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(arch); model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_ds = datasets.ImageFolder(DATA / "val", transform=val_tf)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    probs, ys = [], []
    with torch.no_grad():
        for x, y in val_dl:
            p = torch.sigmoid(model(x.to(device))).cpu().squeeze(1).numpy()
            probs.extend(p.tolist())
            ys.extend([(int(t) == bor_idx) for t in y])   # y=1 -> kamar_bor
    probs = np.array(probs); ys = np.array(ys).astype(int)
    print(f"val crops: {len(ys)}  (kamar_bor={ys.sum()}, kamar_yoq={(1-ys).sum()})")

    # ROC + AUC (positive = kamar_bor)
    fpr, tpr, thr = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    youden = thr[np.argmax(tpr - fpr)]
    print(f"\nROC-AUC: {roc_auc:.4f}   (target >= 0.92)")
    print(f"Youden-J threshold: {youden:.3f}")

    # aniqlik birinchi: biz juda kam notogri ijobiy natijalarni xohlaymiz (belt
    # toqish kerak boʻlgan haydovchi noto'g'ri tarzda kamar_yoq deb belgilangan = nohaq
    # jarima). Haydovchi P(kamar_bor) < t boʻlganda kamar_yoq deb belgilanadi, shuning
    # uchun t ni PASAYTIRISH kamroq haydovchini belgilaydi -> kamar_yoq aniqligini
    # oshiradi (kamroq FP) lekin eslab qolish kamayadi. Aniqlik maqsadiga erishadigan
    # barcha chegaralardan biz ENG YUQORI birini tanlaymiz, chunki u eslab qolishni
    # katta qoldirib, FP chekloviga mos keladi. (Pastga tushish faqat eslab qolishni
    # qurbonlik qiladi, aniqlikka foyda bermaydi.)

    candidates = []
    for t in np.linspace(0.05, 0.95, 91):
        pred_bor = (probs >= t).astype(int)
        pred_yoq = 1 - pred_bor
        true_yoq = 1 - ys
        tp = int(((pred_yoq == 1) & (true_yoq == 1)).sum())
        fp = int(((pred_yoq == 1) & (true_yoq == 0)).sum())
        prec_yoq = tp / (tp + fp) if (tp + fp) else 1.0
        rec_yoq = tp / true_yoq.sum() if true_yoq.sum() else 0.0
        if prec_yoq >= args.precision_target:
            candidates.append((t, prec_yoq, rec_yoq))
    # highest recall among precision-satisfying thresholds
    pf_t = max(candidates, key=lambda c: c[2])[0] if candidates else youden
    print(f"Precision-first threshold (P(kamar_yoq)>={args.precision_target}): {pf_t:.3f}")

    def report(t, label):
        pred_bor = (probs >= t).astype(int)
        acc = (pred_bor == ys).mean()
        # class order for sklearn: 0=kamar_yoq, 1=kamar_bor
        p, r, f1, _ = precision_recall_fscore_support(
            ys, pred_bor, labels=[0, 1], zero_division=0)
        print(f"\n--- {label} (threshold={t:.3f}) ---")
        print(f"Accuracy: {acc:.4f}   (target >= 0.90)")
        print(f"kamar_yoq (y=0): P={p[0]:.4f} R={r[0]:.4f} F1={f1[0]:.4f}"
                f"   (target P>=0.88, R>=0.85)")
        print(f"kamar_bor (y=1): P={p[1]:.4f} R={r[1]:.4f} F1={f1[1]:.4f}")
        cm = confusion_matrix(ys, pred_bor, labels=[0, 1])
        print(f"Confusion (rows=true [yoq,bor], cols=pred [yoq,bor]):\n{cm}")
        # FP = belted driver predicted as no-belt = true bor, pred yoq
        fp = int(((pred_bor == 0) & (ys == 1)).sum())
        fn = int(((pred_bor == 1) & (ys == 0)).sum())
        print(f"FP (belted -> fined!): {fp}    FN (no-belt -> missed): {fn}")
        return fp, fn

    fp_y, fn_y = report(youden, "Youden-J")
    fp_p, fn_p = report(pf_t, "Precision-first scan")

    # Ishlab chiqarish qoidasi (CLAUDE.md 4 aniqlik birinchi): eng kam noto'g'ri ijobiy
    # natijaga (nohaq jarima) ega bo'lgan chegarani tanlaymiz. ESLATMA: bu ROC-da aniqlik
    # skaneri Youden-J ni FP bilan yengmaydi, shuning uchun Youden-J odatda ishlab chiqarish
    # tanloviga olinadi — biz FP bilan tanlaymiz, nom bilan emas va qaysi biri g'alaba qozonganini aytamiz.
    if fp_y <= fp_p:
        print(f"\n>> PRODUCTION threshold = Youden-J ({youden:.3f}) — fewest FP ({fp_y}).")
    else:
        print(f"\n>> PRODUCTION threshold = precision-scan ({pf_t:.3f}) — fewest FP ({fp_p}).")

    # save ROC figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC — kamar_bor")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "roc_curve.png", dpi=120)
    print(f"\nROC saved -> {FIG / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
