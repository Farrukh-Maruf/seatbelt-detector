from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "classifier"
WEIGHTS_OUT = REPO / "task3_classification" / "weights"

#   kamar_bor - 0, kamar_yoq - 1.
IMAGENET_MEAN = [0.485, 0.456, 0.406] # ImageNet stats for normalization (pretrained weights expect this)
IMAGENET_STD = [0.229, 0.224, 0.225]  # ImageNet stats for normalization (pretrained weights expect this)


def build_model(arch: str) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def loaders(batch: int):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),          # OK for a centred driver crop
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(DATA / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA / "val", transform=val_tf)

    # WeightedRandomSampler to counter imbalance (sample classes ~equally)
    targets = [y for _, y in train_ds.samples]
    class_count = [targets.count(0), targets.count(1)]
    weights = [1.0 / class_count[y] for y in targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    return train_dl, val_dl, class_count, train_ds.class_to_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, class_count, class_to_idx = loaders(args.batch)
    print(f"class_to_idx={class_to_idx}  train class counts (idx0,idx1)={class_count}")
    print("NOTE: positive class y=1 is kamar_bor (target inverted from ImageFolder idx).")

    model = build_model(args.arch).to(device)
    # pos_weight balances BCE for the minority positive (kamar_bor)
    n_bor = class_count[class_to_idx["kamar_bor"]]
    n_yoq = class_count[class_to_idx["kamar_yoq"]]
    pos_weight = torch.tensor([n_yoq / max(1, n_bor)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    bor_idx = class_to_idx["kamar_bor"]

    def to_pos(y):  # ImageFolder idx -> y=1 for kamar_bor
        return (y == bor_idx).float()

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x = x.to(device); target = to_pos(y).to(device).unsqueeze(1)
            opt.zero_grad()
            loss = criterion(model(x), target)
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        # quick val accuracy @0.5 (real ROC/Youden in evaluate_classifier.py)
        model.eval(); correct = 0; n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device); target = to_pos(y)
                prob = torch.sigmoid(model(x)).cpu().squeeze(1)
                correct += ((prob > 0.5).float() == target).sum().item(); n += len(target)
        acc = correct / max(1, n)
        print(f"epoch {epoch+1:02d}/{args.epochs}  loss={tot/len(train_dl.dataset):.4f}  val_acc@0.5={acc:.3f}")

    out = WEIGHTS_OUT / f"{args.arch}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                "class_to_idx": class_to_idx}, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
