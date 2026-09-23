import argparse
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 7


class SkinDataset(Dataset):
    def __init__(self, images, labels, train=False):
        self.x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        self.y = torch.from_numpy(labels.flatten()).long()
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        if self.train:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [2])  # horizontal flip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [1])  # vertical flip
            x = torch.rot90(x, k=int(torch.randint(0, 4, (1,))), dims=[1, 2])
        return x, y


def get_loaders(path, batch_size):
    d = np.load(path)
    print("Arrays:", list(d.keys()))
    train_ds = SkinDataset(d["train_images"], d["train_labels"], train=True)
    val_ds = SkinDataset(d["val_images"], d["val_labels"])
    test_ds = SkinDataset(d["test_images"], d["test_labels"])
    print(f"Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")

    counts = np.bincount(train_ds.y.numpy(), minlength=NUM_CLASSES)
    print("Train class distribution:", counts)
    class_weights = torch.tensor(counts.sum() / (NUM_CLASSES * counts),
                                 dtype=torch.float32)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        class_weights,
    )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNETclassifier(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Sequential(nn.Dropout(0.3),
                                  nn.Linear(base + base * 8, num_classes))

    @staticmethod
    def _match(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            return F.interpolate(x, size=ref.shape[2:], mode="nearest")
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        feat = torch.cat([d1.mean(dim=[2, 3]), b.mean(dim=[2, 3])], dim=1)
        return self.head(feat)


class TransferResnet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, img_size=64, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.backbone.fc.in_features, num_classes))
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(F.softmax(logits, dim=1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def compute_metrics(y_true, probs):
    y_pred = np.argmax(probs, axis=1)
    labels = list(range(NUM_CLASSES))
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr",
                            average="macro", labels=labels)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_macro_ovr": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_model(name, model, loaders, class_weights, device, epochs, lr,
                patience=7, min_delta=1e-4):
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Lowers LR when val AUC stalls; fits early stopping better than a fixed cosine schedule.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 2))

    best_auc, best_state, best_epoch = -1.0, None, 0
    epochs_no_improve = 0
    epochs_run = 0
    print(f"\n===== Training {name} (max {epochs} epochs, patience {patience}) =====")
    start = time.time()

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        val_probs, val_y = predict(model, val_loader, device)
        val_m = compute_metrics(val_y, val_probs)
        auc = val_m["auc_macro_ovr"]
        scheduler.step(0.0 if np.isnan(auc) else auc)

        print(f"[{name}] epoch {epoch:02d}/{epochs} | "
              f"loss {running / len(train_loader.dataset):.4f} | "
              f"val acc {val_m['accuracy']:.4f} | val AUC {auc:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if best_state is None or (not np.isnan(auc) and auc > best_auc + min_delta):
            if not np.isnan(auc):
                best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{name}] early stopping at epoch {epoch} "
                      f"(no val AUC improvement for {patience} epochs; "
                      f"best was epoch {best_epoch})")
                break

    train_time = time.time() - start

    model.load_state_dict(best_state)
    test_probs, test_y = predict(model, test_loader, device)
    results = compute_metrics(test_y, test_probs)
    results["training_time_sec"] = round(train_time, 2)
    results["best_val_auc"] = float(best_auc)
    results["best_epoch"] = best_epoch
    results["epochs_run"] = epochs_run
    results["params_millions"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    torch.save(best_state, f"{name}_best.pt")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dermamnist.npz")
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs (ceiling)")
    ap.add_argument("--patience", type=int, default=7, help="early-stopping patience")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-unet", type=float, default=1e-3)
    ap.add_argument("--lr-transfer", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, class_weights = get_loaders(
        args.data, args.batch_size)
    loaders = (train_loader, val_loader, test_loader)

    all_results = {}
    all_results["UNet"] = train_model(
        "UNet", UNETclassifier(), loaders, class_weights, device,
        args.epochs, args.lr_unet, patience=args.patience)
    all_results["ResNet18_Transfer"] = train_model(
        "ResNet18_Transfer", TransferResnet(), loaders, class_weights, device,
        args.epochs, args.lr_transfer, patience=args.patience)

    print("\n" + "=" * 90)
    print(f"{'Model':<20}{'Acc':>8}{'F1(macro)':>11}{'F1(wtd)':>9}{'AUC':>8}"
          f"{'BestEp':>8}{'Time(s)':>10}{'Params(M)':>11}")
    print("-" * 90)
    for name, r in all_results.items():
        print(f"{name:<20}{r['accuracy']:>8.4f}{r['f1_macro']:>11.4f}{r['f1_weighted']:>9.4f}"
              f"{r['auc_macro_ovr']:>8.4f}{r['best_epoch']:>8d}"
              f"{r['training_time_sec']:>10.1f}{r['params_millions']:>11.2f}")
    print("=" * 90)

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results.json and model weights (*.pt)")


if __name__ == "__main__":
    main()