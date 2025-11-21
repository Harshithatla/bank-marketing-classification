# Code will be added here
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

from data import load_data
from model import ResidualMLP


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return sum(losses) / len(losses)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true = []
    y_probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(yb.cpu().numpy())

    return (
        sum(losses) / len(losses),
        y_true,
        y_probs
    )


def train_model():

    os.makedirs("results", exist_ok=True)

    train_loader, val_loader, test_loader, input_dim = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResidualMLP(input_dim).to(device)

    pos_weight = torch.tensor([3.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    history = {"train_loss": [], "val_loss": []}

    EPOCHS = 30

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}  Train={train_loss:.4f}  Val={val_loss:.4f}")

    # Plot & Save Loss Curve
    plt.figure(figsize=(8,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("results/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Final Evaluation
    _, y_true, y_probs = evaluate(model, test_loader, criterion, device)
    y_pred = (torch.tensor(y_probs) >= 0.5).numpy().astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("results/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    train_model()
