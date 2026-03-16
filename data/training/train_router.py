"""
NyayaMitra — Query Router Training Script.

Fine-tunes DistilBERT for multi-task legal query classification:
    Task 1: Domain classification (8 classes)
        criminal, property, family, constitutional, labor, consumer, ip, general
    Task 2: Query type classification (3 classes)
        rights, procedure, case_outcome

Architecture:
    DistilBERT base → shared hidden layer → two classification heads

Training:
    - 504+ labeled queries from router_training_data.json
    - 80/20 train/val split stratified by domain
    - Multi-task loss: domain_loss + query_type_loss
    - Early stopping on validation accuracy
    - Exports model to models/router/ for inference

Usage:
    # On GPU machine or Google Colab
    python -m data.training.train_router

    # With custom settings
    python -m data.training.train_router --epochs 20 --lr 2e-5 --batch-size 16

    # Evaluate only (no training)
    python -m data.training.train_router --eval-only

    # Export for deployment
    python -m data.training.train_router --export
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logger = structlog.get_logger()

# Paths
TRAINING_DATA = PROJECT_ROOT / "data" / "training" / "router_training_data.json"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "router"

# Label mappings
DOMAIN_LABELS = [
    "criminal", "property", "family", "constitutional",
    "labor", "consumer", "ip", "general",
]
QUERY_TYPE_LABELS = ["rights", "procedure", "case_outcome"]

DOMAIN_TO_ID = {d: i for i, d in enumerate(DOMAIN_LABELS)}
ID_TO_DOMAIN = {i: d for d, i in DOMAIN_TO_ID.items()}

QTYPE_TO_ID = {q: i for i, q in enumerate(QUERY_TYPE_LABELS)}
ID_TO_QTYPE = {i: q for q, i in QTYPE_TO_ID.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Task Model
# ═══════════════════════════════════════════════════════════════════════════════


def build_model():
    """
    Build the multi-task DistilBERT classifier.

    Architecture:
        DistilBERT → [CLS] pooling → shared dense (256) → dropout →
            → domain_head (256 → 8)
            → qtype_head (256 → 3)
    """
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel

    class MultiTaskQueryRouter(nn.Module):
        def __init__(self, num_domains=8, num_qtypes=3, dropout=0.3):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
            hidden_size = self.bert.config.dim  # 768

            # Shared representation layer
            self.shared = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Task-specific heads
            self.domain_head = nn.Linear(256, num_domains)
            self.qtype_head = nn.Linear(256, num_qtypes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation
            cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            shared_repr = self.shared(cls_output)  # (batch, 256)
            domain_logits = self.domain_head(shared_repr)  # (batch, 8)
            qtype_logits = self.qtype_head(shared_repr)  # (batch, 3)
            return domain_logits, qtype_logits

    return MultiTaskQueryRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════


def load_training_data(path: Path | None = None) -> list[dict]:
    """Load and validate training data."""
    path = path or TRAINING_DATA
    with open(path) as f:
        data = json.load(f)

    valid = []
    for entry in data:
        domain = entry.get("domain", "")
        qtype = entry.get("query_type", "")
        query = entry.get("query", "")
        if domain in DOMAIN_TO_ID and qtype in QTYPE_TO_ID and query:
            valid.append(entry)

    logger.info("training_data_loaded", total=len(data), valid=len(valid))
    return valid


def create_datasets(data: list[dict], val_split: float = 0.2, seed: int = 42):
    """
    Create train/val datasets with stratified split by domain.

    Returns tokenized PyTorch datasets.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import DistilBertTokenizer
    from sklearn.model_selection import train_test_split

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    queries = [d["query"] for d in data]
    domain_ids = [DOMAIN_TO_ID[d["domain"]] for d in data]
    qtype_ids = [QTYPE_TO_ID[d["query_type"]] for d in data]

    # Stratified split by domain
    train_idx, val_idx = train_test_split(
        range(len(data)),
        test_size=val_split,
        random_state=seed,
        stratify=domain_ids,
    )

    class RouterDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices
            self.encodings = tokenizer(
                [queries[i] for i in indices],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            self.domain_labels = torch.tensor([domain_ids[i] for i in indices])
            self.qtype_labels = torch.tensor([qtype_ids[i] for i in indices])

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "domain_label": self.domain_labels[idx],
                "qtype_label": self.qtype_labels[idx],
            }

    train_dataset = RouterDataset(train_idx)
    val_dataset = RouterDataset(val_idx)

    logger.info(
        "datasets_created",
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )

    return train_dataset, val_dataset, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════


def train(
    epochs: int = 15,
    lr: float = 2e-5,
    batch_size: int = 16,
    patience: int = 5,
    domain_weight: float = 1.0,
    qtype_weight: float = 0.5,
) -> dict:
    """
    Train the multi-task query router.

    Args:
        epochs: Maximum training epochs.
        lr: Learning rate for AdamW optimizer.
        batch_size: Training batch size.
        patience: Early stopping patience (epochs without improvement).
        domain_weight: Loss weight for domain classification.
        qtype_weight: Loss weight for query type classification.

    Returns:
        Training results dict with metrics and model path.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("training_device", device=str(device))

    # Load data and create model
    data = load_training_data()
    train_dataset, val_dataset, tokenizer = create_datasets(data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    domain_criterion = nn.CrossEntropyLoss()
    qtype_criterion = nn.CrossEntropyLoss()

    # Training state
    best_val_acc = 0.0
    best_epoch = 0
    no_improvement = 0
    history = {"train_loss": [], "val_domain_acc": [], "val_qtype_acc": [], "val_combined_acc": []}

    print(f"\n  Training on {device} | {len(train_dataset)} train / {len(val_dataset)} val")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {batch_size} | Patience: {patience}")
    print("  " + "─" * 60)

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            domain_labels = batch["domain_label"].to(device)
            qtype_labels = batch["qtype_label"].to(device)

            optimizer.zero_grad()
            domain_logits, qtype_logits = model(input_ids, attention_mask)

            loss = (domain_weight * domain_criterion(domain_logits, domain_labels) +
                    qtype_weight * qtype_criterion(qtype_logits, qtype_labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # ── Validate ──
        model.eval()
        domain_correct = 0
        qtype_correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                domain_labels = batch["domain_label"].to(device)
                qtype_labels = batch["qtype_label"].to(device)

                domain_logits, qtype_logits = model(input_ids, attention_mask)
                domain_preds = domain_logits.argmax(dim=1)
                qtype_preds = qtype_logits.argmax(dim=1)

                domain_correct += (domain_preds == domain_labels).sum().item()
                qtype_correct += (qtype_preds == qtype_labels).sum().item()
                total += domain_labels.size(0)

        domain_acc = domain_correct / total
        qtype_acc = qtype_correct / total
        combined_acc = (domain_acc + qtype_acc) / 2

        history["val_domain_acc"].append(domain_acc)
        history["val_qtype_acc"].append(qtype_acc)
        history["val_combined_acc"].append(combined_acc)

        # Check improvement
        if domain_acc > best_val_acc:
            best_val_acc = domain_acc
            best_epoch = epoch + 1
            no_improvement = 0

            # Save best model
            MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "router_model.pt")
            tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
        else:
            no_improvement += 1

        print(
            f"  Epoch {epoch + 1:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Domain: {domain_acc:.3f} | "
            f"QType: {qtype_acc:.3f} | "
            f"Combined: {combined_acc:.3f}"
            f"{' ★' if no_improvement == 0 else ''}"
        )

        if no_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    # Save label mappings and config
    config = {
        "model_name": "distilbert-base-uncased",
        "num_domains": len(DOMAIN_LABELS),
        "num_qtypes": len(QUERY_TYPE_LABELS),
        "domain_labels": DOMAIN_LABELS,
        "qtype_labels": QUERY_TYPE_LABELS,
        "domain_to_id": DOMAIN_TO_ID,
        "qtype_to_id": QTYPE_TO_ID,
        "best_epoch": best_epoch,
        "best_domain_acc": best_val_acc,
        "max_length": 128,
        "training_samples": len(train_dataset),
    }
    with open(MODEL_OUTPUT_DIR / "router_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save training history
    with open(MODEL_OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("  " + "─" * 60)
    print(f"\n  Best domain accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Model saved to: {MODEL_OUTPUT_DIR}")

    return {
        "best_domain_acc": best_val_acc,
        "best_epoch": best_epoch,
        "model_dir": str(MODEL_OUTPUT_DIR),
        "history": history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate():
    """
    Evaluate the trained model and print detailed metrics.

    Loads the saved model and runs on the full dataset,
    producing per-class precision/recall/F1 and confusion matrix.
    """
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, confusion_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "router_model.pt", map_location=device))
    model.eval()

    # Load data
    data = load_training_data()
    _, val_dataset, _ = create_datasets(data)
    val_loader = DataLoader(val_dataset, batch_size=32)

    all_domain_preds = []
    all_domain_true = []
    all_qtype_preds = []
    all_qtype_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            domain_logits, qtype_logits = model(input_ids, attention_mask)

            all_domain_preds.extend(domain_logits.argmax(dim=1).cpu().numpy())
            all_domain_true.extend(batch["domain_label"].numpy())
            all_qtype_preds.extend(qtype_logits.argmax(dim=1).cpu().numpy())
            all_qtype_true.extend(batch["qtype_label"].numpy())

    print("\n  ═══════════════════════════════════════════════")
    print("  Domain Classification Report:")
    print("  ═══════════════════════════════════════════════")
    print(classification_report(
        all_domain_true, all_domain_preds,
        target_names=DOMAIN_LABELS,
        digits=3,
    ))

    print("  ═══════════════════════════════════════════════")
    print("  Query Type Classification Report:")
    print("  ═══════════════════════════════════════════════")
    print(classification_report(
        all_qtype_true, all_qtype_preds,
        target_names=QUERY_TYPE_LABELS,
        digits=3,
    ))

    # Domain confusion matrix
    cm = confusion_matrix(all_domain_true, all_domain_preds)
    print("  Domain Confusion Matrix:")
    print("  " + " " * 14 + "  ".join(f"{d[:4]:>4}" for d in DOMAIN_LABELS))
    for i, row in enumerate(cm):
        print(f"  {DOMAIN_LABELS[i]:<14} " + "  ".join(f"{v:>4}" for v in row))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Export for Deployment
# ═══════════════════════════════════════════════════════════════════════════════


def export_onnx():
    """
    Export the trained model to ONNX format for fast CPU inference.

    ONNX Runtime is significantly faster than PyTorch for inference,
    especially on CPU-only deployment.
    """
    import torch

    device = torch.device("cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "router_model.pt", map_location=device))
    model.eval()

    # Dummy input
    dummy_input_ids = torch.randint(0, 30000, (1, 128)).to(device)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long).to(device)

    onnx_path = MODEL_OUTPUT_DIR / "router_model.onnx"

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["domain_logits", "qtype_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "domain_logits": {0: "batch"},
            "qtype_logits": {0: "batch"},
        },
        opset_version=14,
    )

    logger.info("onnx_exported", path=str(onnx_path))
    print(f"\n  ONNX model exported to: {onnx_path}")
    print(f"  Size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="NyayaMitra Query Router Training")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate saved model only")
    parser.add_argument("--export", action="store_true", help="Export to ONNX after training")
    args = parser.parse_args()

    print()
    print("═" * 60)
    print("  NyayaMitra — Query Router Training")
    print("═" * 60)

    if args.eval_only:
        evaluate()
    else:
        results = train(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
        )

        # Always evaluate after training
        evaluate()

        # Check acceptance criteria
        domain_acc = results["best_domain_acc"]
        print()
        if domain_acc >= 0.85:
            print("  ══════════════════════════════════════════════")
            print(f"  ║  PASS — Domain accuracy {domain_acc:.1%} ≥ 85%     ║")
            print("  ══════════════════════════════════════════════")
        else:
            print("  ══════════════════════════════════════════════")
            print(f"  ║  FAIL — Domain accuracy {domain_acc:.1%} < 85%     ║")
            print("  ══════════════════════════════════════════════")

        if args.export:
            export_onnx()

    print()


if __name__ == "__main__":
    main()