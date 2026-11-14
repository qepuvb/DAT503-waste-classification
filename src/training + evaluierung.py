#!/usr/bin/env python3
# Fetch data
import os
import hashlib
import urllib.request
import zipfile
from pathlib import Path

# Arbeitsverzeichnis festlegen
os.chdir(Path.cwd())

# Verzeichnisse erstellen
data_dir = Path("data")
raw_dir = data_dir / "raw"

raw_dir.mkdir(parents=True, exist_ok=True)

# Download-URL
url = (
    "https://github.com/garythung/trashnet/raw/"
    "6fa2b878c6c1b4304b91109070ce0edf9279bb31/"
    "data/dataset-resized.zip"
)

zip_path = data_dir / "dataset-resized.zip"

# SHA256-Prüfsumme
expected_hash = (
    "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
)

# Datei herunterladen, falls sie noch nicht existiert
if not zip_path.exists():
    print("Downloading TrashNet dataset...")

    urllib.request.urlretrieve(url, zip_path)

# SHA256 überprüfen
print("Checking SHA256 checksum...")

sha256 = hashlib.sha256()

with open(zip_path, "rb") as file:
    for chunk in iter(lambda: file.read(8192), b""):
        sha256.update(chunk)

actual_hash = sha256.hexdigest()

if actual_hash != expected_hash:
    raise ValueError("SHA256 checksum verification failed!")

print("SHA256 checksum verified!")

# ZIP-Datei entpacken
print("Extracting dataset...")

with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extractall(raw_dir)

# __MACOSX-Verzeichnis entfernen
macosx_dir = raw_dir / "__MACOSX"

if macosx_dir.exists():
    import shutil
    shutil.rmtree(macosx_dir)

# Alle .DS_Store-Dateien entfernen
for file in raw_dir.rglob(".DS_Store"):
    file.unlink()

print("Dataset successfully downloaded and extracted!")

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torch
from tqdm import tqdm

SPLIT_CSV_URL = (
    "https://raw.githubusercontent.com/qepuvb/"
    "DAT503-waste-classification/main/data/splits/split.csv"
)
EXCLUDED_CSV_URL = (
    "https://raw.githubusercontent.com/qepuvb/"
    "DAT503-waste-classification/main/data/splits/excluded.csv"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrashNetDataset(Dataset):
    """Loads images using paths and labels from the repository split CSV."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: Path,
        classes: list[str],
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = self.root_dir / row["path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Bild nicht gefunden: {image_path}\n"
                "Prüfe --data-dir und ob dataset-resized korrekt entpackt wurde."
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]
        return image, label


def load_split_data(data_dir: Path):
    print("Lade Split-CSV-Dateien ...")
    split_df = pd.read_csv(SPLIT_CSV_URL)
    excluded_df = pd.read_csv(EXCLUDED_CSV_URL)

    required_columns = {"path", "label", "split"}
    missing = required_columns - set(split_df.columns)
    if missing:
        raise ValueError(f"split.csv fehlen Spalten: {sorted(missing)}")

    valid_splits = {"train", "val", "test"}
    found_splits = set(split_df["split"].dropna().unique())
    unknown_splits = found_splits - valid_splits
    if unknown_splits:
        raise ValueError(f"Unbekannte Split-Bezeichnungen: {unknown_splits}")

    classes = sorted(split_df["label"].dropna().unique().tolist())
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Mindestens ein Split ist leer. Erwartet werden train, val und test."
        )

    # Die ausgeschlossenen Bilder werden absichtlich nicht trainiert oder evaluiert.
    print(f"Klassen: {classes}")
    print(
        f"Anzahl Bilder: train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )
    print(f"Ausgeschlossene Einträge (nicht verwendet): {len(excluded_df)}")

    return train_df, val_df, test_df, classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(loader, desc="Evaluation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size
        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

    return (
        running_loss / total,
        correct / total,
        all_labels,
        all_predictions,
    )


def build_model(num_classes: int, device: torch.device):
    # ImageNet-vortrainierter ResNet50-Backbone
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except AttributeError:
        # Fallback für ältere torchvision-Versionen
        model = models.resnet50(pretrained=True)

    # Zunächst alle vortrainierten Schichten einfrieren.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Nur den Klassifikationskopf trainieren.
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="TrashNet-Klassifikation mit ResNet50"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Ordner, der dataset-resized enthält (Standard: data/raw)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs")
    )
    # Leerliste übergeben, damit Colabs eigene Parameter ignoriert werden
    args = parser.parse_args(args=[])

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwendetes Gerät: {device}")

    # Die CSV-Pfade sind relativ zu data_dir, z.B.
    # dataset-resized/cardboard/cardboard101.jpg
    image_root = args.data_dir
    expected_image_dir = image_root / "dataset-resized"
    if not expected_image_dir.is_dir():
        raise FileNotFoundError(
            f"Bildordner nicht gefunden: {expected_image_dir}\n"
            "Entpacke den TrashNet-Datensatz so, dass dieser Ordner existiert."
        )

    train_df, val_df, test_df, classes = load_split_data(args.data_dir)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Wichtig: Alle drei Datasets erhalten dieselbe Klassenreihenfolge.
    train_dataset = TrashNetDataset(
        train_df, image_root, classes, transform=train_transform
    )
    val_dataset = TrashNetDataset(
        val_df, image_root, classes, transform=eval_transform
    )
    test_dataset = TrashNetDataset(
        test_df, image_root, classes, transform=eval_transform
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, **loader_kwargs
    )

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.fc.parameters(), lr=args.learning_rate
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_resnet50_trashnet.pth"

    best_val_accuracy = -1.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train | Loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Val   | Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "epoch": epoch + 1,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )
            print(f"Bestes Modell gespeichert: {checkpoint_path}")

    # Bestes Modell anhand der Validierungsgenauigkeit laden.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nEvaluation auf dem unabhängigen Test-Split ...")
    test_loss, test_accuracy, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))))

    # Zusätzliche kompakte Metadaten für die spätere Verwendung sichern.
    results_path = args.output_dir / "test_results.txt"
    with results_path.open("w", encoding="utf-8") as file:
        file.write(f"Test Loss: {test_loss:.6f}\n")
        file.write(f"Test Accuracy: {test_accuracy:.6f}\n")
        file.write("\n")
        file.write(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(len(classes))),
                target_names=classes,
                zero_division=0,
            )
        )

    print(f"\nErgebnisse gespeichert unter: {results_path}")
    print(f"Checkpoint gespeichert unter: {checkpoint_path}")

if __name__ == "__main__":
    main()

print ("Done")