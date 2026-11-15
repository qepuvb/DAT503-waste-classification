
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TrashNet ResNet50 – Evaluation in Google Colab.
Start: %run evaluate.py
Das Modell wird nicht erneut trainiert.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm


# ============================================================
# KONFIGURATION
# ============================================================

MODEL_PATH = Path("best_resnet50_trashnet.pth")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = DATA_DIR / "dataset-resized.zip"
OUTPUT_DIR = Path("evaluation_results")

DATASET_URL = (
    "https://github.com/garythung/trashnet/raw/"
    "6fa2b878c6c1b4304b91109070ce0edf9279bb31/"
    "data/dataset-resized.zip"
)

EXPECTED_SHA256 = (
    "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
)

SPLIT_CSV_URL = (
    "https://raw.githubusercontent.com/qepuvb/"
    "DAT503-waste-classification/main/data/splits/split.csv"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# USER-ENTSCHEIDUNG:
# Der Confidence Threshold von 80 % wurde vom Nutzer
# fiktiv festgelegt und nicht statistisch optimiert.
# Max. Softmax-Wahrscheinlichkeit < 0.80: unsicher.
# Max. Softmax-Wahrscheinlichkeit >= 0.80: akzeptiert.
# Diese Regel stellt keinen sicheren OOD-Fallback dar.
CONFIDENCE_THRESHOLD = 0.80

BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42
MAX_ERROR_IMAGES = 18
MAX_GRADCAM_IMAGES = 8
LATENCY_SAMPLE_COUNT = 40
LATENCY_WARMUP = 5
BOOTSTRAP_ITERATIONS = 1000
SHOW_PLOTS = True


# ============================================================
# ALLGEMEINE FUNKTIONEN
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percent(value):
    if value is None:
        return "Nicht definiert"
    return f"{100 * value:.2f} %"


def number(value, decimals=2):
    if value is None:
        return "Nicht verfügbar"
    return f"{value:.{decimals}f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        fields = [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines) + "\n\n"


# ============================================================
# DATENSATZ HERUNTERLADEN
# ============================================================

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Lade TrashNet herunter ...")
        try:
            urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        except Exception:
            ZIP_PATH.unlink(missing_ok=True)
            raise

    print("Prüfe SHA256 ...")
    sha256 = hashlib.sha256()

    with ZIP_PATH.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    if sha256.hexdigest() != EXPECTED_SHA256:
        raise ValueError("SHA256-Prüfung fehlgeschlagen.")

    target = RAW_DIR / "dataset-resized"

    if not target.is_dir():
        print("Entpacke Datensatz ...")
        with zipfile.ZipFile(ZIP_PATH) as archive:
            base = RAW_DIR.resolve()
            for member in archive.infolist():
                destination = (
                    RAW_DIR / member.filename
                ).resolve()
                if (
                    destination != base
                    and base not in destination.parents
                ):
                    raise ValueError("Ungültiger ZIP-Pfad")
            archive.extractall(RAW_DIR)

    if not target.is_dir():
        raise FileNotFoundError(
            f"Datensatzverzeichnis fehlt: {target}"
        )

    macos = RAW_DIR / "__MACOSX"
    if macos.exists():
        shutil.rmtree(macos)

    for file in RAW_DIR.rglob(".DS_Store"):
        file.unlink()

    print("Datensatz bereit.")


# ============================================================
# DATASET
# ============================================================

class TrashNetDataset(Dataset):

    def __init__(
        self, dataframe, root_dir, classes, transform
    ):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = self.root_dir / str(row["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        label = self.class_to_idx[row["label"]]

        return image, label, str(path)


def load_test_split():
    df = pd.read_csv(SPLIT_CSV_URL)

    required = {"path", "label", "split"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Fehlende CSV-Spalten: {missing}"
        )

    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("Testsplit ist leer.")

    if test_df[["path", "label"]].isna().any().any():
        raise ValueError(
            "Testsplit enthält ungültige Einträge."
        )

    return test_df


# ============================================================
# TRAINiertes MODELL LADEN
# ============================================================

def load_model(device):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {MODEL_PATH}"
        )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint enthält kein model_state_dict."
        )

    classes = list(checkpoint["classes"])

    mapping = {
        name: idx
        for idx, name in enumerate(classes)
    }

    if (
        "class_to_idx" in checkpoint
        and checkpoint["class_to_idx"] != mapping
    ):
        raise ValueError(
            "Klassenreihenfolge ist inkonsistent."
        )

    # Architektur wiederherstellen.
    # Kein ImageNet-Download, da trainierte Gewichte
    # vollständig im Checkpoint gespeichert sind.
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, classes, checkpoint


# ============================================================
# TEST-INFERENZ
# ============================================================

@torch.inference_mode()
def run_inference(model, loader, device, num_classes):
    y_true = []
    y_pred = []
    probabilities = []
    paths = []
    batch_times = []

    model.eval()

    for images, labels, image_paths in tqdm(
        loader, desc="Test-Inferenz"
    ):
        images = images.to(
            device, non_blocking=True
        )

        synchronize(device)
        start = time.perf_counter()

        logits = model(images)

        synchronize(device)
        elapsed = time.perf_counter() - start

        probs = torch.softmax(logits, dim=1)

        if probs.shape[1] != num_classes:
            raise RuntimeError(
                "Unpassende Anzahl Modellklassen."
            )

        predictions = probs.argmax(dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.cpu().tolist())
        probabilities.append(probs.cpu().numpy())
        paths.extend(image_paths)
        batch_times.append(elapsed)

    return {
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "probabilities": np.concatenate(
            probabilities, axis=0
        ),
        "paths": paths,
        "batch_inference_seconds": sum(batch_times),
        "batch_times": batch_times,
    }


# ============================================================
# EDGE-ORIENTIERTE LATENZMESSUNG AUF COLAB
# ============================================================

@torch.inference_mode()
def measure_latency(model, dataset, device):
    """Einzelbildmessung auf Colab-Hardware.

    Enthalten:
    - Dateiladen
    - Bilddekodierung
    - Resize/Normalisierung
    - Transfer zum Gerät
    - Modellinferenz
    - Rücktransfer der Vorhersage

    Nicht enthalten:
    - Kameraerfassung
    - Netzwerk
    - Aktorik
    - industrielle Gesamtpipeline
    """
    count = min(
        LATENCY_SAMPLE_COUNT, len(dataset)
    )

    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(dataset),
        size=count,
        replace=False,
    ).tolist()

    def execute(index):
        synchronize(device)
        start = time.perf_counter()

        image, _, _ = dataset[index]
        tensor = image.unsqueeze(0).to(device)

        output = model(tensor)
        _ = output.argmax(dim=1).cpu().item()

        synchronize(device)
        return (
            time.perf_counter() - start
        ) * 1000

    for index in indices[:LATENCY_WARMUP]:
        execute(index)

    times = [
        execute(index)
        for index in tqdm(
            indices, desc="Latenzmessung"
        )
    ]

    return np.asarray(times, dtype=float)


# ============================================================
# KLASSIFIKATIONSMETRIKEN
# ============================================================

def calculate_metrics(results, classes):
    true = results["y_true"]
    pred = results["y_pred"]
    probs = results["probabilities"]

    confidence = probs.max(axis=1)

    accepted = (
        confidence >= CONFIDENCE_THRESHOLD
    )

    correct = true == pred
    n = len(true)
    labels = list(range(len(classes)))

    report = classification_report(
        true,
        pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true,
            pred,
            labels=labels,
            zero_division=0,
        )
    )

    class_rows = []

    for idx, name in enumerate(classes):
        class_mask = true == idx
        class_n = int(class_mask.sum())

        class_accepted = (
            class_mask & accepted
        )

        class_rejected = (
            class_mask & ~accepted
        )

        rejected_rate = (
            float(class_rejected.sum() / class_n)
            if class_n else None
        )

        accepted_accuracy = (
            float(correct[class_accepted].mean())
            if class_accepted.any()
            else None
        )

        class_rows.append({
            "Klasse": name,
            "Bilder": class_n,
            "Precision": float(precision[idx]),
            "Recall": float(recall[idx]),
            "F1": float(f1[idx]),
            "Ablehnung": rejected_rate,
            "Akzeptierte": int(
                class_accepted.sum()
            ),
            "Accuracy akzeptiert":
                accepted_accuracy,
        })

    # Bootstrap-Konfidenzintervall
    # Vereinfachende Annahme:
    # Testbilder werden als unabhängig betrachtet.
    rng = np.random.default_rng(SEED)
    bootstrap_values = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(
            0, n, size=n
        )
        bootstrap_values.append(
            float(correct[indices].mean())
        )

    confidence_interval = np.percentile(
        bootstrap_values, [2.5, 97.5]
    ).tolist()

    return {
        "n": n,
        "correct": int(correct.sum()),
        "accuracy": float(
            accuracy_score(true, pred)
        ),
        "macro_precision": float(
            report["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report["macro avg"]["f1-score"]
        ),
        "weighted_f1": float(
            report["weighted avg"]["f1-score"]
        ),
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "rejection_rate": float(
            (~accepted).mean()
        ),
        "accepted_accuracy": (
            float(correct[accepted].mean())
            if accepted.any()
            else None
        ),
        "accepted_correct": int(
            (correct & accepted).sum()
        ),
        "accepted_wrong": int(
            (~correct & accepted).sum()
        ),
        "confidence": confidence,
        "correct_mask": correct,
        "accepted_mask": accepted,
        "class_rows": class_rows,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            true, pred, labels=labels
        ),
        "accuracy_ci": confidence_interval,
    }


# ============================================================
# PLOT-HILFSFUNKTION
# ============================================================

def save_plot(fig, filename):
    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)
    return filename


# ============================================================
# KLASSENVERTEILUNG
# ============================================================

def plot_classes(stats, classes):
    counts = [
        row["Bilder"]
        for row in stats["class_rows"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    bars = ax.bar(classes, counts)
    ax.bar_label(bars)

    ax.set(
        title="Testbilder pro Klasse",
        xlabel="Klasse",
        ylabel="Anzahl Bilder",
    )

    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()

    return save_plot(
        fig, "class_distribution.png"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion(stats, classes):
    cm = stats["confusion_matrix"]

    fig, ax = plt.subplots(
        figsize=(8, 6.5)
    )

    image = ax.imshow(
        cm, cmap="Blues"
    )

    ax.set_xticks(
        range(len(classes)),
        classes,
        rotation=40,
        ha="right",
    )

    ax.set_yticks(
        range(len(classes)),
        classes,
    )

    ax.set(
        xlabel="Vorhergesagte Klasse",
        ylabel="Tatsächliche Klasse",
        title="Confusion Matrix",
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                ),
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    return save_plot(
        fig, "confusion_matrix.png"
    )


# ============================================================
# CONFIDENCE-VERTEILUNG
# ============================================================

def plot_confidence(stats):
    confidence = stats["confidence"]
    correct = stats["correct_mask"]

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    bins = np.linspace(0, 1, 21)

    ax.hist(
        [
            confidence[correct],
            confidence[~correct],
        ],
        bins=bins,
        stacked=True,
        label=["Richtig", "Falsch"],
    )

    ax.axvline(
        CONFIDENCE_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label="80 % – Nutzerentscheidung",
    )

    ax.set(
        xlabel="Maximale Softmax-Wahrscheinlichkeit",
        ylabel="Anzahl Bilder",
        title="Konfidenzverteilung",
        xlim=(0, 1),
    )

    ax.legend()
    fig.tight_layout()

    return save_plot(
        fig, "confidence_distribution.png"
    )


# ============================================================
# ABLEHNUNGSRATE JE KLASSE
# ============================================================

def plot_rejections(stats):
    rows = stats["class_rows"]

    classes = [
        row["Klasse"] for row in rows
    ]

    values = [
        100 * row["Ablehnung"]
        if row["Ablehnung"] is not None
        else 0
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    bars = ax.bar(classes, values)
    ax.bar_label(bars, fmt="%.1f%%")

    ax.set(
        title="Unsichere Fälle pro Klasse",
        xlabel="Tatsächliche Klasse",
        ylabel="Ablehnungsrate (%)",
        ylim=(
            0,
            max(10, max(values) * 1.2),
        ),
    )

    fig.tight_layout()

    return save_plot(
        fig, "rejection_per_class.png"
    )


# ============================================================
# LATENZVERTEILUNG
# ============================================================

def plot_latency(latencies):
    if len(latencies) == 0:
        return None

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.hist(
        latencies,
        bins=min(
            16, max(5, len(latencies) // 2)
        ),
    )

    ax.axvline(
        np.median(latencies),
        linestyle="--",
        label="Median",
    )

    ax.set(
        xlabel="Dauer pro Bild (ms)",
        ylabel="Anzahl Messungen",
        title="Latenz auf Google Colab",
    )

    ax.legend()
    fig.tight_layout()

    return save_plot(
        fig, "latency_distribution.png"
    )


# ============================================================
# FEHLKLASSIFIKATIONEN EXPORTIEREN
# ============================================================

def export_errors(results, stats, classes):
    true = results["y_true"]
    pred = results["y_pred"]
    confidence = stats["confidence"]

    errors = np.flatnonzero(
        true != pred
    )

    # Fehler mit hoher Konfidenz zuerst.
    errors = sorted(
        errors,
        key=lambda i: -float(confidence[i]),
    )[:MAX_ERROR_IMAGES]

    folder = OUTPUT_DIR / "misclassified_images"
    folder.mkdir(parents=True, exist_ok=True)

    items = []

    for number, idx in enumerate(
        errors, start=1
    ):
        source = Path(results["paths"][idx])

        with Image.open(source) as img:
            image = ImageOps.contain(
                img.convert("RGB"),
                (480, 360),
            )

        filename = f"error_{number:02d}.png"
        image.save(folder / filename)

        items.append({
            "path": (
                f"misclassified_images/{filename}"
            ),
            "true": classes[int(true[idx])],
            "predicted": classes[int(pred[idx])],
            "confidence": float(
                confidence[idx]
            ),
            "rejected": bool(
                confidence[idx]
                < CONFIDENCE_THRESHOLD
            ),
        })

    return items


# ============================================================
# GRAD-CAM BERECHNEN
# ============================================================

def gradcam_for_image(
    model, path, transform, device, target_class
):
    captured = {}

    def forward_hook(module, inputs, output):
        captured["activation"] = output

    handle = (
        model.layer4[-1]
        .register_forward_hook(forward_hook)
    )

    try:
        with Image.open(path) as image_file:
            pil_image = image_file.convert("RGB")

        tensor = (
            transform(pil_image)
            .unsqueeze(0)
            .to(device)
        )

        with torch.enable_grad():
            logits = model(tensor)

            activations = captured["activation"]

            gradients = torch.autograd.grad(
                outputs=logits[0, target_class],
                inputs=activations,
                retain_graph=False,
                create_graph=False,
            )[0]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            cam = torch.relu(
                (weights * activations).sum(
                    dim=1, keepdim=True
                )
            )

            cam = (
                torch.nn.functional.interpolate(
                    cam,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            )

            cam = (
                cam[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

        cam -= float(cam.min())

        if cam.max() > 0:
            cam /= float(cam.max())

        original = (
            np.asarray(
                pil_image.resize((224, 224)),
                dtype=np.float32,
            ) / 255.0
        )

        return original, cam

    finally:
        handle.remove()


# ============================================================
# GRAD-CAM GRAFIKEN
# ============================================================

def export_gradcam(
    model, results, stats, classes,
    transform, device
):
    true = results["y_true"]
    pred = results["y_pred"]
    confidence = stats["confidence"]

    correct_indices = np.flatnonzero(
        true == pred
    )
    wrong_indices = np.flatnonzero(
        true != pred
    )

    rng = np.random.default_rng(SEED)

    selected = []
    half = MAX_GRADCAM_IMAGES // 2

    if len(correct_indices):
        selected.extend(
            rng.choice(
                correct_indices,
                size=min(
                    half, len(correct_indices)
                ),
                replace=False,
            ).tolist()
        )

    if len(wrong_indices):
        selected.extend(
            rng.choice(
                wrong_indices,
                size=min(
                    MAX_GRADCAM_IMAGES - len(selected),
                    len(wrong_indices),
                ),
                replace=False,
            ).tolist()
        )

    if len(selected) < MAX_GRADCAM_IMAGES:
        remainder = [
            int(i)
            for i in correct_indices
            if int(i) not in selected
        ]

        selected.extend(
            remainder[
                :MAX_GRADCAM_IMAGES - len(selected)
            ]
        )

    folder = OUTPUT_DIR / "gradcam"
    folder.mkdir(parents=True, exist_ok=True)

    items = []

    for number, idx in enumerate(
        tqdm(selected, desc="Grad-CAM"), start=1
    ):
        original, cam = gradcam_for_image(
            model,
            results["paths"][idx],
            transform,
            device,
            int(pred[idx]),
        )

        fig, axes = plt.subplots(
            1, 3, figsize=(11, 3.8)
        )

        axes[0].imshow(original)
        axes[0].set_title("Original")

        axes[1].imshow(
            cam, cmap="jet",
            vmin=0, vmax=1,
        )
        axes[1].set_title("Grad-CAM")

        axes[2].imshow(original)
        axes[2].imshow(
            cam, cmap="jet",
            alpha=0.45,
            vmin=0, vmax=1,
        )
        axes[2].set_title("Überlagerung")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Tatsächlich: {classes[int(true[idx])]} | "
            f"Vorhersage: {classes[int(pred[idx])]} | "
            f"Konfidenz: {100 * confidence[idx]:.1f} %"
        )

        fig.tight_layout()

        filename = f"gradcam_{number:02d}.png"

        fig.savefig(
            folder / filename,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )

        if SHOW_PLOTS:
            plt.show()

        plt.close(fig)

        items.append({
            "path": f"gradcam/{filename}",
            "true": classes[int(true[idx])],
            "predicted": classes[int(pred[idx])],
            "confidence": float(confidence[idx]),
            "correct": bool(
                true[idx] == pred[idx]
            ),
        })

    return items


# ============================================================
# MARKDOWN-BERICHT
# ============================================================

def build_report(
    stats, hardware, plots,
    errors, cams, classes, checkpoint
):
    lines = []

    def add(text=""):
        lines.append(str(text))

    def table(headers, rows):
        add(markdown_table(headers, rows))

    add("# Evaluationsbericht – TrashNet-Abfallsortierung")
    add()
    add(
        f"*Erstellt am "
        f"{datetime.now().strftime('%d.%m.%Y %H:%M')}*"
    )

    add("## 1. Überblick")
    add(
        "Untersucht wurde ein trainiertes ResNet50-Modell "
        "für die Bildklassifikation von Abfall. "
        "Die Evaluation verwendet ausschließlich den "
        "vorgegebenen Testsplit. Ein erneutes Training "
        "fand nicht statt."
    )

    table(
        ["Eigenschaft", "Wert"],
        [
            ("Modell", "ResNet50"),
            ("Testbilder", stats["n"]),
            ("Klassen", ", ".join(classes)),
            (
                "Testsplit",
                "Unverändert übernommen",
            ),
            (
                "Confidence Threshold",
                "80 % – Nutzerentscheidung",
            ),
            (
                "Messumgebung",
                hardware["device_description"],
            ),
        ],
    )

    add("### Relevanz für die Abfallsortierung")

    table(
        ["Metrik", "Relevanz", "Begründung"],
        [
            (
                "Recall pro Klasse",
                "**Sehr hoch**",
                "Zeigt übersehene Materialien.",
            ),
            (
                "Precision pro Klasse",
                "**Sehr hoch**",
                "Zeigt falsch zugeordnete Materialien.",
            ),
            (
                "Confusion Matrix",
                "**Sehr hoch**",
                "Zeigt Verwechslungen zwischen Klassen.",
            ),
            (
                "Ablehnungsrate",
                "**Sehr hoch**",
                "Zeigt nicht automatisch zuordenbare Bilder.",
            ),
            (
                "Accuracy akzeptierter Bilder",
                "**Sehr hoch**",
                "Beschreibt die Qualität akzeptierter Entscheidungen.",
            ),
            (
                "Latenz und FPS",
                "**Sehr hoch**",
                "Relevant für spätere Edge-Integration.",
            ),
            (
                "F1 und Macro-F1",
                "**Hoch**",
                "Kombinieren Precision und Recall.",
            ),
            (
                "Grad-CAM",
                "**Hoch**",
                "Unterstützt die Analyse von Modellentscheidungen.",
            ),
            (
                "Gesamt-Accuracy",
                "**Mittel**",
                "Kann klassenbezogene Fehler verdecken.",
            ),
            (
                "Bootstrap-Intervall",
                "**Ergänzend**",
                "Beschreibt statistische Unsicherheit.",
            ),
        ],
    )

    add("## 2. Testdatensatz")
    add(
        "Der Testsplit wurde vor dem Training festgelegt "
        "und in dieser Evaluation weder verändert "
        "noch methodisch neu bewertet."
    )
    add(
        f"![Klassenverteilung]({plots['classes']})"
    )

    add("## 3. Klassifikationsmetriken")

    add("### Berechnungsformeln")
    add(
        "**Accuracy:** "
        "`(Anzahl richtiger Vorhersagen) / "
        "(Anzahl aller Bilder)`"
    )
    add(
        "**Precision:** `TP / (TP + FP)`"
    )
    add(
        "**Recall:** `TP / (TP + FN)`"
    )
    add(
        "**F1:** "
        "`2 × Precision × Recall / "
        "(Precision + Recall)`"
    )
    add(
        "**Macro-F1:** "
        "`Summe der F1-Werte aller Klassen / "
        "Anzahl Klassen`"
    )
    add(
        "**Weighted-F1:** "
        "`Summe(F1 je Klasse × Support je Klasse) / "
        "Gesamtzahl Bilder`"
    )
    add(
        "TP bezeichnet richtig positive, FP falsch positive "
        "und FN falsch negative Zuordnungen. "
        "Die Berechnung erfolgt für jede Klasse nach "
        "dem One-vs-Rest-Prinzip."
    )

    table(
        ["Kennzahl", "Ergebnis"],
        [
            (
                "**Accuracy**",
                percent(stats["accuracy"]),
            ),
            (
                "Macro-Precision",
                percent(stats["macro_precision"]),
            ),
            (
                "Macro-Recall",
                percent(stats["macro_recall"]),
            ),
            (
                "**Macro-F1**",
                percent(stats["macro_f1"]),
            ),
            (
                "Weighted-F1",
                percent(stats["weighted_f1"]),
            ),
            (
                "Richtige Vorhersagen",
                f"{stats['correct']} / {stats['n']}",
            ),
        ],
    )

    add("### Metriken pro Klasse")

    table(
        [
            "Klasse", "Bilder",
            "Precision", "Recall", "F1",
        ],
        [
            (
                row["Klasse"],
                row["Bilder"],
                percent(row["Precision"]),
                percent(row["Recall"]),
                percent(row["F1"]),
            )
            for row in stats["class_rows"]
        ],
    )

    add("### Confusion Matrix")
    add(
        "Die Zeilen zeigen die tatsächliche Klasse, "
        "die Spalten die vorhergesagte Klasse. "
        "Einträge außerhalb der Hauptdiagonale "
        "stellen Fehlklassifikationen dar."
    )
    add(
        f"![Confusion Matrix]({plots['confusion']})"
    )

    add("## 4. Unsicherheit und Ablehnungsrate")

    add(
        "**Nutzerentscheidung:** Der Threshold wurde "
        "fiktiv auf 80 % festgelegt. "
        "Es erfolgte keine nachträgliche Optimierung."
    )

    add(
        "**Konfidenz:** "
        "`max(softmax(Modellausgabe))`"
    )
    add(
        "**Ablehnungsrate:** "
        "`Anzahl Vorhersagen mit Konfidenz < 0.80 / "
        "Anzahl aller Bilder`"
    )
    add(
        "**Akzeptanzrate:** "
        "`Anzahl Vorhersagen mit Konfidenz >= 0.80 / "
        "Anzahl aller Bilder`"
    )
    add(
        "**Accuracy akzeptierter Fälle:** "
        "`Richtige akzeptierte Vorhersagen / "
        "Alle akzeptierten Vorhersagen`"
    )
    add(
        "Softmax-Werte sind ohne gesonderte "
        "Kalibrierungsprüfung nicht automatisch "
        "zuverlässige Wahrscheinlichkeiten."
    )

    table(
        ["Kennzahl", "Ergebnis"],
        [
            (
                "**Unsicherheitsrate**",
                percent(stats["rejection_rate"]),
            ),
            (
                "Abgelehnte Bilder",
                stats["rejected"],
            ),
            (
                "Akzeptierte Bilder",
                stats["accepted"],
            ),
            (
                "Akzeptanzrate",
                percent(
                    stats["accepted"] / stats["n"]
                ),
            ),
            (
                "**Accuracy akzeptierter Fälle**",
                percent(stats["accepted_accuracy"]),
            ),
            (
                "Richtige akzeptierte Fälle",
                stats["accepted_correct"],
            ),
            (
                "Falsche akzeptierte Fälle",
                stats["accepted_wrong"],
            ),
        ],
    )

    add(
        f"![Konfidenzverteilung]({plots['confidence']})"
    )

    add("### Ablehnung pro Klasse")
    add(
        "**Formel:** "
        "`Abgelehnte Bilder einer tatsächlichen Klasse / "
        "Alle Testbilder dieser Klasse`"
    )

    table(
        [
            "Klasse", "Ablehnungsrate",
            "Akzeptierte Bilder", "Accuracy akzeptiert",
        ],
        [
            (
                row["Klasse"],
                percent(row["Ablehnung"]),
                row["Akzeptierte"],
                percent(row["Accuracy akzeptiert"]),
            )
            for row in stats["class_rows"]
        ],
    )

    add(
        f"![Ablehnung pro Klasse]({plots['rejections']})"
    )

    add("## 5. Laufzeit und Modellgröße")
    add(
        "**Messumgebung:** "
        + hardware["device_description"]
    )
    add(
        "Alle Messungen erfolgen auf Google Colab "
        "und nicht auf einem tatsächlichen Edge-Gerät."
    )
    add(
        "**Mittlere Latenz:** "
        "`Summe der Einzelbildzeiten / Anzahl Messungen`"
    )
    add(
        "**Median:** Mittlerer Wert der sortierten Messungen."
    )
    add(
        "**P95:** Empirisches 95. Perzentil "
        "der Einzelbildmessungen."
    )
    add(
        "**Rechnerischer Einzelbilddurchsatz:** "
        "`1000 / mittlere Latenz in ms`"
    )
    add(
        "**Batch-FPS:** "
        "`Verarbeitete Bilder / Summe der "
        "Batch-Inferenzzeiten in Sekunden`"
    )
    add(
        "**Modellgröße (MiB):** "
        "`Checkpoint-Dateigröße in Bytes / 1048576`"
    )

    table(
        ["Messgröße", "Messwert"],
        [
            (
                "Mittlere Einzelbildlatenz",
                f"{number(hardware['latency_mean_ms'])} ms",
            ),
            (
                "Median",
                f"{number(hardware['latency_median_ms'])} ms",
            ),
            (
                "P95",
                f"{number(hardware['latency_p95_ms'])} ms",
            ),
            (
                "Maximum",
                f"{number(hardware['latency_max_ms'])} ms",
            ),
            (
                "Anzahl Messungen",
                hardware["latency_count"],
            ),
            (
                "Einzelbild-FPS",
                number(hardware["single_image_fps"]),
            ),
            (
                "Batch-FPS",
                number(hardware["batch_fps"]),
            ),
            (
                "Batchgröße",
                BATCH_SIZE,
            ),
            (
                "Checkpointgröße",
                f"{number(hardware['checkpoint_mib'])} MiB",
            ),
            (
                "Modellparameter",
                hardware["parameter_count"],
            ),
        ],
    )

    add(
        "Die Einzelbildmessung enthält Dateiladen, "
        "Dekodierung, Vorverarbeitung, Gerätetransfer, "
        "Modellinferenz und Rücktransfer der Vorhersage. "
        "Kamera, Netzwerk und Sortiermechanik fehlen."
    )
    add(
        "**Es erfolgt keine Bewertung der "
        "Edge-Tauglichkeit.**"
    )

    if plots["latency"]:
        add(
            f"![Latenzverteilung]({plots['latency']})"
        )

    add("## 6. Fehlklassifikationen")
    add(
        f"Es wurden {stats['n'] - stats['correct']} "
        "Bilder falsch klassifiziert."
    )
    add(
        "Die Beispiele sind nach absteigender "
        "Konfidenz sortiert und stellen keine "
        "repräsentative Zufallsstichprobe dar."
    )

    if errors:
        table(
            [
                "Bild", "Tatsächlich",
                "Vorhersage", "Konfidenz",
                "80-%-Regel",
            ],
            [
                (
                    f"![Fehlerbild]({item['path']})",
                    item["true"],
                    item["predicted"],
                    percent(item["confidence"]),
                    (
                        "Abgelehnt"
                        if item["rejected"]
                        else "Akzeptiert"
                    ),
                )
                for item in errors
            ],
        )
    else:
        add("Keine Fehlklassifikationen gefunden.")

    add("## 7. Grad-CAM")
    add(
        "Grad-CAM visualisiert Bildbereiche, "
        "deren Merkmale den dargestellten "
        "Klassenscore beeinflussen. "
        "Warme Farben zeigen höhere Beiträge."
    )
    add(
        "**Einschränkung:** Grad-CAM ist eine "
        "Interpretationshilfe und liefert keinen "
        "kausalen Nachweis für Shortcut-Learning."
    )

    for item in cams:
        add(
            f"**Tatsächlich:** {item['true']} · "
            f"**Vorhersage:** {item['predicted']} · "
            f"**Konfidenz:** "
            f"{percent(item['confidence'])} · "
            f"**Ergebnis:** "
            f"{'Richtig' if item['correct'] else 'Falsch'}"
        )
        add(
            f"![Grad-CAM]({item['path']})"
        )

    add("## 8. Statistische Unsicherheit")
    add(
        "Für die Accuracy wurde ein Bootstrap "
        f"mit {BOOTSTRAP_ITERATIONS} Wiederholungen "
        "und Ziehen mit Zurücklegen durchgeführt."
    )
    add(
        "**Berechnung:** In jedem Bootstrap-Durchlauf "
        "werden N Testbilder zufällig mit Zurücklegen "
        "gezogen und die Accuracy berechnet. "
        "Die Grenzen entsprechen dem 2,5. und "
        "97,5. Perzentil der Bootstrap-Werte."
    )

    table(
        ["Kennzahl", "Ergebnis"],
        [
            (
                "95-%-Bootstrap-Intervall Accuracy",
                (
                    percent(stats["accuracy_ci"][0])
                    + " bis "
                    + percent(stats["accuracy_ci"][1])
                ),
            ),
        ],
    )

    add(
        "Das Intervall setzt für diese Berechnung "
        "unabhängige Bilder voraus. "
        "Bei gruppenabhängigen Aufnahmen kann "
        "die Unsicherheit unterschätzt werden."
    )

    add("## 9. Grenzen und offene Punkte")

    add(
        "- **Datensplit:** Der vorgegebene Testsplit "
        "wurde unverändert übernommen. "
        "Eine Prüfung der Gruppentrennung "
        "ist nicht Bestandteil dieser Evaluation."
    )
    add(
        "- **OOD-Fallback:** Der 80-%-Threshold "
        "erkennt keine unbekannten Verteilungen "
        "zuverlässig. Eine sichere Standardreaktion "
        "muss noch festgelegt werden."
    )
    add(
        "- **Edge-Gerät:** Laufzeitmessungen liegen "
        "nur für Colab vor. Eine Messung auf dem "
        "Zielgerät und der Vergleich mit einem "
        "Latenzbudget stehen aus."
    )
    add(
        "- **Robustheit:** Die bestehenden Testbilder "
        "erlauben eine klassenbezogene Fehleranalyse. "
        "Robustheit gegenüber Beleuchtungswechsel, "
        "Hintergrundwechsel und anderen Anlagen "
        "kann damit nicht separat nachgewiesen werden."
    )
    add(
        "- **Produktiver Betrieb:** Shadow Mode, "
        "Betriebsüberwachung und Rückkehrplan "
        "sind nicht implementiert."
    )
    add(
        "- **Regression:** MAE, RMSE, MAPE und R² "
        "werden nicht berechnet, weil das Modell "
        "diskrete Klassen und keine kontinuierlichen "
        "Zielgrößen vorhersagt."
    )
    add(
        "- **Zeitreihen/HMM:** Nicht relevant, "
        "weil einzelne Bilder ohne zeitliche "
        "Sequenzstruktur klassifiziert werden."
    )
    add(
        "- **Kreuzvalidierung:** Kein erneutes "
        "Training, kein alternativer Modellkandidat "
        "und kein Modellvergleich."
    )

    add("## 10. Reproduzierbarkeit")

    table(
        ["Parameter", "Wert"],
        [
            ("Checkpoint", MODEL_PATH.name),
            (
                "Trainingsepoche",
                checkpoint.get(
                    "epoch", "Nicht verfügbar"
                ),
            ),
            (
                "Gespeicherte Val-Accuracy",
                checkpoint.get(
                    "val_accuracy", "Nicht verfügbar"
                ),
            ),
            ("Seed", SEED),
            ("Testsplit", SPLIT_CSV_URL),
            ("Datensatz", DATASET_URL),
            ("Eingabegröße", "224 × 224 Pixel"),
            (
                "Normalisierung",
                "ImageNet Mean / Std",
            ),
        ],
    )

    add("*Ende des Evaluationsberichts.*")

    report_path = (
        OUTPUT_DIR / "evaluation_report.md"
    )

    report_path.write_text(
        "\n\n".join(lines),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        device_description = (
            "Google Colab – CUDA / "
            + torch.cuda.get_device_name(0)
        )
    else:
        device_description = "Google Colab – CPU"

    print("Messumgebung:", device_description)

    # Modell laden
    model, classes, checkpoint = load_model(device)

    print("Modell geladen.")
    print("Klassen:", classes)

    # Daten herunterladen
    download_dataset()

    # Vorgegebenen Testsplit laden
    test_df = load_test_split()

    unknown = (
        set(test_df["label"]) - set(classes)
    )

    if unknown:
        raise ValueError(
            f"Unbekannte Testlabels: {unknown}"
        )

    # Gleiche Vorverarbeitung wie beim Training
    # für Validierung/Test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    dataset = TrashNetDataset(
        test_df,
        RAW_DIR,
        classes,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Evaluierung von {len(dataset)} Testbildern."
    )

    # Alle Vorhersagen berechnen
    results = run_inference(
        model, loader, device, len(classes)
    )

    # Klassifikationsmetriken
    stats = calculate_metrics(
        results, classes
    )

    # Hardwaremessungen
    latency_ms = measure_latency(
        model, dataset, device
    )

    batch_seconds = (
        results["batch_inference_seconds"]
    )

    hardware = {
        "device_description": device_description,
        "device_type": device.type,
        "latency_mean_ms": (
            float(np.mean(latency_ms))
            if len(latency_ms) else None
        ),
        "latency_median_ms": (
            float(np.median(latency_ms))
            if len(latency_ms) else None
        ),
        "latency_p95_ms": (
            float(np.percentile(latency_ms, 95))
            if len(latency_ms) else None
        ),
        "latency_max_ms": (
            float(np.max(latency_ms))
            if len(latency_ms) else None
        ),
        "latency_count": len(latency_ms),
        "single_image_fps": (
            float(1000 / np.mean(latency_ms))
            if len(latency_ms) else None
        ),
        "batch_fps": (
            float(stats["n"] / batch_seconds)
            if batch_seconds > 0 else None
        ),
        "checkpoint_mib": (
            MODEL_PATH.stat().st_size
            / (1024 ** 2)
        ),
        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),
        "batch_inference_seconds": batch_seconds,
    }

    # Diagramme erstellen
    plots = {
        "classes": plot_classes(
            stats, classes
        ),
        "confusion": plot_confusion(
            stats, classes
        ),
        "confidence": plot_confidence(stats),
        "rejections": plot_rejections(stats),
        "latency": plot_latency(latency_ms),
    }

    # Fehlklassifikationen exportieren
    errors = export_errors(
        results, stats, classes
    )

    # Grad-CAM erstellen
    cams = export_gradcam(
        model,
        results,
        stats,
        classes,
        transform,
        device,
    )

    # Markdown-Bericht erzeugen
    report_path = build_report(
        stats,
        hardware,
        plots,
        errors,
        cams,
        classes,
        checkpoint,
    )

    # JSON mit Metriken
    metrics_json = {
        "accuracy": stats["accuracy"],
        "macro_precision": stats["macro_precision"],
        "macro_recall": stats["macro_recall"],
        "macro_f1": stats["macro_f1"],
        "weighted_f1": stats["weighted_f1"],
        "confidence_threshold_user_decision":
            CONFIDENCE_THRESHOLD,
        "rejection_rate": stats["rejection_rate"],
        "accepted_accuracy":
            stats["accepted_accuracy"],
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "accuracy_bootstrap_ci_95":
            stats["accuracy_ci"],
        "per_class": stats["class_rows"],
        "colab_hardware": hardware,
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            metrics_json,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Klassenmetriken als CSV
    pd.DataFrame(
        stats["class_rows"]
    ).to_csv(
        OUTPUT_DIR / "metrics_per_class.csv",
        index=False,
        encoding="utf-8",
    )

    # Alle Vorhersagen als CSV
    pd.DataFrame({
        "path": results["paths"],
        "true_class": [
            classes[i]
            for i in results["y_true"]
        ],
        "predicted_class": [
            classes[i]
            for i in results["y_pred"]
        ],
        "confidence": stats["confidence"],
        "accepted_at_80pct":
            stats["accepted_mask"],
        "correct": stats["correct_mask"],
    }).to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
        encoding="utf-8",
    )

    # ZIP mit allen Ergebnissen
    zip_path = shutil.make_archive(
        str(OUTPUT_DIR),
        "zip",
        root_dir=OUTPUT_DIR,
    )

    print("\n=== EVALUATION ABGESCHLOSSEN ===")

    print(
        "Accuracy:",
        percent(stats["accuracy"]),
    )

    print(
        "Macro-F1:",
        percent(stats["macro_f1"]),
    )

    print(
        "Ablehnungsrate:",
        percent(stats["rejection_rate"]),
    )

    print(
        "Accuracy akzeptierter Fälle:",
        percent(stats["accepted_accuracy"]),
    )

    print("Markdown:", report_path.resolve())
    print("ZIP:", Path(zip_path).resolve())

    # Markdown-Vorschau in Google Colab
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display

        if get_ipython() is not None:
            print("\nMarkdown-Bericht:")

            preview = report_path.read_text(
                encoding="utf-8"
            )

            display(Markdown(preview))

    except ImportError:
        pass


if __name__ == "__main__":
    main()