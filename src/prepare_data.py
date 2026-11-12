#!/usr/bin/env python3

import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augmentation import POLICY, preview_variants

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dataset-resized"
SPLIT_CSV = ROOT / "data" / "splits" / "split.csv"
RESULTS = ROOT / "results"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EXPECTED = {
    "cardboard": 403,
    "glass": 501,
    "metal": 410,
    "paper": 594,
    "plastic": 482,
    "trash": 137,
}
CLASS_DE = {
    "cardboard": "Karton",
    "glass": "Glas",
    "metal": "Metall",
    "paper": "Papier",
    "plastic": "Plastik",
    "trash": "Restmüll",
}
EXCLUDE = {
    "dataset-resized/metal/metal91.jpg": "bytegleich mit glass/glass115.jpg, Motiv ist Glas",
    "dataset-resized/plastic/plastic152.jpg": "bytegleich mit glass/glass176.jpg, Motiv ist Glas",
    "dataset-resized/plastic/plastic332.jpg": "bytegleich mit glass/glass389.jpg, Motiv ist Glas",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def list_images():
    if not RAW.is_dir():
        raise SystemExit(f"Rohdaten fehlen: {RAW}\nZuerst: bash scripts/fetch_data.sh")
    paths = []
    for class_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXT:
                continue
            paths.append(path)
    return paths


def dhash(image, size=8):
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    width = size + 1
    bits = []
    for row in range(size):
        for col in range(size):
            left = pixels[row * width + col]
            right = pixels[row * width + col + 1]
            bits.append("1" if left > right else "0")
    return f"{int(''.join(bits), 2):0{size * size // 4}x}"


def inspect(paths):
    rows = []
    for path in paths:
        record = {
            "path": path.relative_to(RAW.parent).as_posix(),
            "label": path.parent.name,
            "bytes": path.stat().st_size,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "width": "",
            "height": "",
            "mode": "",
            "dhash": "",
            "error": "",
        }
        try:
            with Image.open(path) as image:
                image.load()
                record["width"] = image.width
                record["height"] = image.height
                record["mode"] = image.mode
                record["dhash"] = dhash(image)
        except (UnidentifiedImageError, OSError) as exc:
            record["error"] = type(exc).__name__
        rows.append(record)
    return rows


def assign_split(rows):
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda row: row["path"])
        rng.shuffle(group)
        count = len(group)
        n_train = int(count * TRAIN_RATIO)
        n_val = int(count * VAL_RATIO)
        for index, row in enumerate(group):
            if index < n_train:
                row["split"] = "train"
            elif index < n_train + n_val:
                row["split"] = "val"
            else:
                row["split"] = "test"


def write_excluded(rows):
    path = ROOT / "data" / "splits" / "excluded.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "would_have_been", "md5", "reason"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["path"]):
            writer.writerow(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "would_have_been": row["split"],
                    "md5": row["md5"],
                    "reason": row["reason"],
                }
            )


def write_split(rows):
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "split", "width", "height", "bytes", "md5", "dhash"]
    ordered = sorted(rows, key=lambda row: (row["split"], row["label"], row["path"]))
    with SPLIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def plot_counts(rows):
    labels = list(EXPECTED)
    splits = ["train", "val", "test"]
    counts = {
        split: [sum(1 for row in rows if row["label"] == label and row["split"] == split) for label in labels]
        for split in splits
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(labels))
    bottom = [0] * len(labels)
    for split in splits:
        ax.bar(x, counts[split], bottom=bottom, label=split)
        bottom = [a + b for a, b in zip(bottom, counts[split])]
    ax.set_xticks(list(x), [CLASS_DE[label] for label in labels])
    ax.set_ylabel("Bilder")
    ax.set_title("TrashNet, stratifizierter Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "class_distribution.png", dpi=120)
    plt.close(fig)


def plot_examples(rows):
    labels = list(EXPECTED)
    fig, axes = plt.subplots(2, len(labels), figsize=(12, 4.6))
    for col, label in enumerate(labels):
        chosen = [row for row in rows if row["label"] == label and not row["error"]][:2]
        for row_index in range(2):
            ax = axes[row_index][col]
            ax.axis("off")
            if row_index >= len(chosen):
                continue
            with Image.open(RAW.parent / chosen[row_index]["path"]) as image:
                ax.imshow(image.convert("RGB"))
            if row_index == 0:
                ax.set_title(CLASS_DE[label])
    fig.tight_layout()
    fig.savefig(RESULTS / "examples.png", dpi=110)
    plt.close(fig)


def plot_augmentation(rows):
    train = [row for row in rows if row["split"] == "train" and not row["error"]]
    train.sort(key=lambda row: row["path"])
    rng = random.Random(SEED)
    picked = []
    for label in EXPECTED:
        pool = [row for row in train if row["label"] == label]
        if pool:
            picked.append(rng.choice(pool))
    fig, axes = plt.subplots(len(picked), 5, figsize=(12, 10))
    for r, row in enumerate(picked):
        with Image.open(RAW.parent / row["path"]) as image:
            rendered = preview_variants(image)
        for c, (name, variant) in enumerate(rendered):
            ax = axes[r][c]
            ax.imshow(variant)
            ax.axis("off")
            if c == 0:
                ax.set_title(f"{CLASS_DE[row['label']]}\n{name}", fontsize=9)
            elif r == 0:
                ax.set_title(name, fontsize=9)
    fig.suptitle("Augmentation, nur Vorschau, nicht in den Split geschrieben")
    fig.tight_layout()
    fig.savefig(RESULTS / "augmentation_preview.png", dpi=110)
    plt.close(fig)


def markdown_table(headers, body):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(kept, dropped):
    rows = kept + dropped
    on_disk = Counter(row["label"] for row in rows)
    by_split = Counter((row["label"], row["split"]) for row in kept)
    sizes = Counter((row["width"], row["height"]) for row in rows if not row["error"])
    broken = [row for row in rows if row["error"]]
    exact = defaultdict(list)
    perceptual = defaultdict(list)
    for row in rows:
        exact[row["md5"]].append(row["path"])
        if row["dhash"]:
            perceptual[row["dhash"]].append(row["path"])
    exact_dups = [paths for paths in exact.values() if len(paths) > 1]
    hash_dups = [paths for paths in perceptual.values() if len(paths) > 1]

    count_rows = []
    for label, expected in EXPECTED.items():
        count_rows.append(
            [
                CLASS_DE[label],
                label,
                expected,
                on_disk[label],
                by_split[(label, "train")] + by_split[(label, "val")] + by_split[(label, "test")],
                by_split[(label, "train")],
                by_split[(label, "val")],
                by_split[(label, "test")],
            ]
        )
    size_rows = [[f"{w}×{h}", n] for (w, h), n in sizes.most_common()]
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    mismatch = on_disk != Counter(EXPECTED)

    parts = [
        "# Datenqualität TrashNet",
        "",
        f"Erzeugt am {now} durch `src/prepare_data.py`. Seed {SEED}, Anteile Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test Rest.",
        "",
        "## Bestand",
        "",
        markdown_table(
            ["Klasse", "Ordner", "Soll laut README", "Dateien", "im Split", "Train", "Val", "Test"],
            count_rows,
        ),
        "",
        f"Dateien auf der Platte: {len(rows)}. Davon im Split: {len(kept)}. Ausgeschlossen: {len(dropped)}.",
        "",
        "## Technik",
        "",
        markdown_table(["Größe", "Anzahl"], size_rows or [["keine lesbaren Bilder", 0]]),
        "",
        f"Nicht lesbar: {len(broken)}.",
        "",
    ]
    if broken:
        parts.append("\n".join(f"- `{row['path']}` ({row['error']})" for row in broken))
        parts.append("")
    parts += [
        "## Dubletten",
        "",
        f"Exakt gleicher Dateiinhalt (MD5): {len(exact_dups)} Gruppen. In allen dreien liegt dieselbe Datei in `glass` und noch einmal unter `metal` oder `plastic`.",
        "Die falsch einsortierte Kopie ist aus dem Split genommen, die Datei bleibt unter `data/raw`. Liste: `data/splits/excluded.csv`.",
        "Ohne das stünde `glass115` im Test und die bytegleiche `metal91` im Training.",
        "",
        f"Gleicher Differenz-Hash (8×8): {len(hash_dups)} Gruppen. Das ist ein Hinweis, kein Beweis.",
        "Drei Paare sind nach Sichtung dasselbe Foto in derselben Klasse und lagen beide schon im Training: `metal394`/`metal76`, `paper147`/`paper99`, `paper185`/`paper390`. Die bleiben drin.",
        "`cardboard83`, `paper110` und `paper385` teilen sich den Hash `ffffffffffffffff`, sind aber verschiedene Motive. Die bleiben ebenfalls drin.",
        "",
    ]
    if exact_dups:
        parts.append("Exakte Gruppen:")
        parts.append("")
        for group in exact_dups:
            parts.append("- " + ", ".join(f"`{path}`" for path in sorted(group)))
        parts.append("")
    angle = POLICY["rotation_degrees"]
    parts += [
        "## Augmentation",
        "",
        "Gilt nur für den Trainings-Split und wird beim Training gezogen, nicht als zusätzliche Dateien gespeichert.",
        "",
        f"- horizontales Spiegeln, Wahrscheinlichkeit {POLICY['horizontal_flip']}",
        f"- Drehung gleichverteilt von -{angle}° bis +{angle}°, Füllfarbe Weiß. Die Vorschau zeigt +{angle}°.",
        f"- Helligkeit im Faktor {POLICY['brightness_factor'][0]} bis {POLICY['brightness_factor'][1]}",
        "",
        "Val und Test bleiben unverändert. Eine Vorschau liegt in `results/augmentation_preview.png`.",
        "",
        "## Grenze",
        "",
        "Geprüft ist die ganze Menge automatisch: Zählung, Lesbarkeit, Abmessung, MD5, Differenz-Hash.",
        "Angeschaut sind die exakten Dubletten und die Hash-Kollisionen, nicht jedes der 2527 Bilder.",
        "Die Beispieltafel `results/examples.png` zeigt pro Klasse nur zwei Bilder.",
        "",
    ]
    if mismatch:
        parts.append("Die Klassengrößen weichen vom Upstream-README ab. Der Split wurde trotzdem geschrieben.")
        parts.append("")
    (RESULTS / "data_quality.md").write_text("\n".join(parts), encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = inspect(list_images())
    if not rows:
        raise SystemExit(f"Keine Bilder unter {RAW}")
    assign_split(rows)
    kept = []
    dropped = []
    for row in rows:
        reason = EXCLUDE.get(row["path"])
        if reason:
            row["reason"] = reason
            dropped.append(row)
        else:
            kept.append(row)
    write_split(kept)
    write_excluded(dropped)
    plot_counts(kept)
    plot_examples(kept)
    plot_augmentation(kept)
    write_report(kept, dropped)
    print(f"{len(rows)} Dateien, {len(kept)} im Split, {len(dropped)} ausgeschlossen: {SPLIT_CSV}")


if __name__ == "__main__":
    main()
