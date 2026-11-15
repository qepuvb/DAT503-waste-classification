# DAT503 ESA 1 – KI-gestützte Abfallklassifikation

Bildklassifikation von Abfall in sechs Klassen (Karton, Glas, Metall, Papier, Plastik, Restmüll) mit dem Datensatz TrashNet und einem vortrainierten ResNet50 (Transfer Learning).

**Einstieg:** [`projektmanagement.md`](projektmanagement.md) (Plan, Rollen, Risiken, Status) und [`protokoll.md`](protokoll.md) (alle Tätigkeiten je Person).

## Ergebnis

| Kennzahl (Test-Split, 384 Bilder) | Wert |
|---|---|
| Accuracy | 85,4 % |
| Macro-F1 | 82,6 % |
| Schwächste Klasse | Restmüll (Recall 54,6 %) |

Details: [`Evaluation/shortsummary_evaluation_report.md`](Evaluation/shortsummary_evaluation_report.md)

## Projektstruktur

| Ordner / Datei | Inhalt | Rolle |
|---|---|---|
| `requirements-engineering/` | Anforderungen, Stakeholder | Requirements Engineering |
| `projektmanagement.md` | Projektplan, Rollen, Kommunikation, Risiken, Monitoring | Projektmanagement |
| `scripts/fetch_data.sh`, `src/prepare_data.py`, `src/augmentation.py`, `data/` | Datensatz laden und prüfen, Split, Augmentation | Daten und Datenqualität |
| `src/training + evaluierung.py`, `Model/` | Training ResNet50, trainiertes Modell | Modell und Training |
| `Evaluation/` | Evaluationsskript, Berichte, Abbildungen | Evaluation, Edge und Bericht |

## Ausführen

Training und Evaluation sind für **Google Colab** geschrieben. Beide Skripte laden den Datensatz selbst herunter (fester Commit, SHA256-Prüfung) und lesen den Split `data/splits/split.csv` direkt aus diesem Repository.

### 1. Pakete

In Colab sind die Pakete bereits vorhanden. Lokal:

```bash
pip install torch torchvision pandas numpy scikit-learn pillow matplotlib tqdm
```

### 2. Daten prüfen (optional, lokal)

```bash
bash scripts/fetch_data.sh      # lädt TrashNet nach data/raw und prüft die SHA256-Summe
python3 src/prepare_data.py     # prüft die Bilder, schreibt den Split
```

### 3. Training

Skript in Colab hochladen und ausführen:

```python
%run "training + evaluierung.py"
```

Einstellungen (im Skript fest hinterlegt): 10 Epochen, Batch-Größe 32, Lernrate 0,001, Seed 42. Ergebnis im Ordner `outputs/`: `best_resnet50_trashnet.pth` (bestes Modell auf dem Validierungs-Split) und `test_results.txt`.

### 4. Evaluation

`Evaluation/evaluate.py` und das Modell `Model/best_resnet50_trashnet.pth` in dasselbe Colab-Verzeichnis legen, dann:

```python
%run evaluate.py
```

Das Modell wird dabei nicht neu trainiert. Ergebnisse im Ordner `evaluation_results/`: Bericht, Kennzahlen (`metrics.json`, `metrics_per_class.csv`), Konfusionsmatrix, Fehlbilder, Grad-CAM.

## Daten und Lizenz

TrashNet von Gary Thung und Mindy Yang, siehe [`data/README.md`](data/README.md). Die Bilder liegen nicht im Repository.

## KI-Einsatz

Für Code und Dokumentation wurden KI-Werkzeuge verwendet. Details stehen je Rolle im Protokoll bzw. in `Evaluation/README`.
