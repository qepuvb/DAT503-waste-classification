# Kurzzusammenfassung der Modellevaluation – TrashNet / ResNet50

**Stand:** 15.11.2026 · **Testumfang:** 384 Bilder, sechs Abfallklassen · **Umgebung:** Google Colab (CPU)  
**Grundlage:** [Vollständiger Evaluationsbericht](../evaluation_report.md)

## 1. Ergebnisse im Überblick

| Kennzahl | Ergebnis | Einordnung für die Abfallsortierung |
| --- | ---: | --- |
| **Accuracy (alle Bilder)** | **85,42 %** (328/384) | Solide Gesamtleistung, verdeckt aber Unterschiede zwischen Materialien. |
| **Macro-F1** | **82,63 %** | Berücksichtigt alle sechs Klassen gleichgewichtet. |
| **Precision / Recall (Macro)** | **85,73 % / 81,34 %** | Wichtig für Fehlzuordnungen und übersehene Materialien. |
| **Unsicherheits-/Ablehnungsrate** | **44,53 %** (171/384) | Fast jedes zweite Bild würde bei der gewählten Regel zurückgewiesen. |
| **Accuracy akzeptierter Fälle** | **98,59 %** (210/213) | Sehr hohe Trefferquote unter den akzeptierten Bildern, allerdings bei reduzierter Abdeckung. |
| **Mittlere Einzelbildlatenz / P95** | **218,21 ms / 238,38 ms** | Colab-CPU-Messung; keine Aussage über die Leistung auf einem Edge-Gerät. |
| **Einzelbilddurchsatz / Batch-Durchsatz** | **4,58 / 3,15 Bilder/s** | Messmethoden sind unterschiedlich; Werte nicht unmittelbar gleichzusetzen. |
| **Checkpointgröße** | **90,03 MiB** | Relevante Größe für eine spätere Bereitstellung. |

**80-%-Schwellenwert:** Der Confidence Threshold wurde **als Nutzerentscheidung** fiktiv festgelegt, nicht auf den Testdaten optimiert. Er markiert Vorhersagen unter 80 % maximaler Softmax-Ausgabe als unsicher; Softmax-Konfidenz ist ohne Kalibrierung keine verlässliche Fehlerwahrscheinlichkeit.

## 2. Auffälligkeiten nach Abfallklasse

| Klasse | Precision | Recall | Interpretation |
| --- | ---: | ---: | --- |
| **cardboard** | 97,83 % | 73,77 % | Selten fälschlich als Karton erkannt, aber viele echte Kartonbilder verfehlt. |
| **glass** | 86,84 % | 86,84 % | Vergleichsweise ausgeglichen. |
| **metal** | 79,73 % | 95,16 % | Erkennt fast alle Metallbilder, erzeugt aber relativ viele falsche Metallzuordnungen. |
| **paper** | 78,18 % | 95,56 % | Erkennt Papier zuverlässig, verwechselt jedoch andere Materialien mit Papier. |
| **plastic** | 96,77 % | 82,19 % | Hohe Präzision, aber ein Teil der Kunststoffbilder wird übersehen. |
| **trash** | 75,00 % | **54,55 %** | **Schwächste Klasse:** Nur etwa jedes zweite entsprechende Bild wird erkannt. |

Die Fehlbilder zeigen insbesondere **Verwechslungen von `cardboard` mit `paper`** sowie von `trash` mit `paper`. Drei falsche Vorhersagen erreichten trotz des 80-%-Schwellenwerts die Akzeptanzgrenze. Bei `trash` wurden **72,73 %** der Bilder als unsicher zurückgewiesen; nur sechs Bilder dieser Klasse wurden akzeptiert.

## 3. Interpretation und Grenzen

**Kernaussage:** Das Modell zeigt eine brauchbare Ausgangsleistung auf dem vorgegebenen Testsplit. Die 80-%-Regel erhöht die Accuracy der automatisch akzeptierten Vorhersagen deutlich, **reduziert deren Anteil jedoch auf 55,47 %**. Für eine Sortieranwendung sind deshalb neben der Gesamt-Accuracy vor allem **Recall, Precision und Ablehnungsrate je Material** entscheidend.

Das **95-%-Bootstrap-Intervall der Accuracy beträgt 81,77–88,80 %** (unter Annahme unabhängiger Testbilder). Grad-CAM und Fehlklassifikationsbilder unterstützen die qualitative Untersuchung, belegen aber keine Ursachen für Fehler oder Hintergrundabhängigkeiten.

**Offen vor produktivem Einsatz:** Verhalten bei unsicheren und unbekannten (OOD-)Bildern festlegen, Robustheit mit veränderten Aufnahmebedingungen prüfen, reale Edge-End-to-End-Latenz messen sowie Shadow Mode und Rückkehrplan definieren. Der bestehende Testsplit wurde nicht neu bewertet; Regression und Zeitreihenanalysen sind für diese Einzelbild-Klassifikation nicht einschlägig.
