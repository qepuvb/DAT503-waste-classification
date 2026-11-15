# Projektprotokoll

DAT503, ESA 1. Jede Person trägt die eigenen Tätigkeiten hier ein.

| Datum | Wer | Aufgabe | Umsetzung | Ergebnis / Problem |
|---|---|---|---|---|
| 11.11.2026 | Jens Mertens | TrashNet beschaffen | `scripts/fetch_data.sh` lädt das resized-Zip von Commit `6fa2b878` und prüft den SHA256. Entpackt wird nach `data/raw`. Zip und Bilder stehen in `.gitignore`. | Quelle ist fixiert auf: Thung & Yang, TrashNet, sechs Klassen. Commit `6471156`. |
| 11.11.2026 | Jens Mertens | Augmentation festlegen | `src/augmentation.py`, nur für den Trainings-Split: Spiegeln mit Wahrscheinlichkeit 0,5, Drehung bis 15° (Ecken weiß), Helligkeit 0,8 bis 1,2. Keine zusätzlichen Dateien, nur eine Vorschau. | Die zufällige Drehung zwischen −15° und +15° ist die Vorgabe. Die Vorschau zeigt nur +15°. Ziehen muss das die Trainingsrolle. Ebenfalls in `6471156`. |
| 12.11.2026 | Jens Mertens | Prüfskript für die Bilder | `src/prepare_data.py` geht die entpackten Bilder durch. | Skript und Quellennotiz liegen im Repo. |
| 13.11.2026 | Jens Mertens | Split festgelegt | `data/splits/split.csv`, Seed 42, stratifiziert 70 % Train, 15 % Val, Rest Test. Die drei bytegleichen Glasflaschen stehen in `data/splits/excluded.csv` und nicht im Split. | 2524 Bilder: Train 1764, Val 376, Test 384. Dieselbe Datei liegt nicht mehr im Training und im Test. |
| 11.11.2026 | Lucas Frank | Rollen verteilt | Abstimmung im Gruppenchat. | Requirements Engineering: Stephanie Lepuschitz; Projektmanagement: Lucas Frank; Daten und Datenqualität: Jens Mertens; Modell und Training: Felix Zopf; Evaluation, Edge und Bericht: Günther Reinwald. |
| 14.11.2026 | Lucas Frank | Projektplan erstellt | `projektmanagement.md`: Meilensteine mit Gantt, Ressourcenzuweisung, Kommunikation, Risiken, Monitoring. | Zeitplan ist knapp, Modell und Evaluation müssen am Sonntag fertig werden. |
| 14.11.2026 | Felix Zopf | Modell auswählen und trainieren | `training + evaluierung.py`: Skript für das Training und die Evaluierung erstellt | Trainiertes Modell und automatische Evaluierung hochgeladen. |
| 15.11.2026 | Günther Reinwald | Modell evaluieren | 1. Vorhandenen Trainingscode `training + evaluierung.py` analysiert und erforderliche Bestandteile identifiziert.<br>2. Download, Testsplit-Auswahl, Dataset-Klasse und Vorverarbeitung übernommen.<br>3. ResNet50-Architektur rekonstruiert und trainierten `.pth`-Checkpoint geladen.<br>4. Eigenständiges `evaluate.py` für Google Colab entwickelt.<br>5. Geeignete Klassifikationsmetriken anhand der Theorie und Akzeptanzkriterien ausgewählt.<br>6. Konfidenzschwelle von 80 % und Auswertung unsicherer Entscheidungen festgelegt.<br>7. Latenz-, FPS- und Modellgrößenmessung implementiert.<br>8. Confusion Matrix, weitere Diagramme, Fehlklassifikationen und Grad-CAM ergänzt.<br>9. Vollständige Evaluation ausgeführt und Markdown-Bericht mit PNG-Abbildungen erzeugt.<br>10. Ergebnisse für GitHub aufbereitet und KI-Nutzung dokumentiert.<br>11. Learnings von Günther dokumentiert in `evaluation/short_protokoll_Reinwald.md`<br>12. Protokoll `protokoll.md` aktualisiert, `Nicht gelöst` aktualisiert| Trainiertes Modell evaluiert und vollständigen Evaluationsbericht mit Abbildungen auf GitHub bereitgestellt. |

## Nicht gelöst

| Punkt | Grund |
|---|---|
| Keine Durchsicht aller Bilder auf falsch einsortierten Inhalt | 2527 Dateien. Heute sind nur Quelle und Augmentation festgehalten, nicht jedes Bild einzeln angesehen. |
| Edge-HW Evaluierung (Reinwald) | Es steht keine dezidierte HW zur Verfügung. Es wurde keine HW simuliert. Die Evaluierung hat Messwerte für Edge-HW festgelegt, diese für die Runtime dokumentiert. |
| Confidence Threshold (Reinwald) | Es wurde ein Confidence Threshold von 80% festgelegt. Der Wert wurde für die Evaluierung entschieden. Es ist zu Prüfen, ob der Wert für die Anwendung sinnvoll ist |

## Weitere Einträge

| Datum | Wer | Aufgabe | Umsetzung | Ergebnis / Problem |
|---|---|---|---|---|
|  |  |  |  |  |
