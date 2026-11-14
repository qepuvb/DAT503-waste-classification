# Projektprotokoll

DAT503, ESA 1. Jede Person trägt die eigenen Tätigkeiten hier ein.

| Datum | Wer | Aufgabe | Umsetzung | Ergebnis / Problem |
|---|---|---|---|---|
| 11.11.2026 | Jens Mertens | TrashNet beschaffen | `scripts/fetch_data.sh` lädt das resized-Zip von Commit `6fa2b878` und prüft den SHA256. Entpackt wird nach `data/raw`. Zip und Bilder stehen in `.gitignore`. | Quelle ist fixiert auf: Thung & Yang, TrashNet, sechs Klassen. Commit `6471156`. |
| 11.11.2026 | Jens Mertens | Augmentation festlegen | `src/augmentation.py`, nur für den Trainings-Split: Spiegeln mit Wahrscheinlichkeit 0,5, Drehung bis 15° (Ecken weiß), Helligkeit 0,8 bis 1,2. Keine zusätzlichen Dateien, nur eine Vorschau. | Die zufällige Drehung zwischen −15° und +15° ist die Vorgabe. Die Vorschau zeigt nur +15°. Ziehen muss das die Trainingsrolle. Ebenfalls in `6471156`. |
| 12.11.2026 | Jens Mertens | Prüfskript für die Bilder | `src/prepare_data.py` geht die entpackten Bilder durch. | Skript und Quellennotiz liegen im Repo. |
| 13.11.2026 | Jens Mertens | Split festgelegt | `data/splits/split.csv`, Seed 42, stratifiziert 70 % Train, 15 % Val, Rest Test. Die drei bytegleichen Glasflaschen stehen in `data/splits/excluded.csv` und nicht im Split. | 2524 Bilder: Train 1764, Val 376, Test 384. Dieselbe Datei liegt nicht mehr im Training und im Test. |
| 14.11.2026 | Lucas Frank | Rollen verteilt | Abstimmung im Gruppenchat. | Requirements Engineering: Stephanie Lepuschitz; Projektmanagement: Lucas Frank; Daten und Datenqualität: Jens Mertens; Modell und Training: Felix Zopf; Evaluation, Edge und Bericht: Günther Reinwald. |
| 14.11.2026 | Lucas Frank | Projektplan erstellt | `projektmanagement.md`: Meilensteine mit Gantt, Ressourcenzuweisung, Kommunikation, Risiken, Monitoring. | Zeitplan ist knapp, Modell und Evaluation müssen am Sonntag fertig werden. |

## Nicht gelöst

| Punkt | Grund |
|---|---|
| Keine Durchsicht aller Bilder auf falsch einsortierten Inhalt | 2527 Dateien. Heute sind nur Quelle und Augmentation festgehalten, nicht jedes Bild einzeln angesehen. |

## Weitere Einträge

| Datum | Wer | Aufgabe | Umsetzung | Ergebnis / Problem |
|---|---|---|---|---|
|  |  |  |  |  |
