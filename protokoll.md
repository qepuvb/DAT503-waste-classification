# Projektprotokoll

DAT503, ESA 1. Jede Person trägt die eigenen Tätigkeiten hier ein.

| Datum | Wer | Aufgabe | Umsetzung | Ergebnis / Problem |
|---|---|---|---|---|
| 11.11.2026 | Jens Mertens | TrashNet beschaffen | `scripts/fetch_data.sh` lädt das resized-Zip von Commit `6fa2b878` und prüft den SHA256. Entpackt wird nach `data/raw`. Zip und Bilder stehen in `.gitignore`. | Quelle ist fixiert auf: Thung & Yang, TrashNet, sechs Klassen. Commit `6471156`. |
| 11.11.2026 | Jens Mertens | Augmentation festlegen | `src/augmentation.py`, nur für den Trainings-Split: Spiegeln mit Wahrscheinlichkeit 0,5, Drehung bis 15° (Ecken weiß), Helligkeit 0,8 bis 1,2. Keine zusätzlichen Dateien, nur eine Vorschau. | Die zufällige Drehung zwischen −15° und +15° ist die Vorgabe. Die Vorschau zeigt nur +15°. Ziehen muss das die Trainingsrolle. Ebenfalls in `6471156`. |

## Nicht gelöst

| Punkt | Grund |
|---|---|
| Keine Durchsicht aller Bilder auf falsch einsortierten Inhalt | 2527 Dateien. Heute sind nur Quelle und Augmentation festgehalten, nicht jedes Bild einzeln angesehen. |

## Weitere Einträge

| Datum | Wer | Aufgabe | Umsetzung | Ergebnis / Problem |
|---|---|---|---|---|
|  |  |  |  |  |
