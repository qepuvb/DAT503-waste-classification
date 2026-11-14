# Requirements Engineering – KI-gestützte Abfallklassifikation

## 1. Bedarf und Projektumfang

Bei der manuellen Abfalltrennung können Gegenstände falsch zugeordnet werden. Im Projekt untersuchen wir, ob ein KI-Modell anhand eines Fotos erkennen kann, zu welcher Abfallklasse der abgebildete Gegenstand gehört. Dafür verwenden wir die sechs Klassen des TrashNet-Datensatzes.

Ziel ist ein Prototyp zur Bildklassifikation, keine fertige Sortieranlage. Wegen des kurzen Zeitrahmens ist die Umsetzung auf ein Modell begrenzt. Der Einsatz auf einem Edge-Gerät kann im Bericht betrachtet werden, ist aber keine Pflicht für die Umsetzung. Eine Benutzeroberfläche und die Ansteuerung eines Roboterarms liegen außerhalb des Projektumfangs.

## 2. Funktionale Anforderungen

| ID | Anforderung | Wie wird sie geprüft? |
| --- | --- | --- |
| FA-01 | Der Prototyp muss Abfallbilder aus dem TrashNet-Datensatz als Eingabe verarbeiten können. | Das Skript verarbeitet Bilder aus dem festgelegten Testsplit. |
| FA-02 | Das Modell muss jedem verarbeiteten Bild genau eine der sechs TrashNet-Klassen zuordnen. | Für jedes ausgewertete Bild liegt eine Vorhersage vor, die einer gültigen Klasse entspricht. |
| FA-03 | Die Auswertung muss die vorhergesagten Klassen mit den hinterlegten Klassen der Testbilder vergleichen. | Das Auswertungsskript verwendet die hinterlegten Klassen als Referenz für die Berechnung der Kennzahlen. |

## 3. Nicht-funktionale Anforderungen

| ID | Anforderung | Wie wird sie geprüft? |
| --- | --- | --- |
| NFA-01 | Die Modellqualität muss insgesamt und je Klasse nachvollziehbar sein. | Die Auswertung dokumentiert den Anteil richtiger Vorhersagen (Accuracy) und zeigt die Ergebnisse je Klasse, z. B. in einer Konfusionsmatrix. |
| NFA-02 | Training und Auswertung müssen für andere Teammitglieder wiederholbar sein. | Datenquelle, fester Seed, [Datensplit](../data/splits/split.csv), Paketversionen und Aufruf der Skripte sind im Repository dokumentiert. |
| NFA-03 | Die Bewertung muss auf Bildern erfolgen, die nicht für das Training verwendet wurden. | Die Auswertung nutzt den festgelegten Testsplit. Die dokumentierte Augmentation wird nur auf Trainingsbilder angewendet. |

Eine Mindestgenauigkeit ist in diesem Entwurf nicht festgelegt. Die erzielten Ergebnisse werden berichtet, auch wenn einzelne Klassen schlecht erkannt werden.

## 4. Technologievergleich

| Kriterium | Vortrainiertes ResNet18 anpassen (Transfer Learning) | Kleines CNN ohne Vortraining |
| --- | --- | --- |
| Eignung für TrashNet | Bereits gelernte Bildmerkmale können für die sechs Klassen weiterverwendet werden. | Das Modell muss die Bildmerkmale aus den verfügbaren Trainingsbildern selbst lernen. |
| Einrichtung und Bedienbarkeit | Die vortrainierten Gewichte werden geladen und die Ausgabeschicht wird auf sechs Klassen angepasst. | Die Modellarchitektur muss selbst festgelegt werden; vortrainierte Gewichte sind nicht nötig. |
| Rechenaufwand und Kosten | Colab mit GPU ist laut Projektplan für das Training vorgesehen. Laufzeit und mögliche Kosten wurden nicht erhoben. | Der Aufwand hängt von der gewählten Architektur und den Trainingsläufen ab. Laufzeit und mögliche Kosten wurden nicht erhoben. |
| Erweiterbarkeit | Für zusätzliche Klassen müsste das Modell angepasst und neu trainiert werden. | Auch hier wären eine Anpassung und ein erneutes Training nötig. |

**Vorgeschlagene Wahl:** ResNet18 mit Transfer Learning passt zum knappen Zeitrahmen, weil die Gruppe auf einem vortrainierten Modell aufbauen kann. Das kleine CNN ist eine Alternative für den Technologievergleich und wird im Rahmen dieses Prototyps nicht zusätzlich umgesetzt. Der Vergleich ist qualitativ und beruht nicht auf gemessenen Trainingsläufen beider Ansätze.

## 5. Abstimmung und offene Punkte

Die Anforderungen werden im Gruppenchat mit den Rollen Modell und Training sowie Evaluation, Edge und Bericht abgeglichen. Offen ist, ob das Trainingsskript tatsächlich ResNet18 verwendet und welche Kennzahlen die Auswertung bereits ausgibt. Abweichungen zwischen Anforderungen und Umsetzung sowie nicht durchgeführte Prüfungen werden im [Projektprotokoll](../protokoll.md) festgehalten.
