# Stakeholder und Bedarfsanalyse – KI-gestützte Abfallklassifikation

**Hinweis zur KI-Nutzung:** ChatGPT (GPT-6) wurde zur sprachlichen Überarbeitung eines Entwurfs für die Stakeholder Analyse verwendet. Die Unterstützung umfasste auch den Vorschlag für die Trennung zwischen Stakeholdern der Gruppenarbeit und Stakeholdern bei einem realen Einsatz.

## 1. Ausgangslage und Bedarf

Bei der manuellen Abfallsortierung können Gegenstände falsch zugeordnet werden. Wir untersuchen deshalb, ob ein KI-Modell Abfall anhand eines Fotos einer von sechs Klassen des TrashNet-Datensatzes zuordnen kann. Die Auswertung soll zeigen, wie oft die Zuordnung bei den Testbildern gelingt und welche Klassen das Modell verwechselt. Ob die Bilderkennung auch unter realen Bedingungen in einer Sortieranlage zuverlässig funktioniert, wird nicht geprüft.

## 2. Stakeholder der Gruppenarbeit

| Stakeholder | Aufgabe und Interesse im Projekt |
| --- | --- |
| Requirements Engineering | Beschreibt den Bedarf, grenzt den Prototyp ab und dokumentiert Anforderungen, Prüfkriterien und den Technologievergleich. |
| Projektmanagement | Koordiniert Aufgaben, Fristen und Abgabe. Benötigt einen klaren Überblick über den Arbeitsstand und offene Punkte. |
| Daten und Datenqualität | Bereitet den Datensatz vor und dokumentiert die Datenaufteilung. Benötigt eine nachvollziehbare Trennung von Trainings-, Validierungs- und Testbildern. |
| Modell und Training | Wählt das Modell aus und trainiert es. Benötigt vorbereitete Daten, einen klaren Projektumfang und eine geeignete Laufumgebung. |
| Evaluation, Edge und Bericht | Wertet die Ergebnisse aus, betrachtet einen möglichen Edge-Einsatz und dokumentiert die Erkenntnisse. Benötigt einen festen Testsplit und nachvollziehbare Bewertungskriterien. |

## 3. Potenzielle Stakeholder bei einem realen Einsatz

Als mögliches Einsatzszenario betrachten wir eine Sortieranlage, in der die Bilderkennung das Personal bei der Zuordnung von Abfall unterstützt. Die folgenden Gruppen sind nicht am Prototyp beteiligt. Ihre Bedürfnisse sind Annahmen und wurden nicht durch Interviews oder Umfragen überprüft.

| Stakeholder | Angenommener Bedarf |
| --- | --- |
| Mitarbeitende an der Sortierlinie | Möchten die vorgeschlagene Abfallklasse verstehen und unklare oder falsche Zuordnungen überprüfen und bei Bedarf korrigieren können. |
| Betreiber der Sortieranlage | Möchte beurteilen können, ob die Bilderkennung Fehlzuordnungen reduziert und sich in den bestehenden Sortierablauf integrieren lässt. |
| Technischer Betrieb und Instandhaltung | Benötigen eine verständliche Dokumentation, um Kamera, Rechner und Software zu betreuen und Störungen zu beheben. |
| Qualitätsmanagement der Sortieranlage | Möchte erkennen, welche Abfallarten häufig verwechselt werden und wo Nachkontrollen sinnvoll wären. |

## 4. Abgrenzung und Dokumentation

Die Bedarfsanalyse stützt sich auf die Aufgabenstellung und den Projektvorschlag. Für den Prototyp greifen wir den Bedarf an einer nachvollziehbaren Auswertung je Abfallklasse auf. Eine Korrekturfunktion, eine Benutzeroberfläche und die Anbindung an eine reale Sortieranlage sind nicht Teil der Umsetzung.

Die konkreten Anforderungen, Prüfkriterien und der Technologievergleich stehen in [`requirements.md`](requirements.md). Die Rollen wurden im Gruppenchat abgestimmt. Ein gemeinsamer Abgleich der Bedarfsanalyse und der Anforderungen fand nicht statt und wurde im Protokoll dokumentiert. 


