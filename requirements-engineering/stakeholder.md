# Stakeholder und Bedarfsanalyse – KI-gestützte Abfallklassifikation

## 1. Ausgangslage und Bedarf

Bei der manuellen Abfalltrennung können Gegenstände falsch zugeordnet werden. In der Gruppenarbeit untersuchen wir deshalb, ob ein KI-Modell Abfall anhand eines Fotos einer von sechs Klassen des TrashNet-Datensatzes zuordnen kann. Der Prototyp soll zeigen, wie gut das unter den Bedingungen des Datensatzes funktioniert und bei welchen Klassen das Modell Schwierigkeiten hat. Ob das Modell im echten Sortierbetrieb hilfreich wäre, wird mit diesem Prototyp nicht geprüft.

## 2. Beteiligte und ihre Interessen

| Stakeholder | Rolle im Projekt | Bedarf oder Interesse |
| --- | --- | --- |
| Projektteam | Erstellt, testet und dokumentiert den Prototyp. | Ein klar abgegrenzter Projektumfang, abgestimmte Anforderungen und eine Vorgehensweise, die andere Teammitglieder nachvollziehen und wiederholen können. |
| Lehrende | Beurteilen die Gruppenarbeit. | Eine nachvollziehbare Dokumentation der Ergebnisse, der Beiträge der einzelnen Teammitglieder und der aufgetretenen oder ungelösten Schwierigkeiten. |
| Mitarbeitende in der Abfallsortierung (potenzielle Anwendergruppe) | Wären bei einem späteren Einsatz betroffen; sie sind nicht als Beteiligte am Prototyp eingeplant. | **Annahme:** Die vorhergesagte Abfallklasse sollte verständlich sein. Für einen späteren Einsatz wäre außerdem wichtig zu wissen, welche Abfallarten das Modell häufig verwechselt. |

## 3. Abgrenzung und Abstimmung

Die Bedarfsanalyse stützt sich auf die Aufgabenstellung und den Projektplan. Die Bedürfnisse der potenziellen Anwendergruppe sind Annahmen und wurden nicht durch Nutzerinterviews oder Umfragen bestätigt. Für die Gruppenarbeit stehen daher die Klassifikation der Testbilder und die nachvollziehbare Auswertung im Vordergrund. Eine Benutzeroberfläche oder die Integration in eine echte Sortieranlage ist nicht Teil des Prototyps.

Die konkreten Anforderungen und der Technologievergleich stehen in [`requirements.md`](requirements.md). Die Annahmen und der Projektumfang werden im Gruppenchat abgeglichen. Relevante Rückmeldungen oder Änderungen werden im [Projektprotokoll](../protokoll.md) festgehalten.
