# Requirements Engineering – KI-gestützte Abfallklassifikation

**Stand:** Abgabefassung für ESA 1. Der gemeinsame Abgleich der Anforderungen blieb offen.
**Hinweis zur KI-Nutzung:** ChatGPT (GPT-6) wurde zur sprachlichen Überarbeitung eines Entwurfs für dieses Anforderungsdokument verwendet. Die Unterstützung umfasste auch Vorschläge für den Technologievergleich. 

## 1. Bedarfsanalyse und Projektumfang

Bei der manuellen Abfallsortierung können Gegenstände falsch zugeordnet werden. Wir untersuchen, ob ein KI-Modell Abfall anhand eines Fotos einer von sechs Klassen des TrashNet-Datensatzes zuordnen kann. Die Auswertung soll zeigen, wie häufig die Zuordnung auf den Testbildern gelingt und welche Klassen das Modell verwechselt. Damit wird nicht geprüft, ob das Modell unter realen Bedingungen in einer Sortieranlage zuverlässig funktioniert.

Die Bedarfsanalyse stützt sich auf die Aufgabenstellung und den Projektvorschlag. Mitarbeitende in der Abfallsortierung sind eine mögliche Anwendergruppe. Dass sie verständliche Zuordnungen und eine Möglichkeit zur Nachkontrolle benötigen, ist eine Annahme und wurde nicht durch Befragungen bestätigt. Weitere Beteiligte und ihre Interessen sind in [`stakeholder.md`](stakeholder.md) beschrieben.

ESA 1 ist auf einen Prototyp mit einem Modell begrenzt. Eine Benutzeroberfläche, die Anbindung an eine Sortieranlage und die Umsetzung auf einem Edge-Gerät sind nicht Teil des Prototyps. Edge-Einsatz und Vision Transformer sind mögliche Erweiterungen.

## 2. Funktionale Anforderungen

| ID | Anforderung | Prüfkriterium |
| --- | --- | --- |
| FA-01 | Der Prototyp muss Bilder aus dem TrashNet-Datensatz verarbeiten können. | Das Skript liest die Bilder aus dem festgelegten Testsplit ein. |
| FA-02 | Das Modell muss jedem verarbeiteten Bild genau eine der sechs Abfallklassen zuordnen. | Für jedes ausgewertete Bild liegt genau eine Vorhersage mit einer gültigen Klasse vor. |
| FA-03 | Die Auswertung muss die Vorhersagen mit den hinterlegten Klassen der Testbilder vergleichen. | Die Auswertung zeigt den Anteil richtiger Vorhersagen (Accuracy) insgesamt und macht die Ergebnisse je Klasse sichtbar, etwa mit einer Konfusionsmatrix. |

## 3. Nicht-funktionale Anforderungen

| ID | Anforderung | Prüfkriterium |
| --- | --- | --- |
| NFA-01 | Training und Auswertung müssen für andere Teammitglieder wiederholbar sein. | Datenquelle, fester Seed, [Datensplit](../data/splits/split.csv), Laufumgebung, Paketversionen und Skriptaufrufe sind dokumentiert. |
| NFA-02 | Die abschließende Bewertung muss auf Bildern beruhen, die nicht für Training oder Modellauswahl verwendet wurden. | Der feste Testsplit wird nur für die abschließende Bewertung verwendet. Augmentation wird nur auf Trainingsbilder angewendet. |

Eine Mindestgenauigkeit und eine maximale Antwortzeit wurden nicht als Anforderungen festgelegt. Der im Protokoll genannte Konfidenzschwellenwert von 80 % ist eine Einstellung für die Evaluierung und keine geforderte Mindestgenauigkeit. Ob der Schwellenwert für einen realen Einsatz sinnvoll ist, bleibt offen.

## 4. Technologische Bewertung

| Kriterium | Vortrainiertes ResNet18 anpassen (Transfer Learning) | Kleines CNN ohne Vortraining |
| --- | --- | --- |
| Eignung für die Aufgabe | Bereits gelernte Bildmerkmale können weiterverwendet werden. Die Ausgabeschicht muss an sechs Klassen angepasst werden. | Keine vortrainierten Gewichte nötig. Das Modell muss die Bildmerkmale aus den Trainingsbildern selbst lernen. |
| Benutzerfreundlichkeit für das Team | Die vorhandene Architektur kann übernommen werden. Die vortrainierten Gewichte müssen geladen werden. | Es müssen keine Gewichte geladen werden. Architektur und Trainingseinstellungen müssen selbst festgelegt werden. |
| Skalierbarkeit | Mehr Trainingsbilder erhöhen den Aufwand pro Trainingsdurchlauf. Weitere Klassen erfordern eine angepasste Ausgabe und erneutes Training. | Auch hier erhöhen mehr Bilder und zusätzliche Klassen den Trainingsaufwand. Der Rechenbedarf pro Bild hängt von der gewählten Architektur ab. |
| Ressourcen und Kosten | Der Projektplan sieht Colab mit GPU für das Training vor. Laufzeit und Kosten wurden für den Vergleich nicht erhoben. | Der Ressourcenbedarf hängt von der Architektur und der Trainingsdauer ab. Laufzeit und Kosten wurden für diese Option ebenfalls nicht erhoben. |

**Empfehlung:** ResNet18 mit Transfer Learning ist der bevorzugte Ansatz aus dem Projektvorschlag. Für den knappen Zeitrahmen spricht, dass das Team auf einem vortrainierten Modell aufbauen kann. Das kleine CNN dient als Vergleichsoption. Die Bewertung ist qualitativ und beruht nicht auf einem direkten Test beider Ansätze. Die Empfehlung ist kein Nachweis dafür, welches Modell im Trainingsskript tatsächlich verwendet wurde.

## 5. Teamabstimmung und Dokumentation

Die Rollen wurden im Gruppenchat abgestimmt. Ein gemeinsamer Abgleich der Bedarfsanalyse und der Anforderungen fand nicht statt. Die Anforderungen und Prüfkriterien dokumentieren daher den erarbeiteten Soll-Stand, keine gemeinsam bestätigte Freigabe und keinen Nachweis der Umsetzung. Bei Änderungen am Projektumfang oder an den Anforderungen wird dieses Dokument aktualisiert.
