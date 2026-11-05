# Konzept: Live-Seite für die User Study „DHS bei Hüftfraktur“

Stand 05.11.2026. Grundlage für die folgenden Chats, hier ist noch nichts am Code geändert.
Quellen: `Literatur/User_Study` (komplett gelesen), Thesis 6.2.1, `gui/pages/page_overlay_live.py`,
`tools/tracking_overlay.py`, Aufnahmen von Fig. 4.21 (`Thesis/Plots/Evaluation/data/IntegratedSystem`).

Abkürzungen der Quellen: Takai 2018 (ADAPT), Wilharm 2013 (Gamma 3, Brainlab), Herman 2009 (Surgix), Hüfner 2003
(Navigierte Osteosynthese), Müller 2011a (DHS, Fortschr Röntgenstr), Müller 2011b (PFNA, Proc IMechE),
Abdulkareem 2012 (TAD-Review), Quah 2017 (Strahlung), Goyal 2021 (trackerlos), Liu 2024 (Roboter + FNS),
Sun 2025 (CS / DHS / FNS).

## 1 Was die Literatur sagt

### 1.1 Der Eingriff und sein Ziel

Die DHS fixiert extrakapsuläre (per-/intertrochantere) Frakturen. Die Schenkelhalsschraube gleitet in der Hülse der
Platte und komprimiert die Fraktur unter Belastung (Abdulkareem). Der Ablauf: geschlossene Reposition auf dem
Extensionstisch, Führungsdraht über einen 135°-Winkel durch die laterale Kortikalis in den Kopf, Länge messen,
überbohren, Schraube, Platte, Abschlusskontrolle in zwei Ebenen.

Worauf alles zielt, ist die Schraubenlage im Kopf:

| Größe | Definition | Zielwert | Quelle |
|---|---|---|---|
| TAD | Abstand Spitze–Apex in AP + in lateral, je um die Vergrößerung korrigiert | < 25 mm, ideal 5–20 mm | Baumgaertner über Abdulkareem, Takai |
| Lage | Quadrant in beiden Ebenen | zentral-zentral, inferior-zentral akzeptiert; superior und anterior am schlechtesten | Abdulkareem, Herman, Takai Fig. 5 |
| Schaftwinkel | Schraube gegen laterale Kortikalis | 135° (Platte), sonst steht die Platte ab | Hüfner, Müller 2011a |
| TSD | Spitze bis Kopfoberfläche entlang der Achse, in 3D | ADAPT voreingestellt 5 mm | Takai |
| Länge | Schraubenlänge aus Drahtlänge | | Herman, Goyal |

Versagen heißt meist Cut-out nach superolateral, in Serien 1–6,8 % (Abdulkareem). Das stärkste Vorhersagemerkmal ist
die TAD, danach die Schraubenlage.

### 1.2 Warum der C-Bogen, warum so viel Strahlung

Der Draht wird perkutan gesetzt, der Knochen ist nicht sichtbar. Die Lage ist eine 3D-Frage, das Röntgenbild aber eine
Projektion, also braucht jede Kontrolle zwei Ebenen. Die Rückmeldung kommt nur Schritt für Schritt: Draht korrigieren,
schießen, schwenken, schießen. Eine Korrektur in der einen Ebene kann die andere verschlechtern. Die Dosis steigt mit
der OP-Zeit, Patient und Team sind betroffen, die Hände des Operateurs liegen nah am Strahl (Wilharm, Müller 2011a).
Quah nennt zwei Treiber: die Reposition und die Drahtlage in beiden Ebenen, dazu die Suche nach dem „perfect picture“.

Konventionelle Größenordnung: 13–33 Bilder oder 17–55 s (Wilharm, Müller, Hüfner), 30 Pulse allein für den Draht
(Herman), Median 16 Bilder (Sun), im Mittel 245 cGy·cm² über 1.203 DHS (Quah). In Hüfners Laboraufbau liefen zwei
C-Bögen gleichzeitig. Das Schwenken kostet außerdem Zeit, ist ergonomisch mühsam und gefährdet die Sterilität
(Müller 2011a, Wilharm).

### 1.3 Die Schwierigkeiten für den Chirurgen

1. Eintrittspunkt und Winkel in zwei Ebenen zugleich treffen, nur mit Standbildern.
2. Fehlversuche: im Mittel 2–4 Bohrungen (Wilharm 4,1; Herman 3,05; Müller 2,3–2,5). Jede Fehlbohrung schwächt die
   laterale Kortikalis und das spongiöse Lager, bis zur subtrochantären Fraktur; der nächste Draht kann in den alten
   Kanal rutschen.
3. Tiefe: nicht durch den Kopf ins Gelenk, aber nah genug an den subchondralen Knochen.
4. Der Draht biegt sich. Das sieht keine Navigation (Hüfner, Müller 2011b). Hüfner bohrte deshalb navigiert mit dem
   steifen 3,2-mm-Bohrer und schob den Führungsdraht danach ein.
5. Das Fragment kann sich nach der Aufnahme verdrehen, das Bild veraltet unbemerkt (Wilharm). In Hüfners Gruppe 3
   führte die Kontrollaufnahme in 30 % der Fälle zu einer neuen Bohrung.
6. Patientengut: über 80 Jahre, vorerkrankt, kurze OP-Zeit ist ein Ziel an sich (Wilharm, Quah).

### 1.4 Was computerassistierte Systeme belegen

| Studie | Modell | Bilder/Pulse konv. → nav. | Bohrversuche konv. → nav. | Zeit konv. → nav. | Genauigkeit |
|---|---|---|---|---|---|
| Wilharm 2013 | Sawbone, 1 Operateur | 33 → 2,6 | 4,1 → 1 | 584 → 662 s (n. s.) | TAD gleich |
| Müller 2011a (DHS) | Sawbone, 1 Operateur | 12,8/16,3 → 4,7/4,0 | 2,3/2,5 → 1 | 14 → 27–29 min | TAD gleich |
| Müller 2011b (PFNA) | Sawbone, 1 Operateur | 16,7/14,5 → 4,7/4,0 | 1,7/2,5 → 1 | deutlich länger | TAD gleich |
| Hüfner 2003 | Kunstknochen | 55,4 s → 3,8–7,9 s | | 14,9 → 14,7–15,8 min | TAD gleich |
| Herman 2009 (Surgix) | 61 Patienten, nicht randomisiert | 30,1 → 17,6 (erfahren), Anfänger 28,2 | 3,05 → 1,33; beim 1. Versuch 10 → 78 % | 5,9 → 3,6 min Draht | TAD gleich |
| Goyal 2021 | 81 Patienten, nicht randomisiert | 37,8 → 20,9 (p = 0,059) | 5,3 → 3 | 19,6 → 9,9 min Draht | |
| Takai 2018 (ADAPT) | 55 Patienten, ohne Kontrolle | | | | RMSE TSD 0,63 / TAD 1,53 mm |

Daraus folgt für die Studie: Keine der Vergleichsstudien, die die TAD messen, verbessert sie signifikant. Belegt sind weniger Bilder, weniger
Fehlbohrungen, weniger Schwenken. Der Preis ist Aufbauzeit: DRB per Schanz-Schraube, Registrierung, Kalibrieren der
Instrumente. Herman nennt eine Lernkurve von rund vier Eingriffen. Bei Herman und Goyal bestehen Interessenkonflikte,
bei Takai ist ein Autor bei Stryker angestellt.

Endpunkte unserer Studie sollten also Bohrversuche, Bildzahl und Zeit sein, die TAD nur als Sicherheitskriterium.

### 1.5 Was die Anzeigen zeigen

| System | Anzeige | Was wir übernehmen können |
|---|---|---|
| ADAPT (Takai Fig. 2, 4) | Zwei runde Röntgenbilder nebeneinander, Trajektorie mit mm-Lineal, virtuelle Schraube, große Felder TAD, „Default Tip Distance“, „Distance to go“, Nagelwinkel | Bildaufteilung AP/lateral; „Distance to go“ als wichtigste Zahl; 3D-Kopfmodell aus zwei Bildern (erkennt laut Takai teils das Acetabulum falsch, deshalb manuell nachstellbar) |
| Surgix (Herman Fig. 3) | Verlängerungslinie des Drahts, Lineal, Implantat-Schablone; legt Information auf das echte Bild, statt ein virtuelles zu erzeugen | Rückmeldung vor dem Bohren; Lineal auf echtem Röntgenbild; alte Drahtlinien bleiben stehen |
| Brainlab (Wilharm Fig. 3–5) | AP + lateral nebeneinander, Werkzeug als Rendering mit Verlängerung, Beschriftung | Zwei Standbilder reichen für alle Schritte |
| SurgiTAIX (Müller Abb. 3–7) | Planung: Sicherheitskreis im Kopf, Zylinder im Hals, Warnung „Perforation!“; Führung: Fadenkreuz mit Kreis (Spitze) und Raute (Achspunkt), Tiefenbalken ab lateraler Kortikalis, Winkel zur Schaftkortikalis; „Zero-dose“-Vorschau zum Einstellen des C-Bogens | Safe Zone als einfache 2D-Formen in beiden Bildern; Winkelanzeige (von Hüfner gefordert); Vorschau vor dem Schuss |
| Goyal | Vorhersage der Drahtlage, Schablone für Schraube + Platte, Längen | Schablone als spätere Stufe |

Was keiner hat und wir haben: das Kamerabild. Es zeigt Haut, Hände und den echten Bohrer und erlaubt die Einstichstelle
ohne zusätzliches Röntgenbild zu wählen. Außerdem braucht die AP-Ansicht keinen Knochenpin, weil die Kamera am C-Bogen sitzt. Das gilt, solange sich Patient und Bogen nach der Aufnahme nicht bewegen.

### 1.6 Was daraus für DeCAF folgt

1. **Vorhersage vor dem Bohren ist der Kern.** Die Vorwärtslinie auf dem Röntgenbild adressiert direkt die Fehlbohrungen.
2. **Zwei Standbilder, danach nicht mehr schwenken.** Das ist der Hebel bei Dosis und Sterilität.
3. **Zahlen ergänzen das Bild, sie ersetzen es nicht.** Surgix legt bewusst nur eine Informationsschicht auf das echte Bild.
4. **Das Bild altert.** Jede Röntgenansicht braucht ein sichtbares Alter und eine Kontrollaufnahme im Ablauf.
5. **Steifes Werkzeug.** Ein gebogener Draht bleibt unsichtbar, also für die Studie einen steifen Bohrer wie bei Hüfner.
6. **Wenig Zusatzschritte.** Jede Klickfolge landet in der OP-Zeit, Müller 2011a hat sie fast verdoppelt.

## 2 Geometrie: eine Formel für alle Panels

Namen im Code: `T_xc` bildet **X-ray → Kamera** ab und ist in **Metern** gespeichert (die Thesis nennt diese Abbildung
`T_cx`). Tool-Punkte und `d_x` sind in Millimetern. Das sind die beiden Stolperfallen beim Einbau.

```
P_x   = R^T (P_c − 1000 t)               Punkt im X-ray-Frame
uv_x  = K_x P_x / z_x                    Pixel im Arbeitsbild (gespiegelt, entzerrt)
S_c   = R (d_x / z_x · P_x) + 1000 t     Punkt auf Π entlang des Quellstrahls, im Kamera-Frame (Thesis Gl. FW_Tip_On_Plane)
uv_c  = projectPoints(S_c; K_c, dist)    wo Panel 1 ihn zeichnet
```

Allgemein mit Ansicht `v`:

```
P_x(v) = T_x←c(v) · T_c←r(v) · T_r←c(jetzt) · P_c(jetzt)
```

Für das aktuelle AP-Bild ist die Kette `T_c←r(v) · T_r←c(jetzt)` die Einheit, solange der C-Bogen steht. Für lateral
kommt sie aus dem Referenzobjekt. Panel 2 und Panel 3 sind also dieselbe Klasse mit anderer Kette.

Drei Folgerungen:

- **Das Röntgen-Panel hängt nicht von `d_x` ab**, denn `uv_x` enthält `d_x` nicht. Es braucht auch keinen Pointer. Ein
  Fehler der Ebene wirkt nur in Panel 1, und dort bewegen sich Röntgeninhalt und Magenta gemeinsam.
- **Das Lineal ist echt in 3D.** Die Striche werden entlang der Achse in mm gesetzt und projiziert, ein geneigter Bohrer
  bekommt also von selbst engere Striche.
- **Die Auflösung liegt im Röntgen-Panel.** Auf Π sind das 8,36 px/mm im Röntgenbild, 3,2 px/mm in der Kamera. Abgelesen wird in Panel 2 und 3.

### Geprüft an den Aufnahmen von Fig. 4.21

Skripte in `scripts/`. Getrackte Pose aus `snapshot.json`, Bohrerschatten aus dem Röntgenbild:

| Aufnahme | Neigung gegen Π | Spitze X-ray-Sicht → Schatten | heute (4.4.1) | Achse X-ray-Sicht ↔ Schatten | heute (4.4.1) |
|---|---|---|---|---|---|
| A_2 | 1,7° | 0,61 mm | 1,32 mm | 0,23° | 0,5° |
| B_2 (3 Snapshots) | 27,7° | 0,26–0,47 mm | 0,70 mm | 0,04–0,11° | 8,8° |
| Bone_New | 0,4° | 1,09 mm | 0,90–1,08 mm | 0,08° | 0,04° |

Einschränkung: Bei A ist die Verbesserung teils Zufall, denn der getrackte Tiefenfehler hebt dort den Längsversatz auf. Die Spitzenzahl ist also kein Gütesiegel; B zeigt den echten Gewinn bei der Achse.
Auf dem Kamerabild liegen grün und magenta bei B 10,5° auseinander, die Spitzen 1,9 px.

## 3 Ziel 1: Thesis 6.2.1 im heutigen Live-Bild

Eine Änderung an Zeichnung und Geometrie, das Layout bleibt.

**Anzeige**
- Grün wie heute: Ring + Kreuz, gestrichelter Bohrer 50 mm, Vorwärtslinie bis zum Bildrand. Pixelgleich zum heutigen Stand.
- Magenta: gleiche Glyphen an `uv_c` von Spitze und Rückpunkt. Vorwärtslinie und Bohrer **nur innerhalb der Röntgenmaske**,
  da außerhalb kein Röntgeninhalt zum Vergleich da ist. Der Spitzen-Glyph nur, wenn die Spitze im Feld liegt. Die
  Vorwärtslinie zeigt schon vorher, wo der Bohrer im Röntgenbild landen wird.
- Orange (`MARKED_COLOR`) gilt weiter nur für die grüne Kamerasicht; Magenta bei markierter Pose gedämpft oder aus (im Chat entscheiden).
- Zwei Schalter im Rail statt einem: Kamerasicht, X-ray-Sicht. Im Modus Tracking-Test ohne Röntgen ist Magenta aus.

**Code**
- Neu `tools/xray_view.py`, ohne Qt: `XrayView(T_xc_m, K_x)` mit `to_xray`, `uv_x`, `on_plane_cam(d_x)`, `off_plane_mm`, `inclination_deg`.
- `TrackingOverlay`: zweiter Pfad, `update()` rechnet beide Sichten, `draw_into(img, show_camera, show_xray)`. Der Docstring verlangt das dort.
- Clip an der konvexen Hülle der Maske im Arbeitsbild, dort ist sie wirklich konvex. Die Hülle einmal je Röntgenbild berechnen.
- `d_x` aus `_current_d_x_mm` samt Slider. Auch die zweite Stelle im Snapshot-Code (Zeile ~2685) zeichnet das Tool.
- Snapshot und Frame-Log erweitern: `tip_x_mm`, `tip_uv_x`, `back_uv_x`, `tip_uv_on_plane`, `off_plane_mm`, `inclination_deg`, neue Ebene `04_rgb_overlay_drill_xrayview.png`.

**Abnahme**
- Grüne Zeichnung pixelgleich zu `03_rgb_overlay_drill.png` der bestehenden Snapshots.
- Regression auf den drei Aufnahmen: Tabelle aus Abschnitt 2 reproduziert, Test überspringt sich, wenn die Daten fehlen.
- Slider `d_x` ±10 mm: Magenta bleibt auf dem Bohrerschatten, die grüne Linie nicht.
- Zusatzkosten unter 0,1 ms je Frame im echten Log (`PROFILE_LIVE`), Offline-Werte sind hier erfahrungsgemäß zu optimistisch.
- Das neue Bild vom Nutzer im Labor mit einer B-artigen Lage geprüft.

**Interessanter Nebeneffekt:** Wenn grün und magenta zusammenfallen, liegt der Bohrer in Π und parallel dazu. Die Schere
der beiden Linien ist also selbst eine Anzeige. Ob Chirurgen das so lesen, ist offen (Thesis 6.2.1) und eine echte Studienfrage.

## 4 Ziel 2: drei Panels

### 4.1 Layout

Entwürfe mit echten Daten von B_2_live, je 1920 × 1080:

- `mock_layout_B.png`: **drei gleich große Quadrate** 640 × 640, darunter Bedienbereich. **Empfehlung.**
- `mock_layout_A.png`: Live groß links, zwei Röntgenbilder rechts gestapelt.

Gründe für B:

- Röntgen-Panels größer (5,2 statt 4,2 px/mm auf Π), und dort wird abgelesen.
- AP und lateral nebeneinander wie bei ADAPT und Brainlab.
- Panel 1 ist ein Ausschnitt in nativer Auflösung um das Röntgenfeld. Das Feld bedeckt nur rund 6 % des Kamerabilds, der Maßstab bleibt also gleich und das Skalieren entfällt.
- Unten bleibt Platz für große Zahlen, die man aus 1–2 m liest, und für alle Bedienelemente, ohne dass etwas ein Bild verdeckt.

Der Nachteil von B ist weniger Kontext im Kamerabild, das Bohrer-Label liegt oft außerhalb. Abhilfe: Doppelklick maximiert ein Panel, und die Statuszeile meldet „Marker verdeckt“.

Annahme: 1920 × 1080 und Vollbild (`showFullScreen`, die Seitenleiste ist auf der Live-Seite schon ausgeblendet). Bei einem anderen Monitor neu rechnen.

### 4.2 Panels

**Panel 1, Live.** Wie Ziel 1. Marken erst nach dem Zuschnitt zeichnen, damit die Linien in Panelpixeln scharf bleiben. Die Vollbild-Ebene für Snapshots bleibt unverändert.

**Panel 2, Röntgen AP.**
- Hintergrund: das Arbeitsbild (gespiegelt, entzerrt), einmal je Größe vorskaliert, außerhalb der Maske schwarz. Im Entwurf ist unten noch ein heller Randstreifen vom Entzerren zu sehen.
- Ausrichtung: **Kameraorientierung**. `H_xc` spiegelt das Arbeitsbild vertikal, das Rohbild ist gegen die Kamera also um 180° gedreht. Ohne Anpassung käme der Bohrer in Panel 1 von rechts unten und in Panel 2 von links oben. Das ist genau das Hand-Auge-Problem, das Müller per Landmarken löst, bei uns genügt eine Drehung. Die Darstellung wird als ganzzahlige Transformation je Panel geführt und ist umstellbar, falls der Operateur auf der anderen Seite steht.
- Marken nur Magenta. Bohrer gestrichelt, damit der echte Schatten in den Lücken sichtbar bleibt.
- Lineal auf der Vorwärtslinie, an der Feldgrenze geclippt.
- Spitzenring mit Radius 2 mm auf Höhe der Spitze: So zeigt der Ring zugleich die Toleranz (< 2 mm, Thesis). Vorschlag.
- Kopfzeile: Name, Alter der Aufnahme. Ohne frische Pose höchstens 0,25 s halten, dann Marken weg und „kein Tracking“. Eine veraltete Pose darf hier nie aussehen wie eine aktuelle.
- Technik: eigenes Widget `XrayToolPanel` mit `paintEvent`, Pixmap + QPainter-Vektormarken. Das gibt saubere Linien und Text ohne Vollbildkopien. Kosten im echten Log messen, Ziel gesamt unter 2 ms je Frame. Die Regeln aus `decaf-live-performance` gelten.

**Panel 3, lateral.** Dieselbe Klasse. Erst Platzhalter. Für die Optik gibt es echtes Material: `Thesis/Plots/Limitation/data/OverlayError_AP_LAT/LAT1_live/00_xray.png`, allerdings ohne registriertes Werkzeug.

### 4.3 Distanzen: wo liegt die Null?

Dein Vorschlag war 0 am Eintritt in die Maske. Beides ist in `mock_ruler_variants.png` zu sehen.

Numerisch ist er besser, als ich erwartet hatte. Auf der echten Maske von B_2 ändert sich die Ablesung um median
0,14 mm und höchstens 0,40 mm je mm seitlichem Versatz, über alle Achsrichtungen durch die Spitze. Eine Kippung um 5° ändert sie um etwa 1 mm. Bei einer Spitze nah am Feldrand und streifendem Schnitt wird es mehr, aber das Wackeln ist nicht das Problem.

Das Problem ist die Bedeutung: Der Feldrand ist kein anatomischer Ort, er hängt an Kollimator und Lagerung. Die Zahl „61 mm hinter der Null“ sagt nichts darüber, wie tief der Bohrer im Knochen ist oder wie weit es noch ist. Beim gleichen Eingriff wäre sie nach jeder neuen Aufnahme anders.

Vorschlag in drei Stufen:

| Stufe | Null | Ablesung | Bedarf |
|---|---|---|---|
| Standard | **Spitze** | Der Wert, wo das Lineal den Zielpunkt im Bild kreuzt, ist der Rest-Weg. Das entspricht dem Lineal bei Surgix und ADAPT. | nichts, sofort da |
| Einstieg | **„Null setzen“** beim Aufsetzen auf die Kortikalis | Einbringtiefe = Schraubenlänge (Thesis 6.2.1). Unabhängig von `T_xc`, `d_x` und Overlay. | ein Tastendruck, Fußschalter wäre eine USB-Taste |
| Mit 2 Bildern | **Kopf** (Kreis in AP und lateral) | TSD, „Distance to go“, TAD, Quadrant | Ziel 4 |

Die Maske bleibt nützlich als Clip für Linie und Lineal. Den Vorschlag „0 am Maskeneintritt“ können wir im Design-Chat als Umschalter behalten und direkt vergleichen.

Grenze jeder Einzelansicht: Kreuzt das Lineal im Bild den Zielpunkt, heißt das noch nicht, dass die Achse ihn in 3D trifft. Erst die zweite Ebene zeigt das. Genau deshalb braucht es Panel 3.

### 4.4 Zahlen im Bedienbereich

Sofort verfügbar: Spitze zur Ebene Π (`n^T P_x − d_x`), Neigung gegen Π (`arcsin|n^T R^T a|`), Thesis 6.2.1. Danach:
Einbringtiefe, später Winkel zum Schaft (zwei Klicks auf die laterale Kortikalis im AP-Bild, von Hüfner als Verbesserung gefordert), TSD und TAD. Felder ohne Wert bleiben sichtbar mit Strich, damit das Layout nicht springt.

TAD geht ohne 3D-Rekonstruktion: je Ansicht ein Klick auf den Apex, dann `X_v = |uv_tip − uv_apex| · z_x,tip / f`. Die getrackte Tiefe ersetzt dabei die Vergrößerungskorrektur über den Schraubendurchmesser.

### 4.5 Bedienung

Heute liegt der Rail rechts über dem Bild und würde Panel 3 verdecken. Lösung: **Bedienkarten unten nebeneinander** (siehe `mock_layout_B.png`), links die Zahlen, rechts die Karten. Es sind dieselben Widget-Objekte wie heute (`slider_alpha`, `slider_dx`, `btn_track`, `chk_*`, `btn_plane`, `btn_save`), nur in einem horizontalen Layout. Die Signale bleiben, nichts überdeckt mehr ein Bild.

Es gibt zwei Nutzer: Der Chirurg ist steril und bedient nichts, die Versuchsleitung bedient alles. Deshalb ein **Studienmodus**, der die Karten ausblendet und die Zahlen vergrößert, per Taste umschaltbar. Für den Chirurgen genügt ein Fußschalter als USB-Taste („Null setzen“). `keyPressEvent` gibt es schon.

### 4.6 Statisch designen

Neu `gui/design_live_panels.py`: lädt einen Snapshot-Ordner (`00_xray`, `01_rgb`, `snapshot.json`), optional `B_2_tracking.npz` für das echte Zittern der Pose. Dazu Regler, die den Bohrer virtuell verschieben und kippen, damit man Verhalten am Feldrand und beim Eintritt sieht. Kamera und Tracker sind nicht nötig, das läuft auch zu Hause. Die Panels sind dieselben Widgets, die die Live-Seite später einbaut. Erst danach kommt die Integration in `OverlayLivePage`.

Neben B_2 lohnt `Evaluation_Integrated_System_Bone_New/2_live`: dort liegt der Bohrer in einem Knochenphantom, das zeigt den Kontrast gegen Knochen.

## 5 Ziel 3 als Ausblick: lateral und Referenzobjekt

Dein Ablauf (AP kalibrieren → lateral: Bild + Referenzpose → AP: Bild + Referenzpose) passt. Die Thesis stützt ihn: AP1 → LAT1 → AP2 schließt mit 0,37 / 0,70 mm, die AP-Kalibrierung überlebt also die Hin- und Rückdrehung. Der Pointer folgt erst danach in AP, weil nur Panel 1 `d_x` braucht. Vier Punkte, die vorher klar sein sollten:

**a) Das Referenzobjekt muss von beiden Kamerastellungen sichtbar sein.** Die Kamera dreht um 90° mit. Eine flache Marke auf dem Tisch ist lateral kantenständig. Kandidaten:

| Variante | Vorteil | Nachteil |
|---|---|---|
| Zweiter HydraMarker-Zylinder, Achse parallel zur Drehachse des Bogens | gleicher Tracker, IR-Fusion, gleiche Metrik wie am Bohrer (Tiefenbias kürzt sich teils heraus) | Modell und `.field` nötig |
| Planare Marke (Pointer-Paddel) auf 45°-Keil | sofort vorhanden, gut für einen ersten Test | Pose nur aus einer Ebene, verkürzt in beiden Stellungen |

Empfehlung: erst der Keil als Machbarkeitstest, dann der Zylinder. Es reicht, das Objekt nur beim Erfassen zu tracken (N Frames mitteln, Bogen steht), dann läuft der Tracker wieder aufs Werkzeug. Gleichzeitiges Tracken zweier Marker ist für den Anfang nicht nötig.

**b) Die schwächste Richtung des Trackers trifft das Seitenbild voll.** Das ist die wichtigste Einschränkung. Der Tiefenfehler der Kamera liegt im AP-Bild nur mit Faktor `tan β ≈ 0,42` vor, im Seitenbild aber mit ≈ 0,92 (Richtung aus `T_xc` von B_2, Drehung um `x_x`). Die Thesis misst für die Spitze einen Tiefenbias von +1,83 mm und RMS 2,19 mm. Im Seitenbild liegen rund 1,7 mm davon also genau in der Richtung, für die man die Seitenaufnahme macht: anterior–posterior. Dazu kommen 1,2–1,35 mm aus der AP-Kalibrierung in lateral (5.2.2). Ohne Gegenmaßnahme wären das grob 2–3 mm gegen eine Toleranz von 2 mm. Das ist eine Abschätzung, keine Messung.

Gegenmaßnahmen: die Modellskalierung aus 5.2.4 (Variante A: TRE 2,19 → 0,98 mm), in der Studie ist sie anders als in der Thesis erlaubt. Sie verschlechterte aber A_2 im AP-Overlay von 1,70 auf 2,52 mm, weil der Pointer-Bias dort heute kompensiert, also muss der Pointer mit betrachtet werden. Dazu kommen die Bilddrehung aus 5.2.2 (φ ≈ 1,73°, nur `R`) und die Messung des Werkzeugs relativ zum Referenzobjekt, wodurch sich ein gemeinsamer Tiefenfehler teils aufhebt.

**c) Der Bildtransfer ist der Engpass.** Heute kommt das Röntgenbild als IMA-Datei per USB, das dauert Minuten, und die C-Bogen-Uhr geht rund 50 min falsch. Deshalb braucht es einen Assistenten, der die Reihenfolge führt: Pose lateral erfassen, Pose AP erfassen, dann beide Bilder laden. Surgix und ADAPT greifen stattdessen den Videoausgang des C-Bogens ab; das wäre eine spätere Option, verlangt aber eine neue Intrinsik.

**d) Kleiner Bonus.** Mit dem Referenzobjekt kennt die Software den Drehwinkel des Bogens. Das erlaubt einen „Lateral erreicht“- und „zurück in AP“-Hinweis, also die Idee der Zero-dose-Vorschau von Müller. Die Feldgrenze lässt sich ebenfalls vor dem Schuss einblenden. Beides spart Probebilder. Wird das Objekt zusätzlich im Betrieb geprüft, meldet das Bewegungen des Patienten (Thesis 6.2.3, Wilharm).

**Test für Ziel 3:** wie Fig. 4.21, aber lateral: Bohrer ablegen, laterale Kontrollaufnahme, X-ray-Sicht gegen den Schatten. So messen wir die Kette direkt.

## 6 Offene Entscheidungen

| Nr. | Frage | Empfehlung | Nötig für |
|---|---|---|---|
| 1 | Layout A oder B | B | Chat 2 |
| 2 | Null des Lineals | Spitze, dazu „Null setzen“ | Chat 2 |
| 3 | Röntgenbild in Kameraorientierung oder wie am C-Bogen | Kameraorientierung | Chat 2 |
| 4 | Werkzeug der Studie | steifer Bohrer um 3,2 mm wie bei Hüfner. Der Hebel Marke–Spitze liegt heute bei rund 179 mm, ein längerer Bohrer vergrößert ihn, also neu kalibrieren und Achsfehler prüfen | Ziel 3 |
| 5 | Referenzobjekt, gibt es einen zweiten Markerzylinder? | Keil zuerst | Ziel 3 |
| 6 | Passt die Bahn ins Feld? | Feld auf Π rund 122 mm bei `d_x` 550 (∝ `d_x`). Die Strecke Eintritt–Apex am Phantom messen, das Phantom notfalls höher lagern | Ziel 3 |

Für die Studie selbst, später: Aufbau wie Wilharm/Müller (Sawbone in Schaum, Tuch darüber, Lage zwischen den Durchgängen ändern), Vergleich mit konventionell, und als echte Designfrage grün, magenta oder beides.

## 7 Reihenfolge der Chats

| Chat | Inhalt | Ergebnis |
|---|---|---|
| 1 | Ziel 1, Abschnitt 3 | Magenta im heutigen Live-Bild, Tests, Laborcheck |
| 2 | Ziel 2 statisch, Abschnitt 4 | `XrayToolPanel`, Layout, Lineal, `design_live_panels.py` auf B_2. **Mit der GUI-Anpassung alle Einzeichnungen deutlich schneller** (Nutzerwunsch 05.11.; Stand Ziel 1 in `LABORCHECK_Ziel1.md`) |
| 3 | Ziel 2 live | Einbau in `OverlayLivePage`, Bedienkarten, Studienmodus, Performance im Log |
| 4 | Ziel 3, Abschnitt 5 | Referenzobjekt, Assistent, Panel 3 echt, lateraler Test |
| 5 | Klinische Größen | Einbringtiefe, Schaftwinkel, Kopfkreis, TAD/TSD, Distance to go, Safe Zone |
| 6 | Studienmodus | Bedingungen, Ereignis-Log (Bohrversuche, Bilder, Zeit) |

Chat 1 hängt von keiner offenen Entscheidung ab.
