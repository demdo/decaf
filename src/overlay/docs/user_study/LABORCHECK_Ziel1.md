# Laborcheck Ziel 1: X-ray-Sicht in Magenta

Stand 05.11.2026. Am Rechner geprüft ist alles in `tests/test_xray_view.py`, aber ohne Kamera und ohne Engine. Diese Punkte kann nur das Labor zeigen.

## Anzeige

| Nr. | Lage | Erwartung |
|---|---|---|
| 1 | Bohrer stark geneigt gegen Π, wie B (rund 25–30°), in der Aufnahme sichtbar | Magenta liegt auf dem Bohrerschatten, Grün daneben. Die Linien laufen auseinander (B: 10,5°), die Spitzen liegen fast aufeinander (B: 1,9 px). |
| 2 | Bohrer in Π und parallel dazu, wie A | Grün und Magenta fallen fast zusammen. |
| 3 | Slider Plane depth ±10 mm | Röntgeninhalt und Magenta bewegen sich gemeinsam, Grün bleibt stehen. |
| 4 | Spitze über den Feldrand führen | Ring nur im Feld. Die Vorwärtslinie erscheint schon, bevor die Spitze im Feld ist. Nichts Magentafarbenes außerhalb der Maske, auch nicht am Rand. |
| 5 | Marker teilweise verdecken | Grün wird orange, Magenta gedämpft. **Selbst entscheiden:** Ist das gedämpfte Magenta auf dem Röntgenbild noch lesbar, oder soll es ganz aus? Es ist eine Konstante: `XRAY_MARKED_COLOR`. |
| 6 | Schalter einzeln aus und an | Jeder Schalter blendet nur seine Farbe aus. |
| 7 | Tracking-Test ohne Röntgen | Kein Schalter „Show X-ray view“, kein Magenta. |

## Abläufe, die nur mit Stubs liefen

| Nr. | Ablauf | Erwartung |
|---|---|---|
| 8 | „Set reference plane“, danach Tracking neu starten | Magenta folgt dem neuen Bild und dem neuen `d_x`, die Feldgrenze ist die des neuen Bildes. |
| 9 | Snapshot während des Trackings | `03_…` wie bisher, neu `04_rgb_overlay_drill_xrayview.png`; `snapshot.json` hat `tool.xray_view`. Layer 04 entspricht dem Schirm, wenn beide Schalter an sind. |

## Kosten im echten Log

Evaluation-Modus, einige Minuten mit Tracking. In `live_profile.log` stehen die Stufen `xview` (Zeichnen je Frame) und `xview_up` (Geometrie je Pose) mit drei Nachkommastellen, je Frame auch in `<label>_display.npz` (`xview_ms`) und `<label>_tracking.npz` (`xview_update_ms`).

| Messung | Zeichnen | Geometrie je Pose | Summe je Frame | Grün zum Vergleich |
|---|---|---|---|---|
| Echte Seite, Kamera und Engine als Stub (05.11.) | 0,057 ms | 0,132 ms | 0,19 ms | 0,14 ms |
| Labor | | | | |

Das Ziel unter 0,1 ms ist damit noch nicht erreicht. Der Nutzer hat am 05.11. entschieden, dass alle Einzeichnungen in Schritt 2 mit der GUI-Anpassung deutlich schneller werden sollen, also nicht schon hier. Der Stub-Wert ist nur ein Anhaltspunkt: Die Last der Engine ist dort nachgebildet.

Prüfen: `interval` bleibt bei 33 ms, und `pose` und `polls empty` ändern sich mit Magenta an und aus nicht.
