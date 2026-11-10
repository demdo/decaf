# Isolierter DICOM-Empfangstest (Windows / decaf)

Das Skript `dicom_storage_scp_standalone.py` ist unabhängig von der Overlay-Anwendung.
Standard: Laptop `192.168.6.243`, TCP-Port `6000`, C-Arm `192.168.6.244`.
Am C-Arm muss nichts umkonfiguriert werden. `CONQUEST` ist der bekannte Zielname,
aber kein bestätigter AE-Title. Der lokale Platzhalter `DECAF_SCP` erzwingt keine
Übereinstimmung: unbekannte Calling-/Called-AE-Titles werden angenommen und protokolliert.

## Installation und Start

In einer Anaconda PowerShell Prompt:

```powershell
conda activate decaf
Set-Location 'C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\decaf\src\overlay\tests'
python -m pip install "pynetdicom>=3,<4" "pydicom>=3,<4"
python .\dicom_storage_scp_standalone.py --self-test
python .\dicom_storage_scp_standalone.py
```

Alternativ zu pip: `conda install -n decaf -c conda-forge "pynetdicom>=3,<4" "pydicom>=3,<4"`.
Nur eine Installationsvariante verwenden. Zum Erstellungszeitpunkt war in decaf
pydicom 3.0.1 vorhanden, pynetdicom fehlte. Die Projektumgebung wird durch das
Anlegen dieser Dateien nicht verändert.

Erst nach `LISTENING 192.168.6.243:6000` ein Testbild über das bestehende Ziel
`CONQUEST` senden. Beenden mit Strg+C. Für ausführlichere Verhandlungsdiagnose:

```powershell
python .\dicom_storage_scp_standalone.py --debug
```

Der Selbsttest nutzt nur localhost und einen freien Port, erzeugt synthetische
Daten und entfernt diese danach. Er prüft C-ECHO, Annahme eines unbekannten
Ziel-AE-Titles sowie zweimaligen C-STORE derselben Instanz mit zwei getrennten,
wieder einlesbaren Dateien. Er prüft nicht die reale Verbindung zum C-Arm.

## Dateien und Protokoll

Unter `received_dicom` neben dem Skript liegen die `.dcm`-Dateien und pro Start
eine neue `.log`-Datei. Das Arbeitsverzeichnis der Konsole ist dafür unerheblich.
Die Konsole und das Log zeigen TCP-Verbindungen, Calling-/Called-AE-Titles,
angebotene/akzeptierte Storage-Kontexte, Ablehnungen, Abbrüche und Empfangserfolge.
AE-Titles erscheinen erst nach einer DICOM-Association-Anfrage; ein TCP-SYN allein
enthält diese Information nicht.

Dateinamen bestehen ausschließlich aus UTC-Zeitstempel und zufälliger UUID.
Auch wiederholter Empfang derselben SOP-Instanz erzeugt eine neue Datei.
Zuerst wird exklusiv eine `.part`-Datei geschrieben und auf den Datenträger
geflusht. Erst danach wird sie ohne Überschreiben als `.dcm` veröffentlicht.
Dafür nutzt das Skript unter Windows eine Umbenennung im selben Ordner, die bei
existierendem Ziel fehlschlägt. Bei Schreibfehlern erhält der Sender einen Fehlerstatus;
Erfolg `0x0000` wird erst nach Veröffentlichung gemeldet. Nach einem Prozessabsturz
können unvollständige `.part`-Dateien übrig bleiben; diese gelten nicht als Empfangserfolg.
Komprimierte Pixel werden nicht dekomprimiert. Unterstützt werden die der
installierten pynetdicom-Version bekannten Storage-Klassen und Transfer-Syntaxen.
Unbekannte/private Klassen können abgewiesen werden und erscheinen im Log.

Die DICOM-Dateien enthalten unverändert die übertragenen Metadaten einschließlich
möglicher Patientendaten. Dieser Test ist kein PACS, bietet keine TLS-Verbindung
und keine AE-Authentifizierung. Den Zugriff auf die isolierte Verbindung begrenzen.

## Windows-Firewall

Falls nötig, in einer **PowerShell als Administrator** eine gezielte eingehende
Regel anlegen. Diese Anleitung ändert die Firewall nicht automatisch:

```powershell
New-NetFirewallRule -Name 'DeCAF-DICOM-SCP-Test-6000' -DisplayName 'DeCAF DICOM SCP Test 6000' -Direction Inbound -Action Allow -Protocol TCP -LocalAddress 192.168.6.243 -LocalPort 6000 -RemoteAddress 192.168.6.244 -Profile Any
```

`Any` berücksichtigt, dass die Direktverbindung als öffentlich erkannt werden kann;
die Regel bleibt auf die beiden Adressen und diesen Port beschränkt. Firewall
nicht global abschalten. Nach dem Test die selbst angelegte Regel entfernen:

```powershell
Remove-NetFirewallRule -Name 'DeCAF-DICOM-SCP-Test-6000'
```

## Fehlersuche

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -eq '192.168.6.243'
Get-NetTCPConnection -LocalPort 6000 -State Listen -ErrorAction SilentlyContinue
```

- Bind-Fehler 10049: Die Adresse 192.168.6.243 ist momentan keinem lokalen Adapter zugewiesen.
- Bind-Fehler 10048: Port belegt, beispielsweise durch einen laufenden Conquest-Server.
- Wiederholtes SYN ohne SYN/ACK: Listener, lokale IP, Firewall und Netzverbindung prüfen.
- TCP-Verbindung ohne AE-Titles: Noch keine vollständige DICOM-Association empfangen.
- Association abgelehnt/abgebrochen: Log und bei Bedarf `--debug` prüfen; die tatsächlichen AE-Titles und angebotenen SOP-Klassen/Transfer-Syntaxen stehen im Log.
- C-STORE-Fehler: Speicherplatz und Schreibrechte prüfen; Fehlerdetails stehen im Log.

Wireshark-Filter: `ip.addr == 192.168.6.244 && tcp.port == 6000`.
Bei Bedarf Port 6000 über „Decode As“ als DICOM interpretieren.
Ein erfolgreicher localhost-Test ersetzt keinen Empfangstest mit dem C-Arm.

## Referenzen

- https://pydicom.github.io/pynetdicom/stable/examples/storage.html
- https://pydicom.github.io/pynetdicom/stable/user/association_accepting.html
- https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filewriter.dcmwrite.html
- https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule
