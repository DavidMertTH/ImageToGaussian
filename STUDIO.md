# SHARP Studio

Lokale Weboberfläche für Gaussian Splats mit SHARP oder Depth Anything 3. Ein Bild per Drag & Drop in den Viewer ziehen oder **Bild öffnen** anklicken. Anschließend die Bilderweiterung oder die direkte 3D-Verarbeitung wählen. Die Schaltflächen zeigen das ausgewählte Modell, etwa **180° + DA3**, **360° + DA3** oder **Nur DA3**; mit SHARP entsprechend. Bild-Ausfülltool und 3D-Modell werden vorher über das Zahnrad festgelegt. Nach der Verarbeitung lädt der Viewer die erzeugte `.ply`-Datei automatisch.

Der Viewer füllt das Fenster. Vorhandene Szenen lassen sich im **Bilder**-Menü öffnen. Mit der Maus drehen und zoomen, unten die Ansicht zurücksetzen oder den Splat speichern.

## Installation

Auf einem neuen PC einmal im Projektordner ausführen:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Voraussetzungen: aktueller NVIDIA-Treiber, Python 3.12 (mit „py launcher“) und Node.js. Das Skript richtet SHARP und die Weboberfläche ein und fragt dann, welche Modelle dazukommen sollen (Bild-Ausfülltools, Depth Anything 3, KI-Upscaler, weitere Panorama-Modelle). Bereits installierte sind markiert; weitere lassen sich später durch erneutes Ausführen ergänzen. Für FLUX.1-dev, FLUX.1 Fill, FLUX.2 Klein 9B und Krea 2 muss man vorher auf Hugging Face die Modellbedingungen annehmen; das Skript nennt fehlende Freigaben mit Link und fragt bei Bedarf nach der Hugging-Face-Anmeldung.

## Start unter Windows

Doppelklick auf **`SHARP Studio.cmd`** im Projektordner. Alternativ in PowerShell:

```powershell
& '.\SHARP Studio.cmd'
```

Ein direkter Aufruf von `.\run-studio.ps1` scheitert, wenn Windows die Ausführung von Skripts blockiert (Standardeinstellung).

Die Anwendung öffnet `http://127.0.0.1:43127` im Browser. Der erste Start installiert die Web-Abhängigkeiten, danach werden sie lokal aus `web/node_modules` verwendet. Die Basisumgebung `.venv` muss vorhanden sein.

Die Anwendung akzeptiert JPG, PNG und WebP bis 30 MB. Uploads und 3D-Ergebnisse liegen unter `output/ui/`. Vorhandene Szenen aus `output/Demo Images/` werden ebenfalls angezeigt. Die Verarbeitung läuft lokal mit dem ausgewählten 3D-Modell. SHARP verwendet automatisch CUDA, wenn eine kompatible GPU erkannt wird; der native DA3-Pfad verwendet CUDA in `.venv-depth`.

## VR-Ansicht (WebXR)

Unterstützt der Browser eine WebXR-VR-Sitzung, erscheint unten die Schaltfläche **VR**. Ohne erkanntes Headset bleibt sie ausgeblendet. Die VR-Sitzung beginnt an der aktuellen Kameraposition und mit derselben Blickrichtung; die Szene erscheint in ihrem metrischen Maßstab. **VR beenden** oder das Verlassen im Headset stellt die vorherige Desktop-Ansicht wieder her.

Steuerung mit Controllern:

| Eingabe | Wirkung |
| --- | --- |
| Linker Stick | In Blickrichtung bewegen und seitwärts gehen |
| Linker Trigger / linker Griff | Nach oben / nach unten |
| Rechter Stick links/rechts | Drehen |
| Rechter Trigger / rechter Griff gedrückt halten | Schneller / langsamer |
| **A** / **B** | Nächstes / vorheriges Bild |
| **X** oder **Y** | Zurück zum ursprünglichen Kamerastandpunkt |

WebXR benötigt einen sicheren Kontext. Der Server lauscht nur auf `127.0.0.1`. Es gibt daher zwei Wege:

- **PC-VR** (Quest Link/Air Link, SteamVR, Windows Mixed Reality): Headset verbinden und Studio wie gewohnt in Chrome oder Edge auf dem PC öffnen.
- **Quest eigenständig**: Quest per USB mit aktiviertem Entwicklermodus verbinden, `adb reverse tcp:43127 tcp:43127` ausführen und im Quest-Browser `http://localhost:43127` öffnen.

Die VR-Ansicht rendert in der vollen nativen Auflösung des Headsets. Gaussian Splats sind in VR rechenintensiv; große Szenen können vor allem auf eigenständigen Headsets ruckeln.

## 3D-Modell

Über das Zahnrad oben lässt sich das **3D-Modell** wählen. **SHARP / Depth Pro** ist der Standard. **Depth Anything 3** erzeugt mit dem nativen Nested-Giant-Modell sowohl metrische Tiefe als auch Gaussians. Im DA3-Modus werden Apple Depth Pro und SHARP vollständig ersetzt; auch die Gaussian-Vorhersage erfolgt durch DA3. Die Auswahl wird im Browser gespeichert und gilt für neue Fotos sowie 180°- und 360°-Szenen. Bereits laufende Aufträge verwenden ihre ursprüngliche Auswahl.

Depth Anything 3 einmalig einrichten:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-depth.ps1
```

Das Skript richtet die separate Python-Umgebung `.venv-depth` ein und lädt [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) nach `models/depth/nested-giant-large`. Die Gewichte umfassen etwa 6,76 GB. `models/depth/native-ready.json` kennzeichnet vorhandene Gewichte und importierbare Laufzeitabhängigkeiten; das Setup führt keine Bildinferenz aus. Danach den Studio-Server neu starten und die Einstellungen erneut öffnen oder die Browserseite neu laden. Solange der native DA3-Pfad nicht eingerichtet ist, bleibt die Option deaktiviert.

Die offiziellen Modellgewichte stehen unter **CC BY-NC 4.0** und sind für nicht kommerzielle Nutzung vorgesehen. Siehe die [offizielle Modellkarte](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1). Der DA3-Code steht unter [Apache 2.0](https://github.com/ByteDance-Seed/depth-anything-3/blob/main/LICENSE).

`da3_splats.py` verarbeitet ein normales Foto direkt mit DA3. Bei 360° werden sechs Perspektivansichten erzeugt, bei 180° fünf. Die Einzelansichten werden mit ihren bekannten Richtungen um denselben Kamerastandpunkt zusammengeführt. Für eine 16-GB-Grafikkarte ist `--process-res 352` der Standard pro Ansicht. Der native Gaussian-Pfad arbeitet experimentell; metrische Skalierung und Übergänge sind keine Garantie für messgenaue, nahtlose Geometrie.

Direkte DA3-Verarbeitung:

```powershell
.venv-depth\Scripts\python.exe da3_splats.py --input bild.jpg --output scene.ply --projection perspective
.venv-depth\Scripts\python.exe da3_splats.py --input panorama.png --output scene.ply --projection equirectangular --coverage 360 --process-res 352
```

Für eine 180°-Projektion stattdessen `--coverage 180` verwenden. `--process-resolution` ist ein Alias für `--process-res`; `--view-resolution` steuert die Auflösung der gewonnenen Perspektivansichten.

## SHARP-Auflösung und KI-Upscaling

SHARP rechnet jedes Bild auf ein Quadrat um und erzeugt daraus pro Bildpunkt-Paar zwei Gaussians. Über das Zahnrad lässt sich diese **SHARP-Auflösung** wählen:

| Auflösung | Gaussians pro Ansicht | Grafikspeicher | Hinweis |
| --- | --- | --- | --- |
| 1024 px | ca. 0,5 Mio. | ca. 5 GB | schnell, gröber |
| 1536 px (Standard) | ca. 1,2 Mio. | ca. 7,5 GB | Auflösung, mit der SHARP trainiert wurde |
| 2048 px | ca. 2,1 Mio. | ca. 11 GB | feinere Details |
| 2560 px | ca. 3,3 Mio. | über 16 GB | auf 16-GB-Karten deutlich langsamer |

Eine höhere Auflösung lohnt sich nur, wenn das Bild selbst genug Details hat. In einem Vergleich mit einem 3840 × 2160-Foto gab 2048 statt 1536 eine sichtbar schärfere Szene (+1 dB PSNR in der Ausgangsansicht). Bei Bildern um 1600 px brachte 2048 dagegen keine messbare Verbesserung. Oberhalb von 1536 arbeitet SHARP außerhalb seiner Trainingsauflösung; die Tiefe kann sich dadurch leicht verändern.

**KI-Upscaling** vergrößert Fotos, deren lange Seite kleiner als die gewählte SHARP-Auflösung ist, vorher mit [Real-ESRGAN x4plus](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause). Bei 180°- und 360°-Szenen wird jede Perspektivansicht vergrößert. Die KI erfindet dabei plausible Feinstrukturen; sie stellt keine echten Details wieder her. Einmalig einrichten:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-upscale.ps1
```

Das Skript installiert `spandrel` in `.venv` und lädt die Gewichte (64 MB) nach `models/upscale/`. Beide Einstellungen gelten nur für SHARP und werden mit dem Auftrag gespeichert.

Direkt aufrufen:

```powershell
.venv\Scripts\python.exe sharp_predict.py --input bild.jpg --output scene.ply --resolution 2048 --upscale auto
.venv\Scripts\python.exe ai_upscale.py --input bild.jpg --output bild_2048.png --target 2048
```

`sharp predict --resolution 2048` funktioniert ebenfalls; `panorama_splats.py` akzeptiert `--resolution` und `--upscale`. Das Splat-Budget von 360°- und 180°-Szenen wächst mit der Auflösung (3 Mio. bei 1536 px).

## Bild-Ausfülltool und 180°-Erweiterung

Über das Zahnrad **Bild-Ausfülltool** auswählen:

| Tool | Ergebnis |
| --- | --- |
| **FLUX Fill · 180° vor dir** (Standard) | Erweitert das Foto zur vorderen Halbkugel. Der projizierte Originalausschnitt bleibt bis auf einen schmalen Übergangsrand erhalten. |
| **FLUX.2 Klein + LoRA · 360° rundum** | Erzeugt die bisherige vollständige Rundum-Szene. |

Die Auswahl wird im Browser gespeichert und beim Upload im Auftrag festgehalten. Bereits laufende Aufträge und vorhandene Szenen behalten ihre Auswahl. **Nur SHARP** beziehungsweise **Nur DA3** verarbeitet das Foto direkt.

### FLUX Fill einrichten

FLUX.1 Fill benötigt Zugriff auf das Modell bei Hugging Face. Zuerst auf der [Modellseite](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) die Modellbedingungen annehmen. Bei vorhandener `.venv-panorama` anschließend anmelden und das Tool einrichten:

```powershell
.venv-panorama\Scripts\hf.exe auth login
powershell -ExecutionPolicy Bypass -File .\setup-panorama.ps1 -FillTool flux-fill
```

Bei einer neuen Installation zunächst `setup-panorama.ps1 -FillTool flux-fill` starten, damit die Python-Umgebung angelegt wird. Wenn der Download wegen fehlender Anmeldung abbricht, `hf.exe auth login` wie oben ausführen und das Setup erneut starten. Das Setup lädt die quantisierten Transformer- und T5-Komponenten aus [FLUX.1 Fill NF4](https://huggingface.co/diffusers/FLUX.1-Fill-dev-nf4) sowie die benötigten Basiskomponenten nach `models/fill180/`. Die Modellrevisionen sind im Generator festgelegt. Bildaufträge verwenden ausschließlich die lokalen Dateien.

Danach den Studio-Server neu starten und die Browserseite neu laden. In den Einstellungen **FLUX Fill · 180° vor dir** sowie das gewünschte 3D-Modell wählen. Solange die Fill-Modelle fehlen, bleibt die Schaltfläche **180° + SHARP** beziehungsweise **180° + DA3** deaktiviert und die Oberfläche zeigt den Einrichtungshinweis.

Im 180°-Modus läuft automatisch:

1. Das Foto wird mit angenommenen 65° horizontalem Bildwinkel auf die vordere Halbkugel projiziert.
2. FLUX Fill ergänzt fehlende Bereiche in fünf überlappenden Perspektivansichten mit je 120° Bildwinkel. Bereits vorhandene Bereiche werden beibehalten. Damit keine harten Kanten entstehen, malt FLUX zusätzlich einen schmalen Streifen über vorhandenen Inhalt, der weich überblendet wird; Farb- und Helligkeitsabweichungen der neuen Bereiche werden am Rand ausgeglichen. Am Originalfoto ist dieser Streifen nur wenige Pixel breit (`--blend`, Standard 48 Patch-Pixel, beim Original ein Viertel davon).
3. Das Ergebnis wird als **2048 × 2048 Pixel große equirectangulare Projektion mit 180° horizontal und 180° vertikal** gespeichert. Links und rechts sind unterschiedliche Halbkugelränder; sie werden nicht miteinander verbunden. Dies ist keine kreisförmige Fisheye-Datei.
4. Das gewählte 3D-Modell verarbeitet fünf Ansichten und führt ihre Gaussians um denselben Ursprung zusammen. Gaussians hinter dem ursprünglichen Kamerastandpunkt werden entfernt; die Szene enthält die vordere Halbkugel.
5. Im Viewer ist die horizontale Blickrichtung auf die vordere Halbkugel begrenzt. Die Bewegung erfolgt mit W/S, A/D und Q/E; die Pfeiltasten drehen den Blick und Ziehen mit der Maus kippt ihn.

Der geometrische Outpainting-Ansatz orientiert sich am [180°-Workflow von camera-comfyUI](https://github.com/Alexankharin/camera-comfyUI/blob/main/workflows/Outpaint_fisheye180.json) und wurde für die equirectangulare 3D-Pipeline angepasst. Neue Bildbereiche werden erfunden. Zwischen den einzelnen Erweiterungen und 3D-Ansichten können sichtbare Übergänge oder widersprüchliche Tiefen entstehen.

Die Verarbeitung läuft mit NF4-Quantisierung, CPU-Offload und VAE-Tiling in `.venv-panorama`. Eine CUDA-Grafikkarte ist erforderlich. Im Ergebnisordner liegen `panorama.png`, `panorama.control.png`, `panorama.json`, `scene.ply` und `scene.json` sowie die üblichen Auftragsdateien.

Einzelne Verarbeitungsschritte:

```powershell
.venv-panorama\Scripts\python.exe fill180_generate.py --input bild.jpg --output halbkugel.png
.venv\Scripts\python.exe panorama_splats.py --input halbkugel.png --output scene.ply --coverage 180
```

Die zweite Zeile erzeugt SHARP-Splats. Für native DA3-Splats das erweiterte Bild stattdessen mit `da3_splats.py --projection equirectangular --coverage 180` in `.venv-depth` verarbeiten.

Optionen für FLUX Fill: `--seed`, `--fov`, `--steps` (Standard 30), `--size` (Standard 2048), `--patch-size` (Standard 1024; bei knappem GPU-Speicher 768), `--guidance` und `--prompt`.

## 360°-Erweiterung

Einmalig einrichten:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-panorama.ps1 -FillTool klein-lora
```

Das Skript erstellt eine separate Python-Umgebung `.venv-panorama` und lädt FLUX.2 Klein **Base 4B** sowie die Panorama-LoRA von nomadoor herunter (zusammen etwa 16 GB). Die Modelle liegen unter `models/panorama/`. Die anschließende 3D-Verarbeitung übernimmt das ausgewählte Modell.

Danach den Studio-Server neu starten, die Browserseite neu laden und in den Einstellungen **FLUX.2 Klein + LoRA · 360° rundum** auswählen. Ein Bild hineinziehen und im Dialog **360° + SHARP** beziehungsweise **360° + DA3** wählen, um eine Rundum-Szene zu erzeugen. **Nur SHARP** beziehungsweise **Nur DA3** verarbeitet das Foto direkt. Solange die Panorama-Modelle fehlen, ist die Rundum-Erweiterung deaktiviert und der Dialog zeigt einen Hinweis.

Im 360°-Modus läuft automatisch:

1. Das Foto wird als Perspektivansicht mit angenommenen 65° horizontalem Bildwinkel in eine Kugelprojektion eingesetzt.
2. FLUX.2 Klein mit Panorama-LoRA ergänzt die Umgebung zu einem 2048 × 1024 Pixel großen Panorama. Das Foto dient als Referenz; die KI kann dabei auch Details im Originalausschnitt verändern.
3. Sechs überlappende Perspektivansichten mit je 100° Bildwinkel werden aus dem Panorama gewonnen, einschließlich Himmel und Boden.
4. Das ausgewählte 3D-Modell berechnet die Gaussians jeder Ansicht. Die bekannten Brennweiten und Orientierungen dienen zur Zusammenführung um denselben Ursprung; die Gaussians werden als gemeinsame PLY gespeichert.
5. Der Viewer zeigt die Gaussians der Rundum-Szene am Ursprung. Ziehen zum Umschauen, Scrollen zum Ändern des Bildwinkels. Transparente Lücken können sichtbar bleiben. Die heruntergeladene PLY enthält ausschließlich die Gaussians.

Für die Bewegung im 360°-Viewer einmal in den Viewer klicken: **W/S** vorwärts und rückwärts, **A/D** seitlich, **Q** nach unten und **E** nach oben. **Shift** beschleunigt die Bewegung. Mausziehen kippt die Szene, die **Pfeiltasten** drehen die Blickrichtung; **Shift** vergrößert die Drehschritte. Scrollen zoomt. **Zurücksetzen** bringt die Kamera zum Ursprung zurück.

FLUX und das gewählte 3D-Modell laufen nacheinander in getrennten Prozessen. FLUX verwendet standardmäßig CPU-Offload und VAE-Tiling, damit die Erzeugung auf einer 16-GB-Grafikkarte möglich ist. Die Panorama-Erzeugung dauert deutlich länger als eine einzelne 3D-Vorhersage; andere laufende Anwendungen und verfügbarer RAM beeinflussen die Dauer.

Laufende Aufträge erscheinen nach einem Neuladen der Seite wieder. Ein Server-Neustart startet unterbrochene Aufträge erneut. Jedes Ergebnisverzeichnis enthält `manifest.json`, `job.json` und `pipeline.log`; 360°-Ergebnisse zusätzlich `panorama.png`, `panorama.generated.png`, `panorama.control.png`, `panorama.json`, `scene.ply` und `scene.json`.

### Grenzen

Dies ist eine experimentelle Rundum-Szene aus einzeln berechneten Perspektivansichten. Sie eignet sich zum Umschauen nahe dem ursprünglichen Kamerastandpunkt. Die zusammengeführten Vorhersagen werden nicht gemeinsam geometrisch optimiert: Übergänge, doppelte Konturen und Unterschiede in der Tiefe sind möglich. Der ursprüngliche Bildwinkel wird geschätzt. Das Panorama liefert keine Informationen darüber, wie die tatsächlich unsichtbare Umgebung aussieht.

### Einzelne Verarbeitungsschritte

```powershell
.venv-panorama\Scripts\python.exe panorama_generate.py --input bild.jpg --output panorama.png
.venv\Scripts\python.exe panorama_splats.py --input panorama.png --output scene.ply --save-views
```

Die zweite Zeile verwendet SHARP. Die entsprechende DA3-Verarbeitung steht im Abschnitt **3D-Modell**.

Panorama-Optionen: `--seed`, `--fov`, `--steps`, `--width`/`--height` (2:1), `--prompt`, `--offload model` für schnellere Erzeugung bei ausreichend freiem GPU-Speicher. `--preserve-original` setzt die ursprünglichen Quellpixel wieder ein, kann dadurch aber sichtbare Übergänge erzeugen. Das Standardmodell ist die nicht destillierte Base-Variante; die LoRA ist darauf abgestimmt.

Quellen: [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B), [Panorama-LoRA](https://huggingface.co/nomadoor/flux-2-klein-4B-360-erp-outpaint-lora), [Panorama-Workflow des Autors](https://github.com/nomadoor/ComfyUI-Panorama-Stickers).

## Weitere Panorama-Modelle

Neben FLUX Fill und FLUX.2 Klein 4B lassen sich größere Modelle zur Bilderweiterung installieren. Sie brauchen eine Grafikkarte mit viel Speicher (entwickelt für eine RTX PRO 6000 mit 96 GB) und erscheinen nach der Installation im Zahnrad unter **Bild-Ausfülltool** in der Gruppe „Weitere Modelle“; darunter wird 180° oder 360° gewählt.

| Modell | Grafikspeicher | Download | Lizenz |
| --- | --- | --- | --- |
| DiT360 (FLUX.1-dev + Panorama-LoRA) | ab ~40 GB | ~34 GB | FLUX.1-dev: nicht-kommerziell |
| FLUX Fill, volle Genauigkeit (nur 180°) | ~40 GB | ~34 GB | nicht-kommerziell |
| FLUX.2 Klein 9B + 360°-LoRA | ~40 GB | ~34 GB | nicht-kommerziell |
| Qwen-Image-2512 + 360°-LoRA | ~64 GB | ~58 GB | Apache 2.0 / MIT |
| Krea 2 + 360°-Outpaint-LoRA | ~48 GB | ~40 GB | Krea 2 Community License |

Diese Modelle bekommen automatisch eine Beschreibung der Umgebung des Fotos als Prompt (Qwen3-VL 8B, einmalig ~17 GB). Personen und Hauptmotiv bleiben darin bewusst unerwähnt, damit die Modelle sie nicht in die Umgebung kopieren. Der Bildwinkel des Fotos wird aus den EXIF-Daten gelesen (sonst 65°). 360°-Modelle zeichnen das ganze Panorama; anschließend wird das Originalfoto mit weichem, farblich angeglichenem Rand wieder eingesetzt. Für 180° wird die vordere Hälfte ausgeschnitten und mit Real-ESRGAN auf 2048 × 2048 vergrößert.

Der Code liegt in `outpaint/`; jedes Modell läuft in seiner eigenen Python-Umgebung (`.venv-panorama`, `.venv-outpaint`, `.venv-dit360`). Einzelnes Modell direkt testen, zum Beispiel DiT360:

```powershell
.venv-dit360\Scripts\python.exe outpaint\dit360_pano.py --input foto.jpg --output test360.png
```

Alle installierten Modelle auf denselben Fotos vergleichen (Ergebnis in `outpaint\compare\`):

```powershell
.venv-outpaint\Scripts\python.exe outpaint\compare.py foto1.jpg foto2.jpg
```

## Pipeline-Ansicht

Oben links öffnet **Pipeline** eine Zeitleiste der Verarbeitungsschritte für die gewählte Szene oder den gerade laufenden Auftrag. Jeder Schritt zeigt Status, Uhrzeit, Dauer und seine Zwischenergebnisse; während ein Auftrag läuft, erscheinen sie, sobald sie fertig sind. Ein Klick auf ein Bild zeigt es groß, ein Klick auf **Gaussians** lädt dieses Zwischenergebnis in den 3D-Viewer und richtet den Blick auf die jeweilige Ansicht. **Zur fertigen Szene** kehrt zurück. Bei Fehlern zeigt der betroffene Schritt die Fehlermeldung und das Ende des Protokolls.

Bei 180°- und 360°-Szenen mit SHARP:

| Schritt | Zwischenergebnisse |
| --- | --- |
| Foto, Projektion | Original und die auf die Halbkugel beziehungsweise Kugel projizierte Vorlage |
| Ausfüllen | 180°: pro Ausschnitt FLUX-Eingabe, Maske, FLUX-Ergebnis und Zwischenstand der Halbkugel; 360°: das FLUX-Ergebnis |
| Panorama | das fertige equirectangulare Bild |
| DA²-Tiefe | nur mit Tiefenangleich: die Entfernungskarte (hell ist nah) |
| SHARP-Ansichten | pro Ansicht das Perspektivbild, SHARPs eigene Tiefe, die DA²-Vorgabe und die Gaussians der Ansicht |
| 3D-Szene | die fertige Szene; **Nähte messen** zeigt, wie gut benachbarte Ansichten zusammenpassen (siehe `seam_report.py`) |

Zwischenergebnisse werden nur für neue Aufträge gespeichert (`fill_steps/`, `scene_views/` im Auftragsordner). Die Gaussians pro Ansicht belegen ungefähr noch einmal so viel Speicher wie die fertige Szene. Ältere Szenen zeigen nur die Dateien, die damals entstanden sind. Für Beispielbilder gibt es keine Zwischenschritte.

## Experiment: Nahtlose Tiefe mit DA²

SHARP schätzt jede Ansicht einer 180°- oder 360°-Szene einzeln, mit jeweils eigenem Tiefenmaßstab. Das erzeugt Sprünge an den Übergängen. [DA²](https://github.com/EnVision-Research/DA-2) (Depth Anything in Any Direction) schätzt die Entfernung dagegen für das ganze Panorama in einem Durchlauf. Mit `--reference-depth` dient diese Karte als gemeinsame Vorgabe: Jede Ansicht bekommt die DA²-Tiefe entlang ihrer Sehstrahlen und gleicht ihre eigene Tiefe daran an. Dafür nutzt sie SHARPs eingebautes, gelerntes Ausrichtungsnetz. Ein gemeinsamer Maßstab wird an der Vorderansicht geeicht.

Im Studio über das Zahnrad unter **Tiefenangleich bei 180° und 360°** die Option **DA² · nahtlose Übergänge (Experiment)** wählen. Sie gilt für neue 180°- und 360°-Szenen mit SHARP; einzelne Fotos und DA3 bleiben unverändert. Die Auswahl wird im Browser gespeichert, Standard ist **Aus**. Szenen mit Tiefenangleich tragen im Bilder-Menü den Zusatz „DA²“. Das DA²-Ergebnis liegt im Auftragsordner als `da2_distance.npy`.

Einmalig einrichten (setzt `.venv-depth` aus `setup-depth.ps1` voraus; Gewichte etwa 1,4 GB, Apache 2.0):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-da2.ps1
```

Verwendung für ein 180°-Bild:

```powershell
.venv-depth\Scripts\python.exe da2_reference.py --input panorama.png --output da2.npy --coverage 180
.venv\Scripts\python.exe panorama_splats.py --input panorama.png --output scene.ply --coverage 180 --reference-depth da2.npy
.venv\Scripts\python.exe seam_report.py scene.ply
```

`seam_report.py` misst, wie weit benachbarte Ansichten dieselbe Oberfläche auseinander platzieren (Median von |log(r₁/r₂)|; 0,05 entspricht etwa 5 %). Beim Testbild „Beach 6“ sank der Median über alle Übergänge von 0,644 auf 0,007; der Übergang zur Bodenansicht von 1,474 auf 0,005.

Grenzen: Apple hat die Tiefenausrichtung mit gemessenen Tiefen trainiert, nicht mit Schätzungen eines anderen Modells. DA² ist auf volle 360°-Panoramen trainiert; Halbkugeln erhalten den echten 180°-Bildwinkel (`--hemisphere direct`, Standard). Die Detailtiefe richtet sich nach DA²s Auflösung von etwa 0,6 Megapixeln für das ganze Panorama. Helligkeitsübergänge aus dem Ausfüllschritt bleiben sichtbar; die Methode gleicht nur die Geometrie an.

## Entwicklung

Für die Entwicklung kann der Python-Server mit `.venv\Scripts\python.exe app.py` und Vite mit `npm --prefix web run dev` separat gestartet werden. Vite leitet `/api` an Port 43127 weiter.
