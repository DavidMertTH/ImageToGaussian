# ImageToGaussian

Lokale Weboberfläche, die aus einem Foto eine begehbare 3D-Szene aus Gaussian Splats macht – wahlweise direkt oder nach einer KI-Erweiterung zu einer 180°-Halbkugel oder einem 360°-Panorama.

- 3D-Modelle: Apple SHARP (Depth Pro) oder Depth Anything 3
- Bilderweiterung: FLUX Fill, FLUX.2 Klein, DiT360, Qwen-Image + 360°-LoRA, Krea 2 + 360°-LoRA
- Einstellbare SHARP-Auflösung, KI-Upscaling (Real-ESRGAN), Tiefenangleich mit DA²
- Läuft vollständig lokal auf einem PC mit NVIDIA-GPU

## Installation und Start

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Das Skript richtet alles ein und fragt, welche Modelle installiert werden sollen. Danach per Doppelklick auf **`SHARP Studio.cmd`** starten. Ausführliche Anleitung: [STUDIO.md](STUDIO.md).

## Lizenzen

Der eigene Code dieses Projekts steht unter der [MIT-Lizenz](LICENSE). Der SHARP-Code in `src/sharp` stammt von Apple ([ml-sharp](https://github.com/apple/ml-sharp)) und steht unter [LICENSE_SHARP](LICENSE_SHARP), die SHARP-Modellgewichte unter [LICENSE_MODEL](LICENSE_MODEL); siehe auch [README_SHARP.md](README_SHARP.md) und [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS). Die optional installierten Modelle haben eigene Lizenzen (siehe STUDIO.md), einige davon nur für nicht-kommerzielle Nutzung.
