# Changelog

All notable changes to ginkyo are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-26

First public preview release.

### Added

- Desktop workbench for vibration and sound data (PySide6 + pyqtgraph)
- Open WAV / CSV / UFF (UNV); multi-channel project pool; CSV export
- `.ginkyo` save/open (embedded waveforms + view pages)
- Analysis: FFT spectrum (window, NFFT, averaging, amplitude scale)
- Analysis: STFT spectrogram with Overlap / fixed Δt / equal-angle / equal-RPM stepping
- Views: multi-page layouts, markers, cursor dock, Mag+Phase / spectrogram panels
- Spectrum measure: Mag Linear/dB display, overall / band RMS, peak pick
- Sample files under `samples/` (sine, UFF, tacho run-up)

### Known limitations

- dB display is `20·log10` relative to 1 (of the series unit); no SPL / ISO vibration references yet
- Band RMS expects peak or rms amplitude scale for engineering meaning (`lin` is not strict)
- Desktop binaries are unsigned (OS may warn on first launch)
- Live acquisition and plugins are not included

## [0.0.1] — 2026-07

Initial private development snapshot.
