# XStream

**XStream** (formerly *CT_Stream*) is a single-workstation system that integrates motorized
CT acquisition, pipelined storage, local GPU reconstruction and automated provenance
recording for parallel-beam computed tomography at the **SSRF BL16U2** fast-imaging
beamline. It is designed for the case in which CT datasets are too large to be shipped to a
central computing centre: reconstruction runs on the beamline workstation itself, while
acquisition is still in progress.

This repository contains the complete C++ source of the workstation.

## Highlights

**Acquisition**
- Three acquisition modes: *step-and-shoot*, *continuous-rotation (fly)* scan, and
  *2D time-series* capture.
- **Software-defined deterministic timing for continuous-rotation scanning.** For a
  commercial large-format sCMOS camera that provides no position-synchronized triggering,
  the workstation (i) calibrates the frame period on line, (ii) derives the maximum angular
  velocity from an explicit edge-blur constraint, and (iii) aligns the frame stride to an
  integer multiple of the calibrated period, so that every retained projection corresponds
  to a known angle. No hardware trigger (PSO/MCS) is required.
- Continuous exposure control from microseconds to seconds (value + unit), with PVCAM
  exposure-resolution normalization that avoids silent sequence timeouts.
- Full automated scan setup: pre/post flat and dark fields, 0°/180° reference images for
  centre-of-rotation calibration, and a machine-readable scan manifest.

**Data path**
- Two-tier memory (driver ring buffer + reusable frame pool) with bounded queues, so that
  memory occupancy is decoupled from the number of frames acquired.
- Acquisition, writing and reconstruction overlap in separate worker threads; UI preview is
  throttled independently of the capture path.
- Concurrent 16-bit TIFF slice output with a configurable number of writer threads.

**Reconstruction**
- GPU reconstruction kernels (CUDA + cuFFT): preprocessing, filtering, back-projection,
  stripe removal, 3D back-projection (`bp3d_kernel.cu`).
- Two analytic methods: **FBP** and **Gridrec**, both as a streaming/real-time path and as
  an offline production path.
- Offline reconstruction keeps the preprocessed projection stack in RAM, reconstructs one
  `zChunk` at a time, converts float32→uint16 on the GPU and enqueues TIFF writing as each
  chunk completes.
- Automatic centre-of-rotation estimation (FFT-based, sub-pixel).
- Phase retrieval, flat/dark normalization, log/linear conversion, projection taper and
  Fourier-wavelet stripe removal.

**Workflow**
- Queue-based reconstruction with reordering, removal and import; a stale-job recovery
  routine re-enqueues tasks interrupted by an abnormal termination.
- Three layers of JSON provenance: the scan manifest stored with the projections, the
  reconstruction parameters stored with the slices, and per-task queue state. Reconstruction
  bootstraps from the manifest, so acquisition and reconstruction share one source of truth.
- Headless reconstruction CLI (`ct_reconstruct_headless`) and a pipeline replay tool
  (`ct_pipeline_replay`) for batch and regression work.

## Accompanying paper

The system is described in:

> N. Zhao, K. Du, K. Li, F. Tao, G. Du, and T. Xiao,
> *XStream: an integrated CT acquisition and reconstruction workstation with
> software-defined continuous-rotation timing*, 2026 (submitted; preprint to appear).

Measured against the previous production workflow at the same beamline, the workstation
shortens CT acquisition by 3.4–3.7× (step-and-shoot) and 7.2–17.0× (continuous rotation),
reduces the per-sample storage overhead from 826.8 s to nearly zero, and acquires 20 000-frame
2D time series where the previous software fails at 2 000 frames.

If you use this software in academic work, please cite it — see `CITATION.cff`.

## Requirements

| Component | Notes |
|---|---|
| Windows 10/11 x64 | The workstation is Windows-only (PVCAM, Qt, MSVC) |
| Visual Studio 2022 (MSVC x64) | C++17; CUDA side also C++17 |
| CMake ≥ 3.26 | Presets are provided |
| Qt 6.10 (Widgets) | Built with 6.10.1 msvc2022_64 |
| OpenCV 4.x | 4.12 used in production |
| NVIDIA CUDA Toolkit + cuFFT | Compiled with `--use_fast_math` |
| **PVCAM SDK** (optional) | Teledyne Photometrics; required only for camera acquisition. Without it the build defines no `CT_HAS_PVCAM` and the reconstruction/CLI targets still work. |

## Build

1. Edit the paths in `CMakePresets.json` (`CMAKE_PREFIX_PATH`, `OpenCV_DIR`,
   `CMAKE_MAKE_PROGRAM`) so that they point at your Qt, OpenCV and Ninja installations.
2. Configure and build:

```powershell
cmake --preset vs2022-release
cmake --build --preset build-release --config Release
```

Executables are produced in `build/vs2022-release/`:

- `ct_workstation_app` — the GUI application
- `ct_reconstruct_headless` — headless reconstruction (CLI)
- `ct_pipeline_replay` — replay a recorded acquisition pipeline
- `pvcam_smoke_test`, `pvcam_capture_test` — camera diagnostics (require PVCAM)
- `ct_workstation_tests` — unit tests

> If you change a shared configuration structure (`AppConfig`), do a **clean rebuild**.
> Incremental builds can leave an ABI mismatch that manifests as corrupted runtime values.

## Run

```powershell
# GUI workstation
build\vs2022-release\ct_workstation_app.exe

# Headless offline reconstruction
build\vs2022-release\ct_reconstruct_headless.exe scan_manifest.json ^
    --method offline-fbp --batch 4 --z 16 --tiff-threads 4

# Unit tests
ctest --test-dir build\vs2022-release --output-on-failure -C Release
```

## Configuration

- **Motor controller** — the Kohzu ARIES/TITAN controllers are driven over a plain TCP text
  protocol. Host and port are editable in the *Motor* panel; the default in the source is a
  loopback placeholder. At BL16U2 the controller listens on `10.30.92.201:12321`.
- **Camera** — explicit ROI (`X/Y/Width/Height`, default full frame) and exposure as
  `value + unit` (µs/ms/s). The PVCAM exposure resolution is normalized to milliseconds
  before sequence setup; leaving it implicit can make a sequence set up but stall in
  `EXPOSURE_IN_PROGRESS` until timeout.
- **Reconstruction** — everything (method, chunk size, TIFF thread count, centre-of-rotation
  strategy, preprocessing switches) is stored per task in JSON; see `scan_manifest.json` and
  `recon_params.json` written next to the data.

## Repository layout

```
src/
  app/         application entry point, main window wiring
  acq/         acquisition state machine, CT/fly/2D workers, timing calibration
  camera/      PVCAM camera worker (Kinetix), preview path
  kohzu/       Kohzu ARIES/TITAN TCP text-protocol client
  config/      AppConfig, JSON (de)serialization
  common/      logging, utilities, thread primitives
  cor/         centre-of-rotation estimation
  io/          projection/TIFF I/O
  preprocess/  flat/dark, log/linear, taper, phase retrieval, stripe removal
  recon/       FBP and Gridrec schedulers, real-time and offline paths
  save/        concurrent TIFF slice writer, save-window calibration
  cuda/        GPU kernels, buffers, cuFFT plan cache
  ui/          Qt widgets (preview, histogram, panels)
tests/         unit tests (CTest)
tools/         headless and diagnostic utilities
```

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`. Third-party components (Qt, OpenCV, CUDA,
PVCAM) are **not** bundled and keep their own licenses; see `NOTICE` for details.

## Contributing and customisation

Bug reports and pull requests are welcome. The workstation is deliberately modular so that
beamline-specific behaviour can be added without touching the reconstruction core: acquisition
modes live in `src/acq`, camera-specific code in `src/camera`, motion control in `src/kohzu`,
and the reconstruction pipeline behind a small set of interfaces in `src/recon`.

Groups with requirements that differ from the default configuration are encouraged to fork or
to request a dedicated branch; the acquisition/reconstruction split and the manifest-based
provenance make such customisations straightforward to maintain.
