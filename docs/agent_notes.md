# Agent Notes

This file is for future Codex sessions and maintainers. It records the engineering context that is easy to lose in normal commit messages.

## Current Stable Point - 2026-06-23

Repository root on the beamline workstation:

```text
E:\2026_Chun\2026_06_09\CT_stream_v1
```

The application is a Windows C++/Qt6/CUDA CT workstation for BL16U2 work. It currently combines:

- File-driven real-time FBP reconstruction.
- Offline FBP reconstruction.
- Offline Gridrec reconstruction.
- Kinetix/PVCAM camera capture and live preview.
- Kohzu ARIES motor control over TCP/IP.

## Build Environment

- Windows, MSVC / Visual Studio 2022 Build Tools.
- Qt 6.10.1 at `E:\Qt\6.10.1\msvc2022_64`.
- OpenCV runtime path used during tests: `E:\OpenCV\build\x64\vc16\bin`.
- CUDA and cuFFT installed through NVIDIA CUDA Toolkit.
- PVCAM SDK expected by CMake at `C:/Program Files/Photometrics/PVCamSDK`.

Release build command used successfully:

```powershell
& cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && cmake --build build\vs2022-release --config Release'
```

Clean rebuild command used after changing shared config structures:

```powershell
& cmd.exe /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && cmake --build build\vs2022-release --config Release --target clean && cmake --build build\vs2022-release --config Release'
```

Test command:

```powershell
$env:PATH='E:\OpenCV\build\x64\vc16\bin;' + $env:PATH
ctest --test-dir build\vs2022-release --output-on-failure -C Release
```

## Important Implementation Notes

- `ReconMethod` has three modes: `RealTimeFBP`, `OfflineFBP`, and `OfflineGridrec`.
- Old JSON without `IO.RECON_METHOD` defaults to `RealTimeFBP`.
- Offline algorithms intentionally do not use the real-time full CPU volume path.
- Offline FBP/Gridrec keep a preprocessed `projectionStack [angle,z,x]` in CPU RAM, reconstruct one `zChunk` at a time, and enqueue TIFF saving as chunks complete.
- Offline FBP/Gridrec currently ignore `PHASE.ENABLE`. Phase retrieval should be handled in camera-side preprocessing or by pointing `WATCH_DIR` to preprocessed projection images.
- `RAW Volume` output was intentionally removed from the product UI/config path. Production output is concurrent 16-bit TIFF slices.
- `IO.TIFF_THREADS` controls TIFF writer thread count. UI range is `1..64`, default `16`.
- Offline FBP/Gridrec use GPU-side `float32 -> uint16` conversion through `launchConvertFloatVolumeToUint16()` before downloading chunk data to CPU.
- Save window is calibrated from the center zChunk using fixed percentiles, then reused for the whole dataset. This avoids per-slice contrast changes in Avizo.
- Kinetix preview is intentionally throttled before the UI signal path. Full 3200x3200 frames are too large to emit/render at every exposure; capture queues still receive every non-preview frame, while the UI receives enough frames for live display.
- PVCAM internal capture now normalizes `PARAM_EXP_RES` to millisecond units before `pl_exp_setup_seq` and converts UI exposure milliseconds to the active PVCAM units. On the beamline Kinetix, leaving exposure resolution implicit can make sequence acquisition set up successfully but stay in `EXPOSURE_IN_PROGRESS` until timeout. The worker also logs the selected exposure mode and resolution.
- Kinetix exposure UI is `value + unit` (`us/ms/s`). The worker preserves physical exposure time by selecting the requested PVCAM exposure resolution when available, otherwise falling back to milliseconds and converting units. Live startup currently still opens/configures/closes the camera per run; vendor software feels instant because it keeps the camera opened and preconfigured. A future fast-live pass should introduce a persistent camera session/state machine instead of repeating `pl_pvcam_init/pl_cam_open/pl_exp_setup_seq` for every Live click.

## Known Pitfall

After adding `AppConfig::io.tiffThreads`, an incremental Release build produced an ABI mismatch: headless printed `tiffThreads=4`, but offline reconstruction received `1` or writer logged `64`, and the process exited with `0xC0000005` after finishing. A clean Release rebuild fixed this. If a shared config structure changes again, prefer a clean rebuild before trusting smoke results.

The old nested Debug executable can remain at `build\vs2022-debug\Debug\ct_workstation_app.exe` from an older Visual Studio-generator build. Current CMake presets generate Ninja executables directly at `build\vs2022-debug\ct_workstation_app.exe` and `build\vs2022-release\ct_workstation_app.exe`. If camera logs do not show `App executable`, `Build stamp`, or PVCAM exposure-resolution diagnostics, the user is likely running an obsolete exe path.

## Smoke Verification From 2026-06-23

Commands:

```powershell
$env:PATH='E:\OpenCV\build\x64\vc16\bin;' + $env:PATH
build\vs2022-release\ct_reconstruct_headless.exe E:\2026_Chun\2026_06_09\ct_stream_smoke_data\smoke_fbp.json --method offline-fbp --batch 4 --z 16 --tiff-threads 4
build\vs2022-release\ct_reconstruct_headless.exe E:\2026_Chun\2026_06_09\ct_stream_smoke_data\smoke_gridrec.json --method offline-gridrec --batch 4 --z 16 --tiff-threads 4
```

Results:

- Offline FBP smoke exited with code `0`.
- Offline Gridrec smoke exited with code `0`.
- `recon_params.json` wrote `"TIFF_THREADS": 4`.
- `SAVE_MODE` was not written.

## Hardware Context

Kohzu ARIES controller:

- Host: `10.30.92.201`
- Port: `12321`
- Axis mapping used in UI:
  - `U`, `V`, `Y`, `X0`, `X90`, `R`, `Z`
  - Detector axes: `Det_OT2_Z`, `Det_OT2_Y`

Kinetix camera:

- Integrated through PVCAM SDK.
- Main program supports capture/live preview controls and a frame queue structure.

## Next Work Candidates

- Benchmark TIFF writer thread counts on the new SSD with 3200x3200 data: `8`, `16`, `24`, `32`, possibly `48`.
- Consider using OpenCV direct `imwrite` or a dedicated TIFF library if `imencode + QFile` becomes the bottleneck.
- Consider keeping GPU uint16 conversion and adding pinned host output buffers for chunk downloads if PCIe transfer becomes measurable.
- Do not reintroduce RAW output unless the user explicitly asks; it was removed because multi-threaded TIFF was more practical for current Avizo workflow and SSD behavior.
