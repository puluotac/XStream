# Engineering notes

Notes for maintainers and for users who want to develop XStream further. They record design
decisions and pitfalls that are easy to lose in commit messages.

## Camera integration (PVCAM / Kinetix)

- **Exposure resolution must be explicit.** PVCAM normalizes `PARAM_EXP_RES` internally.
  If the exposure resolution is left implicit, sequence setup can succeed while acquisition
  stays in `EXPOSURE_IN_PROGRESS` until it times out. The camera worker therefore selects an
  explicit resolution and converts the UI exposure (value + unit) to it before
  `pl_exp_setup_seq`, and logs the selected mode and resolution.
- **Hot-open session.** The camera is opened once after the worker thread starts and kept
  open across Live/Capture runs; only the sequence is torn down between runs. The camera is
  closed on shutdown, or on error so that the next run can reopen cleanly.
- **Preview throttling.** A full 3200 × 3200 frame is far too large to emit to the UI at
  every exposure. Preview signals are throttled before the UI signal path, while the capture
  queue still receives every frame. Any change to the preview path must preserve this
  separation, otherwise the event queue backs up under sustained acquisition.

## Data path

- Two-tier memory: a driver-side ring buffer feeds a reusable frame pool; downstream stages
  exchange frames through bounded queues by transferring pointer ownership, so pixel data are
  copied exactly once (driver buffer → frame pool). Keep this property when adding stages.
- The frame pool exists to avoid per-frame allocation and GPU memory fragmentation; its
  capacity is independent of the number of frames acquired.

## Reconstruction

- `ReconMethod` has three values: `RealTimeFBP`, `OfflineFBP` and `OfflineGridrec`. A JSON
  configuration without `IO.RECON_METHOD` defaults to `RealTimeFBP`.
- Offline FBP and Gridrec deliberately do **not** use the real-time full-volume CPU path.
  They keep the preprocessed projection stack `[angle, z, x]` in RAM, reconstruct one
  `zChunk` at a time, convert float32 → uint16 on the GPU, and enqueue TIFF writing as each
  chunk completes. This decouples GPU memory usage from the full volume size.
- The save window (grayscale min/max) is calibrated once from the central `zChunk` using
  fixed percentiles and then reused for the whole dataset, so that slice contrast does not
  drift between layers in third-party viewers.
- Phase retrieval is currently applied in the preprocessing stage that runs before the
  projection stack is assembled; the offline FBP/Gridrec paths do not apply it a second time.

## Build pitfalls

- **Clean rebuild after changing a shared config structure.** An incremental Release build
  after adding a field to `AppConfig` once produced an ABI mismatch: different translation
  units disagreed on the structure layout, the process reported inconsistent values and
  eventually crashed on exit. If `AppConfig` (or any other widely included structure)
  changes, do a clean rebuild before trusting smoke results.
- Executables are emitted directly into `build/vs2022-release/` by the Ninja presets. If
  expected diagnostics are missing from the log, check that the executable being run is the
  one just built and not a stale copy from an older generator layout.

## Threading model

Long-running work (reconstruction, TIFF writing, preview, camera, motion control, pipeline
replay) runs in dedicated `QObject` workers moved to their own threads. Shutdown is
coordinated through a single atomic stop flag per worker. Any new worker should follow the
same pattern and must not block the UI thread.
