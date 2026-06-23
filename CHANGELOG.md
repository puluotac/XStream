# Changelog

## 2026-06-23

- Imported the CT Stream workstation source tree for Git-based version control.
- Current application includes Qt/CUDA real-time FBP, offline FBP, offline Gridrec, Kinetix/PVCAM camera controls, and Kohzu ARIES motor controls.
- Offline FBP/Gridrec production path saves concurrent 16-bit TIFF slices. RAW Volume output was removed from the product UI/config path.
- Added `IO.TIFF_THREADS` and the right-side `TIFF Threads` UI control for configurable TIFF writer concurrency.
- Offline FBP/Gridrec convert reconstructed float32 chunks to uint16 on GPU before device-to-host transfer.
- Fixed offline TIFF writer shutdown so repeated `finish()` calls do not cause a completion-time crash.
- Verified after clean Release rebuild: `ctest --output-on-failure -C Release`, offline FBP smoke, and offline Gridrec smoke passed with `--tiff-threads 4`.
- Fixed Kinetix live/capture preview initialization: the first camera frame now rebuilds the preview histogram/window from real image data, logs frame metadata, and UI preview signals are throttled to avoid full-frame event backlog.
- Fixed Kinetix PVCAM sequence timeout by normalizing `PARAM_EXP_RES` to millisecond exposure units before `pl_exp_setup_seq`; added exposure mode/status diagnostics for future readout timeouts.
- Added a Kinetix exposure unit selector (`us/ms/s`) next to the exposure value and logs PVCAM setup/first-frame timing for Live startup diagnosis.
- Added Kinetix hot-open camera sessions: the app warms the PVCAM camera on startup, keeps the camera open between Live/Capture runs, and releases it on shutdown or camera error.
- Simplified Kinetix UI for production setup: explicit `X/Y/Width/Height` ROI, `Apply ROI`, `Reset Full`, one-button Live toggle, and single-frame TIFF `Snap`.
