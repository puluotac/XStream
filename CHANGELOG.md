# Changelog

## 2026-06-23

- Imported the CT Stream workstation source tree for Git-based version control.
- Current application includes Qt/CUDA real-time FBP, offline FBP, offline Gridrec, Kinetix/PVCAM camera controls, and Kohzu ARIES motor controls.
- Offline FBP/Gridrec production path saves concurrent 16-bit TIFF slices. RAW Volume output was removed from the product UI/config path.
- Added `IO.TIFF_THREADS` and the right-side `TIFF Threads` UI control for configurable TIFF writer concurrency.
- Offline FBP/Gridrec convert reconstructed float32 chunks to uint16 on GPU before device-to-host transfer.
- Fixed offline TIFF writer shutdown so repeated `finish()` calls do not cause a completion-time crash.
- Verified after clean Release rebuild: `ctest --output-on-failure -C Release`, offline FBP smoke, and offline Gridrec smoke passed with `--tiff-threads 4`.
