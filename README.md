# CT Stream Workstation

Windows C++/Qt6/CUDA workstation for BL16U2-style parallel-beam CT acquisition,
hardware control, preview, and reconstruction.

## Stack

- Visual Studio 2022 / MSVC x64
- CMake
- Qt6 Widgets
- OpenCV
- CUDA + cuFFT
- PVCAM SDK for Kinetix camera integration
- Kohzu ARIES/LYNX motor control over TCP/IP

## Configure

```powershell
cmake --preset vs2022-release
cmake --build --preset build-release --config Release
ctest --test-dir build/vs2022-release --output-on-failure -C Release
```

## Current Scope

- Folder-based projection input for reconstruction.
- Real-time FBP path with CPU full volume + GPU slab processing.
- Offline FBP and Offline Gridrec production paths.
- Flat/dark correction, log/linear preprocessing, taper, FFT-based COR.
- Kinetix/PVCAM capture and live preview controls.
- Kohzu ARIES motor UI for sample and detector axes.
- Concurrent 16-bit TIFF slice output with configurable writer thread count.

Large CT datasets, build products, deployed DLLs, and reconstruction outputs are
intentionally excluded from Git. See `docs/agent_notes.md` for current engineering
state and handoff notes.
