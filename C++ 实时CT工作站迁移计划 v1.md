- # C++ 实时CT工作站迁移计划 v1（实施版）

  ## Summary
  在 `E:\2026_Chun\2026_05_26\C++ 实时CT工作站迁移计划 v1` 下新建独立工程，把当前 `v5.0` 的 Python 文件驱动准实时 3D 平行束 FBP 重建工作站迁移为 **Windows-only / NVIDIA-only** 的 **C++17 + CMake + Qt6 Widgets + OpenCV + CUDA + cuFFT** 桌面程序。

  v1 的完成标准固定为：
  - 保留文件夹轮询输入
  - 保留 flat/dark 校正
  - 保留自动 COR，但只实现 **FFT 亚像素法**
  - 保留 Paganin 相位恢复
  - 保留 FBP 预处理与自定义 CUDA 3D BP 核
  - 保留 2D 实时预览、日志、进度条、保存切片
  - 不做相机线程、电机线程、3D 体渲染
  - 不使用 Tomopy，不使用 ASTRA，重建核心全部走自研 CUDA 后端

  ## Implementation Changes
  ### 1. 工程结构与构建
  根目录固定为：
  - `E:\2026_Chun\2026_05_26\C++ 实时CT工作站迁移计划 v1`

  目录结构固定为：
  - `CMakeLists.txt`
  - `CMakePresets.json`
  - `README.md`
  - `cmake/`
  - `resources/`
  - `src/app/`
  - `src/ui/`
  - `src/core/config/`
  - `src/core/io/`
  - `src/core/preprocess/`
  - `src/core/cor/`
  - `src/core/recon/`
  - `src/core/save/`
  - `src/core/common/`
  - `src/cuda/`
  - `tests/`

  构建规则固定为：
  - 生成器：Visual Studio 17 2022
  - 平台：x64
  - C++ 标准：C++17
  - CUDA 标准：与主工程一致，启用 separable compilation
  - Qt：`E:\Qt\6.10.1\msvc2022_64`
  - OpenCV：使用现有 `OpenCVConfig.cmake`
  - 不使用 `.ui` 文件，全部采用 **Qt Widgets 纯代码搭界面**
  - 不引入额外测试框架，使用 `CTest + 自定义测试可执行程序`

  `CMakePresets.json` 固定提供：
  - `vs2022-debug`
  - `vs2022-release`

  ### 2. 目标与模块划分
  工程拆成 3 个目标：
  - `ct_workstation_app`
    - Qt6 主程序
  - `ct_workstation_core`
    - 配置、IO、预处理、COR、保存、重建调度
  - `ct_workstation_cuda`
    - CUDA 核、GPU buffer、cuFFT、重建后端

  Qt 主程序只负责 UI 与线程管理，不直接做算法。

  核心类固定为：
  - `AppConfig`
  - `MainWindow`
  - `ReconstructionController`
  - `ReconstructionWorker`
  - `SaveWorker`
  - `FileProjectionSource`
  - `CorEstimatorFft`
  - `PaganinProcessor`
  - `FbpFilter1D`
  - `GeometryVectorGenerator`
  - `GpuVolumeBuffer`
  - `Bp3dCudaBackend`
  - `ImageNormalizer`
  - `PreviewSliceRenderer`

  ### 3. 线程与运行模型
  v1 固定使用 **QObject + QThread** 模式，不使用 “继承 QThread 写业务逻辑” 的旧写法。

  线程模型固定为：
  - UI 线程
  - Reconstruction 线程
  - Save 线程

  虽然暂不做相机/电机线程，但接口提前按未来扩展留口，不在 v1 中实现。

  对象关系固定为：
  - `MainWindow` 持有 `ReconstructionController`
  - `ReconstructionController` 创建并管理 `QThread`
  - `ReconstructionWorker` 被 `moveToThread()` 到重建线程
  - `SaveWorker` 被 `moveToThread()` 到保存线程
  - 停止控制统一使用 `std::atomic_bool stopRequested`

  ### 4. UI 范围
  Qt6 界面采用“简化但同功能”路线，固定包含：
  - 左侧参数面板
  - 右侧 2D 实时预览
  - 日志窗口
  - 进度条
  - `Start Recon`
  - `Stop`
  - `Save Slices`

  参数分组固定为：
  - IO
  - Geometry
  - Phase
  - Preprocess
  - Compute
  - Preview

  v1 不实现：
  - 3D 体渲染窗口
  - 帮助页
  - Python 版的完整外观还原
  - 高级主题美化

  ### 5. 配置与接口
  `AppConfig` 固定包含 6 个子结构：
  - `GeometryConfig`
  - `PhaseConfig`
  - `PreprocessConfig`
  - `ComputeConfig`
  - `PreviewConfig`
  - `IoConfig`

  字段语义与 Python 版一一对齐，保留：
  - `VOL_W`, `VOL_H`, `TOTAL_ANGLES`
  - `COR_SHIFT`, `COR_AUTO`, `ANGLE_OFFSET_DEG`
  - `BATCH_SIZE`, `Z_CHUNK_SIZE`, `GPU_INDEX`, `Z_BLOCK_SPLIT`
  - `METHOD`, `AVE`, `TAPER_WIDTH`, `FILTER_TYPE`
  - `ENABLE`, `ENERGY_KEV`, `DISTANCE_M`, `PIXEL_SIZE_UM`, `ALPHA_RATIO`
  - `WATCH_DIR`, `SAVE_DIR`
  - `TOMO_PREFIX`, `FLAT_PREFIX`, `DARK_PREFIX`
  - `INPUT_PADDING`, `INPUT_EXT`
  - `FILE_PATTERN_0`, `FILE_PATTERN_180`
  - `FILE_PREFIX`, `SAVE_DEPTH`, `SAVE_FMT`
  - `LIVE_SLICE_INDEX`, `ENABLE_LIVE`

  v1 配置文件格式固定为 JSON，启动时读，保存时写。

  ### 6. 文件驱动输入链
  `FileProjectionSource` 的行为固定为：
  - 轮询 `WATCH_DIR`
  - 按 prefix + padding + ext 规则解析文件名
  - 支持 flat/dark/tomo/0°/180° 参考图读取
  - 对正在写入中的文件做稳健读取重试
  - 批量返回当前可处理投影集合
  - 不使用文件系统 watcher
  - 不做相机 SDK 封装

  OpenCV 图像读取规则固定为：
  - 使用 `cv::imread(..., cv::IMREAD_UNCHANGED)`
  - 内部统一转 `float32`
  - v1 默认只支持单通道灰度投影

  ### 7. 自动 COR
  v1 自动 COR 固定只保留 **FFT 亚像素法**，对应 Python 当前主用的 FFT/Guizar-Sicairos 路线。

  固定流程：
  - 读 `0°` 图与 `180°` 图
  - 根据 correction mode 做 flat/dark 校正
  - 做 `-log`
  - `180°` 图左右翻转
  - 裁掉 margin
  - FFT 相位相关求整像素峰
  - 在峰值附近做亚像素细化
  - 输出 `shift_x / 2.0f`

  v1 不实现：
  - Parabolic 法
  - Search 法

  ### 8. 预处理与 FBP
  预处理顺序固定为：
  1. flat/dark 归一化
  2. 可选 Paganin
  3. `log` 或 `linear`
  4. taper window
  5. detector `u` 方向 1D FBP 滤波

  `PaganinProcessor` 固定行为：
  - 使用 cuFFT
  - 以 `(H, W)` 为单位缓存频域核
  - 支持 batch 图像输入
  - 参数来自 `PhaseConfig`

  `FbpFilter1D` 固定行为：
  - 只对 detector 横向做频域滤波
  - 默认 `shepp-logan`
  - 支持 `ram-lak / shepp-logan / cosine / hamming / hann`
  - 数值顺序与 Python 版保持一致

  ### 9. 几何与重建后端
  `GeometryVectorGenerator` 固定复刻 Python `precompute_vectors()` 公式，向量格式保持与当前自定义核一致。

  GPU 数据布局固定为：
  - batch 输入逻辑：`(H, N, W)`
  - z-chunk 输入：`(z_chunk, N, W)`
  - chunk 输出：`(z_chunk, W, W)`
  - 全局 volume：`(H, W, W)`，`float32`

  `GpuVolumeBuffer` 固定为 **GPU 常驻主累加区**：
  - 整个 3D volume 在显存中分配
  - chunk 重建后直接在 GPU 端累加
  - 预览时仅回传单层切片
  - 保存时分块拷回 CPU

  `Bp3dCudaBackend` 固定采用 **原生 CUDA 直迁**：
  - 直接迁移 `custom_recon.py` 中 `bp3d_final` 的计算逻辑
  - 不先走 ASTRA 过渡
  - CUDA 核定义放在 `src/cuda/bp3d_kernel.cu`
  - 主机封装放在 `src/core/recon/Bp3dCudaBackend.*`

  ### 10. 预览与保存
  实时预览固定为：
  - 每处理完一批角度，取 `LIVE_SLICE_INDEX`
  - 从 GPU volume 拷回单张切片
  - 映射为 `QImage::Format_Grayscale8`
  - 用 `QLabel + QPixmap` 显示
  - v1 不实现复杂缩放视图组件

  保存固定为：
  - 保存线程独立执行
  - 支持 `8bit / 16bit / 32bit`
  - 支持 `tif / tiff`
  - 保存时按 z 分块从 GPU 拉回 CPU
  - 同目录写出 `recon_params.json`

  ## Public Interfaces / Types
  以下接口固定，不留给实现者二次决策：

  - `class ReconstructionController`
    - `void startReconstruction(const AppConfig&)`
    - `void stopReconstruction()`
    - `void startSave(const SaveRequest&)`

  - `class ReconstructionWorker : public QObject`
    - `public slots: void process()`
    - `public slots: void requestStop()`
    - `signals: void logMessage(QString)`
    - `signals: void progressUpdated(int current, int total)`
    - `signals: void previewReady(QImage)`
    - `signals: void corComputed(double)`
    - `signals: void reconstructionFinished()`
    - `signals: void reconstructionFailed(QString)`

  - `class SaveWorker : public QObject`
    - `public slots: void process()`
    - `public slots: void requestStop()`

  - `class FileProjectionSource`
    - `bool waitForReferenceImages(...)`
    - `std::vector<ProjectionFrame> pollNextBatch(...)`

  - `class CorEstimatorFft`
    - `float estimateShift(const cv::Mat& img0, const cv::Mat& img180, const CorrectionContext&)`

  - `class PaganinProcessor`
    - `void configure(const PhaseConfig&)`
    - `void apply(GpuImageBatch&)`

  - `class FbpFilter1D`
    - `void configure(int detWidth, FilterType)`
    - `void apply(GpuSinogramBatch&)`

  - `class Bp3dCudaBackend`
    - `void initialize(const GeometryConfig&, const ComputeConfig&)`
    - `void backProjectChunk(const GpuSinogramChunk&, const std::vector<double>& vectors, int zOffset, GpuVolumeBuffer&)`

  ## Test Plan
  ### 1. 构建与链接
  - Debug/Release 都能在 Visual Studio 2022 直接生成。
  - CMake 能正确找到 Qt6、OpenCV、CUDA、cuFFT。
  - `ct_workstation_app`、`ct_workstation_core`、`ct_workstation_cuda` 三个目标能完整链接。

  ### 2. 单元与数值测试
  - 文件命名解析与 padding 规则正确。
  - 几何向量与 Python 同参数输出一致。
  - COR FFT 法在人工平移样本上误差不超过 `0.1 px`。
  - Paganin 对同一输入重复调用时命中核缓存。
  - FBP filter 的 padding 长度与 Python 一致。
  - CUDA 反投影在小体积测试数据上输出 shape 与 Python 一致。

  ### 3. 集成测试
  - 用现有 `v5.0` 的一组真实投影目录跑完整流程。
  - 能等待 flat/dark，能读取参考图，能自动 COR。
  - 能持续更新预览层。
  - 能中途停止。
  - 能保存切片结果。
  - 与 Python 版同一参数、同一数据比较中心切片，`NRMSE <= 1e-3`。

  ### 4. 失败场景
  - 缺 flat/dark
  - 缺 0°/180°
  - 普通投影命名不连续
  - 图像尺寸与配置不一致
  - 用户点击停止
  - GPU 内存不足

  这些场景都必须：
  - 不崩溃
  - 日志可读
  - 线程能安全退出

  ## Assumptions
  - 平台固定为 Windows 10/11 + NVIDIA GPU。
  - 编译器固定为 Visual Studio 2022 / MSVC x64。
  - 只做文件驱动输入，不接相机，不接电机。
  - 只做 Qt Widgets，不做 QML。
  - 只保留一个自动 COR 方法：FFT 亚像素法。
  - 第一版必须包含 Paganin。
  - 3D 主 volume 常驻 GPU。
  - 第一版不做 3D 体渲染。
  - Python `v5.0` 是唯一行为与数值基线。