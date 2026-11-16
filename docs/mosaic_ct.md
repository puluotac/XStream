# 大视场局部 CT 拼接（Mosaic）

本实现采用 X0/X90 局部步进 CT、Z 分高度、独立重建后拼体数据。
不包含 Y 扩展投影/正弦图拼接，也不是直接复制 Tomosaic 源码。
目前软件和合成数据验收可重复；没有有光组织数据的质量验收结论。

## 数据与操作

左侧在 2D 后增加 Mosaic。曝光、相机 ROI、起始角度、每个 tile 的投影数、采集保存根目录沿用公共控件。
Mosaic 中输入的是样品**毫米尺寸，不是 tile 数量**：对于书页式竖直安装的 10×8×1 mm 脑片，
默认 X0=10、X90=1、Z=8；若竖直方向是 10 mm，应自行交换 X0/Z。
像素尺寸必须采用实际标定值；默认 0.65 µm，不把名义物镜倍率当成标定。

1. 将样品中心对准初始旋转中心，确认样品全角度转动、平场 Y 移出以及各定位轴行程安全。
2. 标定 X0/X90/Z 到当前重建算法切片坐标的方向、旋转角和符号，再勾选 `Axis mapping calibrated`。
   FBP 与 Gridrec 的输出坐标约定可能不同，换算法不能直接复用标定。标定角表示正 X0 电机运动
   在切片 XY 中的方向；规划器对局部物体坐标自动取反。Z sign 同样是正电机方向对应图像方向。
3. 设置 overlap、有效圆直径比例、定位后的 settle 时间，以及每组平场数量和 Y 移出距离。
   规划器使用有效圆内的矩形核心覆盖样品，薄板厚度较小时可减少 X90 位置数。
4. Mosaic 固定使用 Flat 校正。每个 tile 使用自己的平场组，不拍暗场、不需要暗场文件，也不受右侧 Flat-Dark 残留选择影响。
5. Save Path 必须为新目录。界面显示容量估计；本机可查询到余量时会阻止明显空间不足的任务。
6. Start 会快照右侧重建参数。每个 tile 定位后必须独立确认 STR=Idle 且 RDP 到位，随后调用现有步进流程。
   全过程保留脉冲计数位置、目标位置和网格索引；RDP 不是独立编码器测量。
7. 每个 tile 完成后顺序进入现有重建队列，固定输出未窗口化 float32 TIFF。全部成功后自动拼接。
   完成时电机停留在最后一个 tile，不自动回初始位置。

```
样品根目录/
  mosaic_manifest.json         # 规划、轴标定、tile 实际位置和完成情况
  Tile_0001/
    scan_manifest.json
    Tomo_0000001.tif ...
    Flat_0000001.tif ...
    alignment180.tif
    Recon/
      reconstruction.json     # 独立的重建参数快照/任务结果
      recon_params.json
      recon_result.json
      Slice_0000.tiff ...      # 沿用已有局部重建命名
  Tile_0002/ ...
  MosaicRecon/
    reconstruction.json       # 位移、NCC、残差、增益、偏置、输出几何
    Slice_0000001.tif ...      # 最终 float32 拼接体，从 1 开始
```

重新试 Alpha：打开同一 `mosaic_manifest.json`，修改右侧相位恢复参数，设置新的 tile 重建子目录名
和新的最终输出目录名，点击 `Reconstruct all tiles with current parameters`。
存在成功的 `MosaicRecon/reconstruction.json` 时默认复用它的配准坐标；也可指定同一数据集的其他成功结果。
复用会校验源清单指纹，保持位移坐标原值，只重新估计灰度关系。禁止覆盖已有输出。
扫描 JSON 不会因重复重建被改写。队列仍为内存队列，重启后不恢复顺序。

## 算法及边界

- 使用实际脉冲计数和轴标定提供初值，按有效圆掩膜排除无效区域。
- 降采样三维数据：重叠区均值投影的相位相关初值，加有界三维 NCC 搜索。
- 原分辨率重叠小块继续三维 NCC/Pearson 亚体素细化；不把降采样亚像素精度冒充原分辨率精度。
- 同时连接全部 X0/X90/Z 邻接边，做锚定的加权全局平移优化，检查闭环残差；默认超过 1 体素拒绝融合。
- 全局拟合增益/偏置，圆形有效区 feather 融合，按 Z 单层流式写 TIFF。
- 修改 Alpha 后可冻结几何，避免不同参数重建发生配准漂移。
- 低纹理、空白或不一致重叠会报错停止；当前不做非刚性配准、旋转/尺度自校正或自动删除坏边。
- 当前细配准使用重叠中心小块。若真实组织在该处无有效结构，需要增加多候选小块选择，不能降低阈值蒙混通过。
- 预览总体限制 2 GiB，输出单层工作集限制 4 GiB；并非把整个最终体积装进内存。
  TIFF 写出队列限制为一块，重建线程等待写盘，避免磁盘较慢时不断积压 float 体数据。
- 重叠不能消除局部 CT 截断、金属吸收/光子不足、相位恢复边界造成的伪影。
  合成数据通过不代表这些组织成像问题已经解决。

## 可重复测试

使用 VS 开发环境和 CMake preset 构建。preset 为本机中文 MSVC 明确设置 UTF-8 的 `CT_MSVC_INCLUDE_PREFIX`，使 Ninja 的头文件依赖解析稳定。
仅安装其他语言 MSVC 的机器应覆盖为对应语言前缀，或清除此覆盖重新检测。C++/CUDA 主机异常展开使用 `/EHsc`。
旧构建树使用过损坏的中文 showIncludes 前缀时，首次必须 clean 全量构建；否则可能混合旧结构体布局。

```
cmake --preset vs2022-release
cmake --build --preset build-release --target ct_workstation_app ct_mosaic ct_workstation_tests ct_mosaic_motion_tests
ctest --test-dir build/vs2022-release --output-on-failure -C Release
ct_mosaic --self-test 新测试目录
ct_mosaic --make-projections 新合成投影目录
ct_mosaic --stitch mosaic_manifest.json 新输出目录 Recon [先前配准结果.json]
ct_mosaic --check-slices 输出目录 32
```

`CT_UI_SMOKE` 指向截图目录时，软件不自动连接电机，不保存会话参数；正常启动界面后关闭。
同时设置 `CT_MOSAIC_QUEUE_TEST` 为合成投影目录，会读取其中 `test_config.json`，运行实际 GUI 队列及自动拼接。
可用 `CT_MOSAIC_TEST_TILE_RECON`、`CT_MOSAIC_TEST_OUTPUT`、`CT_MOSAIC_TEST_REGISTRATION` 测试再次重建。
这些入口不启动相机采集、不发送真实电机命令。

当前自动测试覆盖：规划范围/有效圆、三维亚体素位移、噪声、低纹理拒绝、闭环冲突、粗到细配准、
2×2×2 体融合、灰度恢复、缺失切片、参数混用、冻结坐标和中途取消。
本机 TCP Kohzu 模拟器覆盖实际 RDP 校验、未知 STR 拒绝、605/803 重试、WTB 失败阻止 APS、Stop 后恢复。

有光时仍需逐级验收：先 2×1，再含 X90/Z 邻居的小网格，最后目标尺寸。
核验单结构无双影、结构宽度保持、接缝灰度/亮暗边、残差、Alpha 冻结坐标；
并实测定位、Y 平场复位、Stop 和异常重启恢复。不要直接用大脑片全任务替代首轮标定。
