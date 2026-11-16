# Mosaic 软件验收记录（2026-11-16）

基线先上传至 GitHub `puluotac/XStream` 的 `baseline/20261116-before-mosaic`。
本地基线 48569d9，远程基线 70ad16e，tree 相同；远程使用 GitHub API 创建，提交元数据不同。
开发分支为 `feature/mosaic-local-ct`。

## 已验证

- MSVC Release 全量构建完成，后续增量构建完成。修复了中文 showIncludes 前缀损坏导致的 Ninja 漏头文件依赖；
  `ninja -t deps` 已能列出 MainWindow.h、AppConfig.h、Mosaic.h。
- CTest 三组通过：原有 core、mosaic 数值/磁盘测试、Kohzu 本机 TCP 模拟器。
- GUI 在禁用自动硬件连接的测试模式下启动、显示 Mosaic、正常退出。
- 实际 GUI 队列处理两套各 180 张 64×32 合成投影，自动 COR、Flat-Dark、GPU FBP、float TIFF、自动拼接成功。
  输出 32 层、64×76；恢复位移 (0,-12,0) 体素，NCC 约 0.999984。
- 同一原始投影修改相位恢复 Alpha=10 和 30，新的 tile 目录/最终目录重复重建成功；冻结位置逐值相同。
  合成参数为 18.54 keV、传播距离 0.001 m、像素 1 µm，仅用于软件回归，并非实验参数推荐。
- 最终 float TIFF 类型、尺寸、有限值、非恒定灰度通过；允许并保留负值，未套 uint16 窗口。
- FBP/Gridrec 各自的16位、32位短测成功，输出 TIFF 位深与配置一致。
- 合成 2×2×2 有12条邻接边；增益偏置恢复、闭环一致性、无噪声灰度 RMSE 约 1.0e-7。
- 实际规划器的负坐标、非整数位移、薄板裁剪和内部覆盖测试通过。
- 空白重叠、缺失切片、混合 Alpha/几何/预处理、不一致闭环被拒绝；取消融合留下 COMPLETED=false。
- Kohzu 模拟器检验 605/803 有界重试、失败 WTB 不发 APS、RDP 错位/未知 STR 不放行、Stop 后恢复。

本机结果在 `build/mosaic_final_queue`、`build/mosaic_queue_validation2`、
`build/mosaic_float_smoke`，截图 `build/mosaic_ui.png`；这些生成文件不提交进源码。
可用 `tools/ct_mosaic.cpp` 中测试入口重新生成，不依赖这些本机目录。

## 尚未验收

本轮没有向真实电机发运动指令，也没有启动真实相机采集。
尚未用有光组织数据验证真实坐标标定、局部 CT 截断质量、接缝双影/结构宽度、真实 Alpha 成像质量。
尚未执行 10×8×1 mm 全数据体量的长时间硬件采集、全尺寸拼接吞吐和硬件强关恢复测试。
暗场目前使用用户提供的匹配实测 TIFF；自动遮光拍暗场待线站确认遮光器接口后实现。
这些限制不能用合成测试结果替代，首次有光应按 `mosaic_ct.md` 从 2×1 开始验收。
