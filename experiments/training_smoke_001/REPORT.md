# B0.1：20 步训练链路验证

状态：完成。日期：2026-10-09。

官方 FontDiffuser 的基础训练入口，从随机初始化完成 20 次优化更新，batch=1，FP32，学习率 1e-4，seed=123。数据来自官方示例：3 种字体 × 氮/潮/舶/镀 4 个字，共 12 张目标图；每次参考取同字体的其他字。

损失由扩散 MSE、VGG16 感知损失和 offset 损失构成，沿用官方基础训练代码；不启用 SCR 风格对比阶段。

保存 `global_step_20/` 下 UNet、内容编码器、风格编码器以及官方完整模型文件。原始损失见 `fontdiffuser_training.log`，提取值见 `metrics.json`，TensorBoard 见 `logs/`。

这项实验只确认前向、反向、参数更新及检查点保存可运行。20 步不代表收敛，12 张字体图也不代表真实多书写者训练。

复现：在项目根目录执行 `bash scripts/run_training_smoke.sh`。
