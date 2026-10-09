# giaok_write：少样本中文手写生成

最终目标：输入中文文本和少量参考手写，生成内容正确、风格接近参考者的图像。
当前阶段：建立中文单字生成基线与公共多书写者底座。个人微调、整行布局暂不开展。

## 项目导航

- [第一阶段流程](docs/PHASE1_GUIDE.md)：任务、数据、模型、训练与验收。
- [基线目标](docs/BASELINE_GOAL.md)：这轮实验的范围与完成条件。
- [论文索引](papers/README.md)：前面讨论的论文及下载状态。
- [预训练推理记录](experiments/baseline_001/REPORT.md)：实际环境、命令、结果和限制。
- [短训练记录](experiments/training_demo_001/REPORT.md)：从随机初始化的官方示例训练及采样。
- [环境说明](docs/ENVIRONMENT.md)：依赖兼容与复现。
- `vendor/FontDiffuser/`：按固定提交下载的官方代码。
- `scripts/`：资源下载和可重复运行的实验入口。
- `checkpoints/`：模型权重，不纳入 Git。

## 快速运行

```bash
.venv/bin/python scripts/run_baseline.py --config configs/baseline_001.json
```

先阅读实验报告中的资源与环境要求。第一轮是官方预训练模型的本地推理复现；它不能代表已经在 CASIA 上重新训练完成，也不能证明新用户的个人风格迁移质量。

原有 `main.py` 保留，新的实验入口位于 `scripts/`。

## 初步生成结果

8 个中文目标字 × 3 种官方风格参考，96×96，20 步，RTX 5070 实测平均约 0.657 秒/字。

![预训练基线对照](experiments/baseline_001/contact_sheet.png)

第一列为目标模板，顶部为参考图片，右侧为模型生成结果。

## 短训练验证

额外从随机初始化，用官方 12 张示例训练了 1000 步，完成保存和重新加载。损失下降，但采样没有形成稳定字形；详情及失败样例见 [短训练报告](experiments/training_demo_001/REPORT.md)。完整真实手写底座仍需 CASIA 数据训练。

```bash
bash scripts/run_training_demo.sh
```
