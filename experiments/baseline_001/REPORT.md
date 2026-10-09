# B0：FontDiffuser 官方预训练推理结果

运行日期：2026-10-09。状态：完成。

## 本轮结论

中文单字生成闭环已跑通。8 个目标字 × 3 张官方风格参考，共 24 张对照结果；另有 1 张官方示例与 1 张重复性检验，共 26 张生成图。

从拼图可观察到明显的风格响应：同一个字，在笔刷感、细线条、粗几何三类参考下产生不同字形。目标字整体可辨认；尚未进行独立 OCR、细笔画人工评分或真实书写者测试，不能将本轮视为 100% 字符正确率或个人手写迁移验证。

## 配置与来源

- 官方代码提交：`7b28ce9c3b357f4fb23296622f458cf169803539`。
- 权重源：作者 Hugging Face Space `yeungchenwa/FontDiffuser-Gradio`。
- 权重 revision：`7bd89f3932f30e0b0062b4ba852a4270a20fd0e4`。
- UNet、内容编码器、风格编码器均 strict=True 加载，SHA-256 与官方 LFS 对应值一致。
- 作者 Google Drive 的 UNet 与风格编码器下载副本校验相同。
- 配置：96×96，FP32，DPM-Solver++，20 步，CFG=7.5，起始 seed=123。
- 内容模板：系统 Noto Serif CJK，简体中文 index=2。
- 字符：湖、海、明、好、我、清、永、鑫。
- 参考是官方字体/书法样例，并非本次用户手写样本。

## 实测

| 项目 | 结果 |
|---|---|
| GPU | RTX 5070 12GB |
| 24 张对照图平均采样耗时 | 0.657 秒/字 |
| 采样耗时范围 | 0.599–0.783 秒/字 |
| 官方示例首次采样 | 1.333 秒 |
| 模型加载 | 1.235 秒 |
| PyTorch 峰值 allocated 显存 | 550.15 MiB |
| PyTorch 峰值 reserved 显存 | 630 MiB |
| 同输入、同 seed 重复输出 | 像素完全一致 |

计时包含 GPU 同步后的采样，不包含下载与落盘；显存是本进程 PyTorch allocator 记录，不等于整张 GPU 的总占用。首个官方样例包含冷启动成本。

## 输出位置

- `contact_sheet.png`：目标模板与三组生成结果。
- `inputs/`：本轮使用的字形模板、参考和官方输入原图。
- `outputs/`：26 张原始生成图片。
- `samples.jsonl`：每个对照样例的耗时、seed、路径和 SHA-256。
- `metrics.json`：机器可读结果，未测指标为 null。
- `manual_review.csv`：后续人工逐字审核表；保留未评分状态。
- `config.json`、`environment.json`、`pip_freeze.txt`、`nvidia_smi.txt`：环境与配置。
- `source_provenance.json`、`weight_manifest.json`：资源来源与校验。
- `run.log`：成功运行日志。
- `attempt_01_cuda_init.log`：初次运行失败，已通过显式初始化 CUDA 修复。

## 复现命令

```bash
cd /home/giaok/PycharmProjects/giaok_write
.venv/bin/python scripts/run_baseline.py --config configs/baseline_001.json
```

## 限制与下一训练里程碑

这轮完成的是官方预训练字体模型的本地推理基线，没有在本机训练出完整通用手写底座。短训练链路验证单独保存在 `../training_smoke_001/`。

接下来应获取 CASIA-HWDB1.1，保留 writer_id 并固定书写者划分，在真实手写数据上训练/适配公共底座。个人用户微调和整行生成留到后续。不能用本轮 24 张字体对照样例估计真实书写者准确率、FID/HWD 或宣称复现论文 SOTA。
