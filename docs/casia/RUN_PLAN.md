# CASIA 首轮训练计划

日期：2026-10-09。实验：`casia_001`。

## 本轮任务

下载 CASIA-HWDB1.1 的完整训练和测试压缩包，严格解析 GNT，按书写者划分数据，在公共 FontDiffuser 权重基础上训练一个共享的中文手写模型，最后做同条件训练前后对照。

本次不是对某个用户做个人适配，也不是从随机初始化重训论文的全部训练周期。5000 步为首轮真实数据实验预算；是否达到质量目标由生成和独立测试决定，不能只凭训练损失。

## 来源

官方说明与下载页：

- https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
- https://nlpr.ia.ac.cn/databases/handwriting/Download.html

多次直连官方 HTTP/HTTPS 超时，实际下载使用公开传输镜像：

- https://github.com/AND-Q/HWDB1.1-/releases/tag/datasets

镜像不是 CASIA 官方分发来源。校验值来自该 GitHub Release 的 SHA-256，不能宣称已经与官方源端校验值比对。下载后又对 GNT 格式、样本数、字符类数及书写者划分进行独立校验。下载来源、大小和校验值见实验 `download_manifest.json`。

CASIA 权利仍属于原提供方；镜像不赋予额外使用权。官方页面提供使用申请与协议，本次按研究用途组织实验，不替用户签署协议。原始数据、真实参考与真实目标图仅存本地，GitHub 提交转换代码、统计、生成图与报告。

## 数据

| 集合 | 书写者 | 样本 |
|---|---:|---:|
| train | 216 | 807,972 |
| validation | 24 | 89,786 |
| test | 60 | 223,991 |
| 合计 | 300 | 1,121,749 |

训练压缩包原有 240 位，随机留出 24 位作为验证；官方 60 位测试书写者不参与训练。类别共 3755，全部以中文 Unicode 标签读取。

发现 25 位书写者存在重复字符，共 308 条超出首条的重复标签记录。保留原始记录，风格参考抽样必须与目标字符不同；不能仅依靠样本索引不同。

处理：保持原始灰度与宽高比例，最长边缩放至 80 像素，居中补白至 96×96。存储为每书写者一个 uint8 NumPy 图像数组及 codepoint 数组，使用内存映射读取，避免创建百万个小文件。

训练镜像 ZIP 嵌套 ALZ，使用固定版本 `kippler/unalz` 本地编译解包（zlib License）。工具提交：`457884d21962caa12085bfb6ec5bd12f3eb93c00`。源码未修改，Linux 构建使用 libc iconv，不安装系统服务。

## 模型与训练

- 官方 FontDiffuser 架构不改动，内容编码器、风格编码器和 U-Net 全部参与共享训练。
- 初始权重来自已经运行通过的官方预训练字体模型。
- 分辨率 96，FP32，batch=2，5000 次优化更新。
- AdamW，起始最大学习率 1e-5，100 步 warmup，余弦衰减到 1e-6。
- CFG 条件丢弃概率 0.1。
- 损失：扩散 MSE + 0.01×VGG16 感知损失 + 0.5×offset 损失。
- 清晰图像预测从噪声预测通过 DDPM 公式向量化恢复，与 scheduler.step 的 pred_original_sample 数值比对。
- 每 500 步固定验证；每 2500 步保存三模块和恢复状态；验证改善时另存 best。
- 暂不加入 SCR、OCR 训练损失、个人 LoRA、整行生成。

5000×2=10,000 次训练样本暴露，不等于遍历全部 807,972 张训练图。日志标注步数和已完成 epoch，不将首轮运行描述为完整收敛。

## 检查

1. GNT 标签字节序、位图尺寸与截断处理测试。
2. 书写者集合无交叉，参考不含目标字符。
3. 固定验证噪声和时间步，比较训练前后损失。
4. 提前固定 32 个验证案例、64 个随机测试案例，以及 3 位测试书写者的常用字展示。
5. 同目标、同参考、同随机种子、同 20 步 DPM-Solver++ 和 CFG=7.5 比较训练前后生成。
6. 使用独立 RapidOCR/PP-OCRv4 检查内容，并同时测量它在真实手写上的识别率；这是辅助指标，不能替代人工笔画审核。
7. 原始数据可视化放 `private/`，公开图库只放生成图。

## 复现

```bash
python3 scripts/download_casia.py
bash scripts/extract_casia_train.sh
.venv/bin/python scripts/prepare_casia.py --workers 4
.venv/bin/python scripts/build_casia_evaluation.py
.venv/bin/python scripts/evaluate_casia.py --checkpoint checkpoints/fontdiffuser --label before
OMP_NUM_THREADS=8 .venv/bin/python scripts/train_casia.py --config configs/casia_001.json
.venv/bin/python scripts/evaluate_casia.py --checkpoint experiments/casia_001/checkpoints/best --label after
```

以上命令适用于新检出且没有实验日志的目录。同名评估和非恢复训练会拒绝已有日志，避免混写。本轮数据准备/评估/汇总脚本固定使用 `casia_001`；开始新实验需同时调整这些输出路径与训练配置，不能只修改 `output_dir`。恢复入口：`scripts/train_casia.py --resume <本地恢复状态>`。恢复参数和 RNG，但 DataLoader 从新的打乱顺序开始，尚不保证逐样本完全一致的中断恢复。

完成后先执行 `.venv/bin/python scripts/check_casia_reference_sensitivity.py` 固定噪声换参考诊断，再执行 `.venv/bin/python scripts/summarize_casia.py` 和 `.venv/bin/python scripts/verify_casia_results.py`，生成对照、曲线、配对统计和完整性检查记录。人工审核 CSV 中空白表示没有进行该项审核，不能当作通过。
