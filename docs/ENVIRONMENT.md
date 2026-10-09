# 实验环境与复现

当前项目 `.venv` 使用 Python 3.12.14。原有 `main.py` 不改动。

## 实际采用

- NVIDIA RTX 5070，驱动 595.91.07。
- PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130。
- diffusers 0.22.0、transformers 4.44.2、accelerate 0.23.0、huggingface-hub 0.25.2。
- 本机已有兼容 GPU 的 PyTorch；独立复制到项目环境，其他项目未修改。
- 精确安装来源记录在 `experiments/baseline_001/runtime_provenance.json`，完整依赖见同目录 `pip_freeze.txt`。

官方仓库原建议 Python 3.9 / Torch 1.13.1 / CUDA 11.7，不能直接用于这次 RTX 5070 环境。transformers 4.33.1 的 tokenizers 0.13.3 在 Python 3.12 上尝试源码构建，本轮改用 4.44.2 对应二进制包。模型架构和官方代码未改动。

## 本机再次运行

```bash
cd /home/giaok/PycharmProjects/giaok_write
.venv/bin/python scripts/run_baseline.py --config configs/baseline_001.json
bash scripts/run_training_smoke.sh
bash scripts/run_training_demo.sh
```

同名输出会更新。保留实验时先复制配置，改 experiment 与 output_dir。

## 重新下载资源

```bash
python3 scripts/download_resources.py sources
python3 scripts/download_resources.py weights
python3 scripts/download_resources.py papers
```

代码和权重固定已验证提交。已存在文件会复用；权重仍按源端 SHA-256 检查。下载使用 curl，PDF 校验需要 pdfinfo / pdftotext。

## 新机器

1. 创建 Python 3.12 环境：`uv venv --python 3.12 .venv`。
2. 安装与硬件兼容的 PyTorch/torchvision CUDA 二进制包；本轮完整版本见 requirements-baseline.txt。精确复现需相同版本，不同 GPU 应重新测量。
3. 安装其余锁定依赖。Torch 的 CUDA 后缀包需要 PyTorch 官方对应索引，不直接依靠普通 PyPI 解析。
4. 下载代码与权重，确认字体路径。配置默认使用系统 Noto Serif CJK TTC 的简体中文 index=2。
5. 执行基线脚本。脚本严格加载 state_dict；版本不兼容时应记录错误，不静默忽略权重。

CASIA-HWDB1.1 已下载并转换，首轮公共数据训练见 [CASIA 报告](../experiments/casia_001/REPORT.md)。个人样本尚未配置。短训练会首次下载约 528 MiB 的官方 VGG16 感知损失权重到 torch 缓存。

绘图使用 matplotlib 3.9.4，并保持 numpy 1.26.4。安装新包时需保留现有版本约束。1000 步训练命令设置 OMP_NUM_THREADS=8；原官方代码与损失保持不变。

## CASIA 环境增量

识别辅助评估使用 rapidocr-onnxruntime 1.4.4、onnxruntime 1.31.0 和 OpenCV 4.11.0.86；实际环境保留 opencv-python-headless 4.10.0.84。完整包列表更新于根目录 `requirements-lock.txt`。Torch 的 `+cu130` 后缀按实际运行版本记录，安装仍需相应 CUDA 索引。

数据下载、ALZ 解包、转换、训练及同条件评估入口见 [CASIA 运行计划](casia/RUN_PLAN.md)。ALZ 工具编译需要 gcc/g++、zlib 和 bzip2 开发头文件。原始数据和真实参考/目标图仅保存本地；GitHub 中的图片来自模型生成。
