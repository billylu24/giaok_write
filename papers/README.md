# 论文资料库

核查日期：2026-10-09。共 10 篇，已从 arXiv、CVF 或 Springer 下载 PDF。每份均通过 PDF 文件头、pdfinfo 和文本提取校验；首页缩略图用于人工核对。

阅读顺序：FontDiffuser → CONSTANT → SLD-Font → One-DM。其余用于部件、整行、轨迹和替代模型路线。

| 论文 | 发表信息 | 本地 PDF | 官方来源 | 校验 |
|---|---|---|---|---|
| FontDiffuser | AAAI 2024 | [FontDiffuser.pdf](FontDiffuser.pdf) | [原始 PDF](https://arxiv.org/pdf/2312.12142) | downloaded |
| One-DM | ECCV 2024 | [One-DM.pdf](One-DM.pdf) | [原始 PDF](https://arxiv.org/pdf/2409.04004) | downloaded |
| HFH-Font | ACM TOG 43(6), 2024 | [HFH-Font.pdf](HFH-Font.pdf) | [原始 PDF](https://arxiv.org/pdf/2410.06488) | downloaded |
| TGC-Diff | ICDAR 2025 | [TGC-Diff.pdf](TGC-Diff.pdf) | [原始 PDF](https://link.springer.com/content/pdf/10.1007/978-3-032-04614-7_9.pdf) | downloaded |
| DiffBrush | ICCV 2025 | [DiffBrush.pdf](DiffBrush.pdf) | [原始 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_Beyond_Isolated_Words_Diffusion_Brush_for_Handwritten_Text-Line_Generation_ICCV_2025_paper.pdf) | downloaded |
| GAR-Font | CVPR 2026 | [GAR-Font.pdf](GAR-Font.pdf) | [原始 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cai_Beyond_Patches_Global-aware_Autoregressive_Model_for_Multimodal_Few-Shot_Font_Generation_CVPR_2026_paper.pdf) | downloaded |
| SLD-Font | 2026 preprint | [SLD-Font.pdf](SLD-Font.pdf) | [原始 PDF](https://arxiv.org/pdf/2602.18874) | downloaded |
| DiffInk | ICLR 2026 | [DiffInk.pdf](DiffInk.pdf) | [原始 PDF](https://arxiv.org/pdf/2509.23624) | downloaded |
| CONSTANT | WACV 2026 | [CONSTANT.pdf](CONSTANT.pdf) | [原始 PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Le_CONSTANT_Towards_High-Quality_One-Shot_Handwriting_Generation_with_Patch_Contrastive_Enhancement_WACV_2026_paper.pdf) | downloaded |
| Latent-Diffusion | CVPR 2022 | [Latent-Diffusion.pdf](Latent-Diffusion.pdf) | [原始 PDF](https://arxiv.org/pdf/2112.10752) | downloaded |

## 精确题目与代码来源

- **FontDiffuser**：FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning。 [官方代码](https://github.com/yeungchenwa/FontDiffuser)。
- **One-DM**：One-DM: One-Shot Diffusion Mimicker for Handwritten Text Generation。 [官方代码](https://github.com/dailenson/One-DM)。
- **HFH-Font**：HFH-Font: Few-shot Chinese Font Synthesis with Higher Quality, Faster Speed, and Higher Resolution。 代码地址未核实，暂不填写。
- **TGC-Diff**：Template-Guided Cascaded Diffusion for Stylized Handwritten Chinese Text-Line Generation。 [官方代码](https://github.com/newbie2niubility/TGC-Diff)。
- **DiffBrush**：Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation。 代码地址未核实，暂不填写。
- **GAR-Font**：Beyond Patches: Global-aware Autoregressive Model for Multimodal Few-Shot Font Generation。 代码地址未核实，暂不填写。
- **SLD-Font**：Disentangle to Create: Structure-Level Disentangled Diffusion Model for Few-Shot Chinese Font Generation。 代码地址未核实，暂不填写。
- **DiffInk**：DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation。 [官方代码](https://github.com/awei669/DiffInk)。
- **CONSTANT**：CONSTANT: Towards High-Quality One-Shot Handwriting Generation with Patch Contrastive Enhancement and Style-Aware Quantization。 [官方代码](https://github.com/duylebkHCM/CONSTANT)。
- **Latent-Diffusion**：High-Resolution Image Synthesis with Latent Diffusion Models。 [官方代码](https://github.com/CompVis/latent-diffusion)。

## 版本与范围

- `manifest.json`：题目、来源和代码链接；题目按下载 PDF 核对。
- `download_report.json`：大小、页数、SHA-256 与状态。
- `*.txt`：从对应 PDF 提取的可搜索文本。
- `qa/first_pages.png`：10 篇论文的首页校验拼图。
- TGC-Diff 官方仓库标注 ICDAR 2025，下载的 Springer PDF 版权年份为 2026。
- SLD-Font 是本文方法简称；实际论文题目为 Disentangle to Create: Structure-Level Disentangled Diffusion Model for Few-Shot Chinese Font Generation。
- 不同论文的字体、手写图像与轨迹任务，以及测试划分并不一致，不能直接比较 SOTA 排名。
- 资料用于当前研究阅读与复现，保留论文版权和原始来源。
