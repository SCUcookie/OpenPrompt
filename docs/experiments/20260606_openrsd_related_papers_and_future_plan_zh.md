# OpenRSD 相关论文与后续方向记录

日期：2026-06-06

用途：为 4 天后的短汇报整理 OpenRSD 邻近文献、创新点、可视化图表设计和 GeoNexus-RSD 下一步实验方向。本文是支持性调研记录，不替代 canonical manuscript `docs/geonexus_short_paper.tex`。

## 1. 背景与结论

OpenRSD 是当前最接近 “open-prompt remote sensing detection” 的公开参考。它把遥感目标检测从固定类别检测推进到开放提示设定，支持文本提示与图像提示，并覆盖 HBB/OBB 与实时推理需求。

截至本记录写入时，还没有确认到公开论文明确声明“以 OpenRSD 为 baseline”或“直接基于 OpenRSD”。因此汇报中不应写成“已有 OpenRSD 后续工作链条”，而应采用更稳妥的表述：以 OpenRSD 为核心邻近工作，同时纳入同方向的开放词汇、提示学习、多模态提示、student-teacher 自学习和伪标签净化论文，构成相关工作集合。

我们的未来方向应收敛为：DOTA2-first 的层级提示或层级正则，DIOR-R 做跨数据集验证，S3 scene adapter 与 S4 pseudo-label purification 暂缓。DOTA v1.5 当前只作为 archive/debug 证据边界，不进入 paper-facing 主结果。

## 2. 相关论文表

| 论文 | 与 OpenRSD 的关系 | 主要创新 | 对我们的启发 |
| --- | --- | --- | --- |
| OpenRSD, ICCV 2025 | 最直接邻近参考；同样面向遥感检测中的开放提示/开放词汇能力。 | 提出 universal open-prompt RS object detector；支持多模态提示；整合多任务检测头，兼顾 HBB/OBB 和实时推理；构建 ORSD+ 作为大规模细化数据。 | 汇报中可把 OpenRSD 作为“问题设定与系统目标”参照；GeoNexus 不应重复泛化口号，而要强调 DOTA2/DIOR-R 上可验证的层级提示和层级正则。 |
| CastDet, ECCV 2024 | OpenRSD 前后的同方向 aerial open-vocabulary detection 工作，不是 OpenRSD baseline。 | 使用 RemoteCLIP 作为额外 teacher；采用 student-teacher self-learning；通过动态标签队列维护 batch 训练中的高质量伪标签。 | S4 后续可借鉴动态伪标签维护，但必须先建立 held-out pseudo-label quality 表，避免只看检测 mAP。 |
| LAE-DINO, AAAI 2025 | 面向遥感开放词汇检测的大规模数据和 foundation detector 路线，与 OpenRSD 的开放类别目标相近。 | 构建 LAE-1M；提出 LAE-DINO；包含 Dynamic Vocabulary Construction 和 Visual-Guided Text Prompt Learning。 | 支持我们把“层级词表/动态词表”作为 DOTA2 S1/S2 的核心，而不是先做复杂 scene adapter。 |
| RT-OVAD, arXiv | 同方向实时开放词汇航拍检测；与 OpenRSD 都关注实时性和图文协同。 | 采用图文协同编码器与文本引导解码器；用 image-to-text alignment loss 替代传统类别回归约束；强调实时开放词汇 aerial detection。 | 可作为图 2 中“文本提示 + 较强开放词汇监督/对齐”的代表；提醒我们效率表和推理速度也应在最终分析中保留。 |
| CoseDet, JSTARS 2025 | 同为遥感开放词汇检测，但更强调上下文语义和 RemoteCLIP 语义注入，不是 OpenRSD 派生。 | 利用上下文语义信息；将 RemoteCLIP embedding 以 pseudo-word 方式注入检测框架；把 region visual/position/shape 信息映射到文本语义空间。 | 支持我们把上下文/层级语义注入写成可控模块；但当前 S3 不稳定，短汇报中只作为未来可吸收思想，不作为主线结果。 |
| RS-MPOD, arXiv 2026 | 与 OpenRSD 共享多模态 prompt 方向，进一步强调 visual prompt 与 text prompt 的互补。 | 使用 visual prompt encoder 从实例样本中提取类别外观线索；支持 text-free category specification；通过多模态融合整合视觉提示和文本提示。 | 强化“视觉提示不是装饰，而是类别指定信号”的论点；可作为未来 DOTA2/DIOR-R S1 的 prompt bank 扩展方向。 |
| VK-Det, AAAI 2026 | 同方向 open-vocabulary aerial detection；更偏视觉知识和原型学习，不依赖 OpenRSD。 | 利用视觉编码器中的 informative region perception；提出 prototype-aware pseudo-labeling；通过类别原型和聚类补偿文本语义偏差。 | 对 S4 的启发是：伪标签净化不能只靠文本相似度，应加入类别原型和类间边界；但这需要先完成 held-out pseudo-label quality 表。 |
| SOAR, AAAI 2026 | 同方向半监督开放词汇 aerial detection；与 OpenRSD 共享开放词汇遥感检测目标。 | 建模 scene background embedding 以间接构建 foreground prior；进行 foreground prior denoising/reconstruction 生成伪标签；结合语言和前景先验增强 query。 | S4 后续可吸收“背景建模 + 前景先验去噪”，尤其适合 DOTA2 密集背景；但在 DOTA2/DIOR-R S1/S2 稳定前不启动。 |

## 3. 建议可视化

### 图 1：相关工作时间线

目标：在一页中说明我们不是孤立提出 prompt detection，而是在 OpenRSD 和 aerial/RS open-vocabulary detection 趋势下收敛到更具体的层级提示路线。

建议顺序：

```text
CastDet (ECCV 2024)
  -> LAE-DINO (AAAI 2025) / RT-OVAD (arXiv)
  -> OpenRSD (ICCV 2025)
  -> RS-MPOD (arXiv 2026) / VK-Det (AAAI 2026) / SOAR (AAAI 2026)
  -> GeoNexus-RSD: DOTA2-first hierarchical prompt / hierarchical regularization
```

图中不要画成“OpenRSD 被这些论文直接继承”，而应标注为“同方向邻近工作”。

### 图 2：二维定位图

目标：把方法族和我们的选择放在同一张坐标图中。

- 横轴：prompt 类型，从 text-only 到 text + visual prompt，再到 hierarchy/context/multimodal prompt。
- 纵轴：监督/伪标签强度，从 closed-set supervised 到 open-vocabulary alignment，再到 self-training / semi-supervised pseudo-label denoising。
- OpenRSD 放在 multimodal open-prompt 区域。
- CastDet/SOAR/VK-Det 放在强伪标签或自训练区域。
- LAE-DINO/RT-OVAD 放在开放词汇对齐与动态词表区域。
- GeoNexus 当前定位在 hierarchy-aware prompt / regularization，伪标签净化暂缓。

### 图 3：当前 DOTA2 S0 baseline 柱状图

目标：短汇报中先建立 DOTA2-first 证据基础，再引出为什么下一步只移植最稳定的 S1/S2。

| Detector | DOTA2 `ss_val` mAP |
| --- | ---: |
| RoI Transformer | 0.6088 |
| Oriented R-CNN | 0.5973 |
| S2ANet | 0.5869 |
| RTMDet-M | 0.3312 |

图中只写 DOTA2 S0 baseline，不混入 DOTA v1.5 GeoNexus S1/S2/S3 结果。

### 表 1：相关论文创新点

可直接复用第 2 节表格，PPT 中压缩为四列：

| 方法 | Prompt/语义来源 | 训练信号 | 可借鉴点 |
| --- | --- | --- | --- |
| OpenRSD | 文本 + 图像提示 | 多阶段 open-prompt 训练 | open-prompt 问题设定 |
| LAE-DINO | 动态词表 + 视觉引导文本提示 | 大规模 LAE-1M | 层级/动态词表 |
| CastDet/SOAR/VK-Det | VLM teacher、背景/前景先验、类别原型 | self-training / semi-supervised | S4 伪标签质量控制 |
| RS-MPOD/CoseDet/RT-OVAD | 视觉提示、上下文语义、图文协同 | prompt/alignment learning | 可作为 S1/S2 扩展设计 |

### 表 2：当前证据边界

| 数据集/证据 | 当前用途 | 可写入汇报的边界 | 禁止表述 |
| --- | --- | --- | --- |
| DOTA2 S0 baselines | paper-path 主基线 | RoI Transformer `0.6088` 是当前最强稳定 S0；ORCNN `0.5973`、S2ANet `0.5869` 为强 secondary baselines；RTMDet-M `0.3312` 为低性能参考。 | 不把尚未完成或被中断的 detector 写成最终完整对比。 |
| DOTA2 GeoNexus S1/S2 | 下一步主线 | 只计划移植最稳定的层级提示或层级正则模块，待完成后再写 paper-facing claim。 | 不提前声称 DOTA2 上已有 GeoNexus gain。 |
| DIOR-R S0/S1/S2 | 必需跨数据集验证 | 先完成 DIOR-R S0 baseline，再复用同一个 S1/S2 模块。 | 不把 DIOR-R 写成已验证结论。 |
| DOTA v1.5 GeoNexus | archive/debug | 可用于解释为什么从 S3/S4 收敛回 S1/S2；只能作为诊断历史。 | 不把 DOTA v1.5 `0.38` 附近结果写成 paper-facing 主结果。 |
| S3/S4 | 暂缓 | S3 scene adapter 和 S4 pseudo-label purification 等 DOTA2/DIOR-R S1/S2 稳定后再做。 | 不把 S4 写成当前已完成贡献。 |

## 4. 未来方向选择

主线：在 DOTA2 上移植稳定的 S1/S2 层级提示或层级正则。优先选择 RoI Transformer 作为强稳定 detector 基座，因为当前 DOTA2 S0 中 RoI Transformer `0.6088` 最强，且与现有 GeoNexus 层级正则经验最接近。

第二数据集：DIOR-R 上先跑 S0 baseline，再复用同一个 S1/S2 模块。DIOR-R 的作用是跨数据集验证，不应在 DOTA2 S1/S2 尚未清楚前临时改故事线。

暂缓：S3 scene adapter 和 S4 pseudo-label purification。S3 在 DOTA v1.5 archive/debug 证据中未稳定超过 S2；S4 需要伪标签质量表支撑，当前不应作为下一步立即实验。

后续 S4 可以吸收 CastDet、SOAR、VK-Det 的伪标签净化思想，包括动态标签队列、背景建模/前景先验去噪、类别原型伪标签。但启动 S4 前必须先做 held-out pseudo-label quality 表，至少记录：

| 检查项 | 目的 |
| --- | --- |
| pseudo-label precision/recall by class | 确认净化不是只提高高频类。 |
| foreground/background error rate | 判断背景误检是否被抑制。 |
| fine-grained class-pair confusion | 检查 large-vehicle/small-vehicle、ship/harbor 等细粒度混淆。 |
| score calibration before/after filtering | 避免过滤阈值只是在重排 confidence。 |
| held-out split metric | 防止在训练集伪标签上自证有效。 |

## 5. 汇报措辞建议

推荐表述：

> OpenRSD 是目前最接近我们问题设定的 open-prompt 遥感检测工作。公开文献中我们暂未确认有论文直接以 OpenRSD 为 baseline，因此本次相关工作采用 OpenRSD 加同方向开放词汇、提示学习和伪标签净化方法作为参照。我们的短期目标不是扩展所有模块，而是在 DOTA2 上先验证层级提示/层级正则，再用 DIOR-R 做跨数据集验证。

避免表述：

- “已有多篇工作基于 OpenRSD 改进。”
- “GeoNexus 已经在 DOTA2 上超过 OpenRSD。”
- “DOTA v1.5 `0.38` 是我们主结果。”
- “S3/S4 是当前稳定贡献。”

## 6. 参考链接

- OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images, ICCV 2025: https://iccv.thecvf.com/virtual/2025/poster/137
- OpenRSD paper PDF, CVF Open Access: https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf
- CastDet: Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning, ECCV 2024: https://eccv.ecva.net/virtual/2024/poster/2279
- LAE-DINO: Locate Anything on Earth, arXiv / AAAI 2025: https://arxiv.org/abs/2408.09110
- LAE-DINO official repository: https://github.com/jaychempan/LAE-DINO
- RT-OVAD: Real-Time Open-Vocabulary Aerial Object Detection via Image-Text Collaboration: https://arxiv.org/abs/2408.12246
- CoseDet: Open-Vocabulary Remote Sensing Object Detection With Contextual Semantic Information, JSTARS 2025: https://doi.org/10.1109/JSTARS.2025.3622239
- RS-MPOD: Beyond Open Vocabulary: Multimodal Prompting for Object Detection in Remote Sensing Images, arXiv 2026: https://arxiv.org/abs/2602.01954
- VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection, arXiv / AAAI 2026: https://arxiv.org/abs/2511.18075
- SOAR: Semi-Supervised Open-Vocabulary Aerial Object Detection via Dual-Aware Enhanced Prior Denoising, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/37671

