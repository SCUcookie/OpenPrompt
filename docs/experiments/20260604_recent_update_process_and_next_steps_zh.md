# 近期项目更新过程与下一步选择

日期：2026-06-04

范围：本文整理 2026-05-24 至 2026-06-04 期间 GeoNexus-RSD 的主要实验推进、当前判断和后续路线选择。所有性能结论只对应已记录的数据集、配置和 MMRotate `DOTAMetric` 结果；不要把 DOTA v1.5、DOTA2 或不同评测协议的数字混用。

## 一、这几天的推进过程

### 1. 先修正检测器基础，而不是直接讲提示学习

项目早期本地 scaffold 检测器 mAP 接近 0，诊断显示问题不只是阈值，而是定位路径和预测分布都不可靠。因此近期路线转向“强检测器优先”：先用 MMRotate 的成熟旋转检测器建立可信闭集基线，再逐步接入 RemoteCLIP、层级提示和场景上下文。

这个决策是正确的。当前论文主张依赖“层级与上下文提示能提升旋转遥感目标检测”，如果底层 detector 不可信，后续 S1-S4 的提升都无法解释。

### 2. DOTA v1.5 强基线已经站稳

DOTA v1.5 reduced tiled split 上，S0 强检测器基线已经完成，当前主基线是 RoI Transformer 3x：

| 阶段 | 实验 | 最好结果 | 备注 |
| --- | --- | ---: | --- |
| S0 | RoI Transformer 3x | `dota/mAP=0.2644`, `AP50=0.2640` | 当前主闭集检测器基线 |
| S0 | Oriented R-CNN 3x | `dota/mAP=0.2620`, `AP50=0.2620` | 接近主基线，可作次基线 |
| S0 | ReDet pretrained 12e | `dota/mAP=0.2382`, `AP50=0.2380` | 低于 RoI Transformer |

关键修复是验证/测试 pipeline 顺序：先 resize 图像，再加载标注、转换 qbox/rbox 并打包 meta keys。之前加载标注早于 resize，会造成 GT 尺度错配，导致 AP 异常接近 0。

### 3. S2 层级正则成为当前最强已完成证据

S2 hierarchy regularizer 已经完成 12e、72e 和 144e 三组结果，均在同一 DOTA v1.5 reduced tiled split 和 MMRotate `DOTAMetric` 下评估：

| 阶段 | 训练长度 | 最好 epoch | 最好 mAP / AP50 | final mAP / AP50 |
| --- | ---: | ---: | ---: | ---: |
| S2 | 12e | 11 | `0.3652 / 0.3650` | `0.3644 / 0.3640` |
| S2 | 72e | 56 | `0.3757 / 0.3760` | `0.3738 / 0.3740` |
| S2 | 144e | 30 | `0.3819 / 0.3820` | `0.3723 / 0.3720` |

当前解释应保持谨慎：S2 144e 的 best 是目前最强 S2 验证点，但 final 低于 72e final，说明更长训练不一定带来稳定最终收益。论文中应区分 best checkpoint 和 final checkpoint，不能只挑最好点写成稳定结论。

### 4. S3 场景适配器已经完成，但增益不稳定

S3 scene/context adapter 72e 和 144e 已完成：

| 阶段 | 训练长度 | 最好 epoch | 最好 mAP / AP50 | final mAP / AP50 |
| --- | ---: | ---: | ---: | ---: |
| S3 | 72e | 51 | `0.3800 / 0.3800` | `0.3759 / 0.3760` |
| S3 | 144e | 65/73 | `0.3813 / 0.3810` | `0.3712 / 0.3710` |

S3 72e 相比 S2 72e 略有提升，但 S3 144e 的 best 略低于 S2 144e best，final 也低于 S3 72e final。当前证据支持“场景适配器可能有帮助”，但还不足以强写成稳定显著贡献。后续需要看类别级 AP、易混类别和场景相关类别，而不是只看整体 mAP。

### 5. S1 RemoteCLIP 重跑成为新的关键闸门

2026-06-03，S1 RemoteCLIP prompt-head rerun 已通过真实 RemoteCLIP smoke test：

- 类别数：`16`
- embedding shape：`[16, 512]`
- checkpoint：`RemoteCLIP-ViT-B-32.pt`

首轮 S1 和 retry 1 都在 epoch 1 iter 190 左右因 CUDA OOM 失败，失败点在 RPN/RCNN dense assignment。之后配置补入 `gpu_assign_thr=256`，与 S2/S3 已验证的内存缓解方式对齐。2026-06-04 的 retry 2 已在 GPU 5 上继续运行，并达到当前 best epoch 25：

- `dota/mAP=0.376255`
- `dota/AP50=0.376`
- checkpoint：`epoch_25.pth`

这件事改变了后续优先级：S1 不再只是“平面提示小增益”的旧归档，而可能成为后续 S2 rerun 的更强初始化点。因此新的主线应先完成并归档 S1，再从最佳 S1 checkpoint 启动下一轮 S2。

### 6. DOTA2 侧完成了 valid-PNG 修复和 S0 证据

DOTA2 原始训练集中发现 PNG 解码问题：`170878` 张训练 PNG 中有 `47` 张损坏。近期处理方式是保留原始数据不动，新建 valid-PNG annotation symlink 目录：

- 有效训练标注：`170831`
- 排除损坏图像标注：`47`
- 原始 `train/annfiles` 未修改

修复后，DOTA2 RoI Transformer valid-PNG S0 已完成 12 epoch，在 `DOTA2_1024_500/ss_val` 上得到：

- `dota/mAP=0.6088`
- `dota/AP50=0.6090`

这只能作为 DOTA2 S0 基线证据，不能写成 GeoNexus S1/S2/S3/S4 结果。另一个 DOTA2 Oriented R-CNN valid-PNG run 保留在 GPU 6 上继续跑，当前 best epoch 8 为 `dota/mAP=0.585885`。

## 二、当前项目判断

### 1. 论文主线应继续保持“DOTA v1.5 GeoNexus 主线”

当前最完整、最可解释的链条仍是：

S0 RoI Transformer strong baseline -> S1 RemoteCLIP flat prompt -> S2 hierarchy regularizer -> S3 scene adapter -> S4 pseudo-label purification。

DOTA2 的结果更适合做扩展数据集、泛化验证或后续补充实验，不应现在抢占主线。否则会出现 DOTA v1.5 模块消融和 DOTA2 S0 基线混在一起的问题。

### 2. S2 是当前最强贡献点，S3 需要更细分析

S2 从 S0 的 `0.2644` 提到最高 `0.3819`，这是目前最强的 paper-facing 证据。它支持“层级提示/层级正则对细粒度旋转检测有帮助”。

S3 的整体 mAP 没有稳定超过 S2 144e，因此不宜把 S3 写成第二个确定大幅提升点。更稳妥的写法是：S3 引入场景上下文后，在部分训练长度和部分场景敏感类别上可能有收益，后续需要用 class-wise AP、混淆矩阵和定性图支撑。

### 3. 现在最重要的选择不是 S4，而是是否重建 S1->S2 证据链

旧 S2/S3 结果很强，但当前 S1 retry 2 已经跑到 `0.376255`，高于旧 S0/S1 许多。如果从这个更强 S1 checkpoint 启动 S2，可能出现两种情况：

1. S2 继续提升：论文链条更可信，说明层级正则在强 S1 上仍有增益。
2. S2 不再提升：说明旧 S2 的提升可能部分来自训练/初始化差异，需要重新定义贡献点。

因此下一步应优先完成 S1 并启动 S2 rerun，而不是急着进入 S4。

## 三、下一步推荐路线

### 近期优先级

1. 完成并归档当前 S1 RemoteCLIP rerun。
   需要记录 best epoch、final epoch、checkpoint、config、log、scalars JSON 和是否使用真实 RemoteCLIP embedding。

2. 从最佳 S1 checkpoint 启动 S2 hierarchy regularizer rerun。
   当前候选是 `epoch_25.pth`，除非后续验证超过 `dota/mAP=0.376255`。S2 rerun 应作为 GeoNexus 主线最高优先级。

3. 暂缓新的 DOTA2 次要 baseline。
   DOTA2 RoI Transformer 已有强 S0 证据，Oriented R-CNN 继续保留即可。S2ANet、RTMDet、R3Det 暂不必重启，除非 S1->S2 主线已经安全完成。

4. 对已完成 S2/S3 做类别级分析。
   优先看 `small-vehicle / large-vehicle`、`ship / harbor`、`plane / airport-like context` 等易受层级和场景影响的类别对。若 S3 的整体 mAP 不稳定，类别级收益会决定它是否能作为主模块。

5. S4 先做伪标签质量评估，不直接进入半监督重训。
   先设计 pseudo-label precision、recall、accepted/rejected examples 和类别混淆统计。只有伪标签净化本身有清晰收益，才值得做耗时重训。

### 分叉选择

| 选择 | 触发条件 | 后续动作 |
| --- | --- | --- |
| A. 继续主线 S1->S2->S3 | S1 正常完成，S2 rerun 有稳定收益 | 作为 JSTARS 主论文主线，补齐消融、类别分析和定性图 |
| B. 收缩到 S2 主贡献 | S3 rerun 或类别分析不能支撑上下文收益 | 主打层级提示，S3 写成辅助/探索模块 |
| C. 推迟 S4 | 伪标签质量评估不明显，或 GPU 时间紧张 | S4 只写未来工作或补充实验，不进入主表 |
| D. 扩展 DOTA2 | DOTA v1.5 主线稳定后仍有时间 | 用 DOTA2 valid-PNG 协议做跨数据集验证，严格标注它不是 DOTA v1.5 消融 |

## 四、写论文时应避免的风险

- 不要声称 open-vocabulary detection，除非后续真的做开放词表或词表鲁棒性评估。
- 不要把 DOTA2 `ss_val` 的 S0 结果和 DOTA v1.5 的 GeoNexus 消融放进同一张主比较表。
- 不要只报告 best checkpoint，必须同时保留 final checkpoint 或说明 model selection 规则。
- 不要把 S3 写成稳定提升，除非补齐类别级或重复实验支持。
- 不要让 S4 变成未完成承诺；没有完整结果就不能进入最终主表。

## 五、当前最清晰的下一步决定

短期内最优路线是：继续监控 S1 retry 2，若后续验证没有超过 epoch 25，则使用 `epoch_25.pth` 作为 S2 rerun 初始化点；S2 rerun 完成前，不再投入 GPU 重启 DOTA2 次要基线。这样能最大化 GeoNexus 主论文链条的可信度，也能避免被 DOTA2 工程线分散。

