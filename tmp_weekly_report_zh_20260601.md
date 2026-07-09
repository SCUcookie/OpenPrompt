# GeoNexus-RSD 周报（2026-06-01）

## 一、本周技术路线

当前论文路线仍按“强检测器基线优先、再逐步引入视觉语言提示模块”的顺序推进：

1. **S0 强检测器基线**：以 DOTA v1.5 reduced tiled split 上的 RoI Transformer 3x 作为主要闭集有监督基线。
2. **S1 RemoteCLIP 平面提示**：在强检测器框架上接入真实 RemoteCLIP 文本提示，验证平面类别提示是否能带来稳定增益。
3. **S2 层级提示正则**：将类别层级关系作为提示正则项，引导细粒度遥感类别的判别边界。
4. **S3 场景/上下文适配**：在 S2 基础上加入场景上下文适配器，评估场景信息对目标检测的贡献。
5. **S4 VLM 辅助伪标签净化**：先做标注保留集上的伪标签质量研究，再考虑半监督重训练收益。

## 二、已完成工作

- **DOTA v1.5 强基线完成**：RoI Transformer 3x 作为当前主基线，最佳 epoch 34 的 `dota/mAP=0.2644`、`dota/AP50=0.2640`。
- **S1/S2 实验归档完成**：S1 frozen-backbone RemoteCLIP 平面提示最佳 `dota/mAP=0.2666`、`dota/AP50=0.2670`；早期 S2 hierarchy-offset 仅达到平齐/轻微提升，不作为主要层级收益证据。
- **S2 hierarchy regularizer 12e/72e 完成并同步记录**：新增 `docs/experiments/20260601_s2_hierarchy_regularizer_12e.md`、`docs/experiments/20260601_s2_hierarchy_regularizer_72e.md` 及对应 metrics JSON。
- **DOTA2/OpenRSD 数据与训练流程推进**：DOTA2 tiled train 标签已生成并用于在线训练流程；当前记录主要作为训练和扩展性证据，尚不能作为 DOTA2 性能结论。
- **服务器队列与实验记录同步**：`New/queues/geonexus_gpu_queue_20260531.json` 已记录 S3 72e、S2 144e、S3 144e 的队列关系和运行状态。

## 三、当前有效论文结果

在同一 DOTA v1.5 reduced tiled split 和 MMRotate DOTAMetric 协议下，当前最强已完成结果来自 **S2 hierarchy regularizer 72e**：

| 方法 | 最佳 epoch | 最佳 mAP / AP50 | 备注 |
| --- | ---: | ---: | --- |
| S0 RoI Transformer 3x | 34 | 0.2644 / 0.2640 | 主检测器基线 |
| S1 RemoteCLIP 平面提示 | 6 | 0.2666 / 0.2670 | frozen-backbone |
| S2 hierarchy regularizer 12e | 11 | 0.3652 / 0.3650 | 首个明确正向层级结果 |
| S2 hierarchy regularizer 72e | 56 | 0.3757 / 0.3760 | 当前最强已完成 S2 证据 |

S2 hierarchy regularizer 72e 的最佳 checkpoint 相比 S0 RoI Transformer 3x 提升约 `+0.1113` mAP，相比 S1 RemoteCLIP 平面提示提升约 `+0.1091` mAP。该结果可以作为当前论文中“层级提示正则有效”的主要实验依据，但应明确限定在 DOTA v1.5 reduced tiled split、MMRotate DOTAMetric、已完成 S2 实验范围内。

## 四、论文阅读与启发

1. **OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images**（ICCV 2025）：该工作面向遥感目标检测提出 open-prompt 框架，支持多模态提示并覆盖水平框/旋转框检测。对 GeoNexus-RSD 的启发是，S2 的层级提示库需要整理成可复用、可对比的 prompt dictionary，并在论文中把 OpenRSD 作为最接近的开放提示/开放词表遥感检测参照，而不是直接声称我们已完成开放词表检测。**启发/下一步**：补一版 DOTA 类别的 flat/hierarchy prompt 字典表，并在 OpenRSD 对比段落中明确我们的当前证据来自闭集 DOTA v1.5 检测协议。
2. **Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community**（AAAI 2025）：该工作将遥感开放词表检测表述为 LAE 任务，并引入动态词表构建和视觉引导文本提示学习。对本项目的价值主要在于帮助 S2/S3 解释“类别语义不是固定字符串，而应随视觉域和训练批次被校准”的动机。**启发/下一步**：在 S2 复现实验之外增加 prompt robustness 分析草案，记录类别名、层级父类、属性短语三类提示对 AP 的影响。
3. **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation**（ICCV 2025）：该工作强调遥感场景中的区域上下文和全局上下文可增强视觉/文本表示，但任务是实例分割，不是本文当前的旋转目标检测指标。它可作为 S3 scene/context adapter 的直接动机来源，说明小目标和易混类别需要上下文消歧。**启发/下一步**：S3 结果归档时除总 mAP 外，优先检查 ship/harbor、small-vehicle/large-vehicle 等易受场景影响的类别对，并补充定性图。
4. **From object to context: Scene knowledge enhanced visual grounding for geospatial understanding**（International Journal of Applied Earth Observation and Geoinformation, 2025）：该工作从遥感视觉定位出发，引入场景知识、关系描述和 query-region alignment，说明地物关系与功能上下文能帮助语言驱动定位。它不直接证明检测 AP 收益，但可指导 S3 adapter 的结构解释和失败案例分析。**启发/下一步**：为 S3 准备“对象-场景关系”分析模板，例如港口-船舶、机场-飞机、道路-车辆，避免只报告整体指标。
5. **LLM-Assisted Semantic Guidance for Sparsely Annotated Remote Sensing Object Detection**（arXiv 2025 预印本）：该工作使用 LLM 语义先验辅助稀疏标注场景下的伪标签分配，并设计类别感知 dense pseudo-label assignment。由于目前是预印本，后续论文中只能作为 S4 设计启发，不能作为强结论依据。**启发/下一步**：S4 先做保留标注集上的伪标签 precision/recall、accepted/rejected 示例和类别混淆评估，再决定是否进入半监督重训练。

## 五、风险与待补齐工作

- **S2 可作为当前最强论文证据，但不是最终结论**：S2 144e 仍在运行，尚需观察更长训练是否继续稳定或回落。
- **S3/S4 证据尚不完整**：S3 scene adapter 72e 在归档时仍处于运行中；S4 伪标签净化还需要先完成伪标签质量实验，不能提前宣称半监督收益。
- **重复实验仍需补齐**：当前 S2 结果很强，但仍建议增加重复运行、次要检测器或关键超参数对照，降低单次运行偶然性的风险。
- **不要夸大 open-vocabulary 结论**：当前证据支持真实 RemoteCLIP 提示和层级正则对该协议下检测结果的提升，不等同于完整开放词表检测能力。
- **DOTA2/OpenRSD 仍是训练流程证据**：已有训练记录证明数据和队列流程可运行，但没有验证结果前不能写入 DOTA2 性能表。

## 六、下周建议

1. 等待并归档 S2 hierarchy regularizer 144e 完整结果，确认是否刷新 S2 最佳值。
2. 继续跟踪 S3 scene adapter 72e，完成后与 S2 72e/144e 对齐比较。
3. 设计 S2 重复实验或轻量对照，优先验证 `+0.11` mAP 级别收益是否稳定。
4. 启动 S4 之前先定义伪标签质量评估表，包括 precision、recall、accepted/rejected 示例和类别混淆分析。

## 七、2026-06-08 补充结论

截至 2026-06-08，论文路线已经明确收敛为 DOTA2-first：DOTA v1.5 结果只保留为 archive/debug 诊断证据，不再作为主表 headline。DOTA2 RoI Transformer S0 为 `0.6088 / 0.6090`，GeoNexus S1 GPU-1 已完成并达到 `0.6177 / 0.6180`，相对 S0 具备正向证据；两个低学习率 S1 复现实验分别为 `0.5997 / 0.6000` 和 `0.6047 / 0.6050`，可作为稳定性对照而不是主结论。

DOTA2 S2 已从 GPU-1 S1 epoch-12 checkpoint 启动，当前主线 epoch-4 为 `0.6038 / 0.6040`，尚未强于 S1，因此 S3/S4、伪标签净化和 routing 仍应暂停，等待 S2 稳定证据。DIOR-R 路线因 ORCNN/RoITrans 的 NaN 和 RetinaNet 的 Inf 现象暂时阻塞，下一步应优先诊断数据记录、旋转框转换、类别映射和 loss target，而不是直接重复发起 detector 训练。

在正式汇报或论文使用前，需要先重新生成并检查 2026-06-08 版本的图表资产，确保图中包含 DOTA2 S1 完成结果、S2 active 状态、DIOR-R blocked 诊断，以及 DOTA v1.5 archive-only 的边界说明。
