# GeoNexus-RSD 项目详尽解析

本文面向本项目当前的研究主线，按“目标是什么、每个模块做什么、公式怎么写、数值怎么代入、哪些内容可直接用、哪些必须训练/实验得到”四个层次展开。

> 研究主线以 `PROJECT_INSTRUCTIONS.md` 为准：
> 1. 强闭集 oriented detector 基线；
> 2. hierarchical prompt bank；
> 3. scene/context prompt adapter；
> 4. VLM-assisted pseudo-label purification。

## 1. 项目在做什么

GeoNexus-RSD 的核心问题不是“单纯做一个检测器”，而是要证明：

1. 遥感图像中的类别层级、场景上下文和视觉语言语义，能够共同提升 oriented object detection；
2. 这种提升不仅体现在最终 mAP，还体现在伪标签质量、混淆类别上的区分能力，以及小目标/密集目标场景下的稳定性；
3. 训练过程需要一个可信的强基线，然后在其上逐步验证 prompt bank、context adapter、VLM purification 的增益。

因此，本项目不是“把 prompt 贴上去就完了”，而是一个有顺序的实验链：

- S0：强闭集 oriented detector baseline
- S1：flat class-name prompt classifier
- S2：hierarchical prompt bank
- S3：hierarchy + scene/context adapter
- S4：hierarchy + context + VLM-assisted pseudo-label purification
- S5：可选 routing ablation

下面把每一层的数学与工程细节拆开。

## 2. 数据集、切片与输入输出

### 2.1 数据集采用什么

项目主语境是 DOTA-style oriented remote sensing detection，优先顺序通常是：

1. DOTA v1.0 / DOTA v1.5 作为强闭集 oriented baseline 的主战场；
2. DOTA v2 在数据处理链路和 tile 管线验证上可作为扩展；
3. 若服务器已有 staged 资产，则先跑已准备好的版本，不要混用不同版本的数值。

### 2.2 Tile 切片的公式

对一张大图，设宽高分别为 $W,H$，tile 尺寸为 $T$，stride 为 $S$。第 $i,j$ 个 tile 的左上角坐标为：

$$
x_{i}=iS,\quad y_{j}=jS.
$$

覆盖的 tile 数量为：

$$
n_x = \left\lceil\frac{W-T}{S}\right\rceil + 1,\quad
n_y = \left\lceil\frac{H-T}{S}\right\rceil + 1.
$$

如果 $T=1024, S=768$，则重叠量为：

$$
\text{overlap}=T-S=256.
$$

如果图像宽高都是 $6000$，则：

$$
n_x=n_y=\left\lceil\frac{6000-1024}{768}\right\rceil+1
=\left\lceil 6.479...\right\rceil+1=8.
$$

所以总 tile 数约为：

$$
N_{tile}=8\times 8=64.
$$

这不是为了“恢复原图”，而是为了保证：

- 显存可承受；
- 小目标在局部窗口中不至于太小；
- 跨边界目标能在 overlap 中至少被一个 tile 覆盖到。

### 2.3 输入输出

输入：RGB 图像 $x\in\mathbb{R}^{H\times W\times 3}$。

输出：每个目标一个 oriented box，常写为：

$$
b=(x_c,y_c,w,h,\theta)
$$

或者在 DOTA 原始标注中写成 4 个点：

$$
(x_1,y_1,x_2,y_2,x_3,y_3,x_4,y_4).
$$

## 3. 什么是 C3、C4、C5，以及 P3、P4、P5

### 3.1 C3 / C4 / C5 的区别

假设 Backbone 是 ResNet 风格，C3/C4/C5 是不同 stage 的输出特征：

- C3：下采样倍数约为 8，分辨率高，细节多，语义较弱；
- C4：下采样倍数约为 16；
- C5：下采样倍数约为 32，分辨率低，但语义最强、感受野最大。

共同点：它们都是卷积特征图，具有通道数 $C$，可用于后续 neck。

差异可以用特征图形状表示为：

$$
C_l \in \mathbb{R}^{\frac{H}{2^l}\times \frac{W}{2^l}\times d_l}
$$

这里 $l$ 只是示意层级，实际 stage 命名以 backbone 实现为准。

### 3.2 P3 / P4 / P5 的来源

P3/P4/P5 是 FPN 生成的金字塔特征，不等于原始 C3/C4/C5，而是“融合版”。典型写法：

$$
P_5 = \mathrm{Conv}_{1\times1}(C_5)
$$

$$
P_4 = \mathrm{Conv}_{1\times1}(C_4) + \mathrm{Up}(P_5)
$$

$$
P_3 = \mathrm{Conv}_{1\times1}(C_3) + \mathrm{Up}(P_4)
$$

随后通常再接一个 $3\times 3$ 卷积平滑：

$$
\tilde P_l = \mathrm{Conv}_{3\times3}(P_l).
$$

### 3.3 为什么上采样能“传递语义”

上采样不是恢复原始像素，而是把高层语义特征扩大到高分辨率网格上。深层特征 $P_5$ 中每个位置已经聚合了更大感受野的语义信息，插值只是把这些语义值分布到更密集的位置上，再与浅层细节特征相加。这样高分辨率层得到的不是“原图”，而是“带语义的高分辨率特征”。

## 4. Anchor、Assignment、IoU、GIoU 的完整逻辑

### 4.1 Anchor 是什么

Anchor 是预先定义的一组参考框。对某个 feature map 位置 $(u,v)$，可能放置多组 anchor：

$$
a=(x_a,y_a,w_a,h_a,\theta_a).
$$

比如尺度取 $\{32,64,128\}$，长宽比取 $\{1:1,1:2,2:1\}$，每个位置就有 9 个 anchor。

### 4.2 Assignment 是什么

Assignment 是把 anchor 分成正样本、负样本和忽略样本。依据通常是 IoU。

IoU 定义为：

$$
\mathrm{IoU}(A,B)=\frac{\mathrm{area}(A\cap B)}{\mathrm{area}(A\cup B)}.
$$

若某 anchor 与任意 GT 的最大 IoU 大于阈值 $T_{pos}$，则是正样本；若小于 $T_{neg}$，则是负样本；中间区域忽略。

例如：

- $T_{pos}=0.5$
- $T_{neg}=0.4$

若某 anchor 与某 GT 的 IoU = 0.62，则为正样本；如果 IoU = 0.18，则为负样本；如果 IoU = 0.46，则可能忽略。

### 4.3 回归目标如何编码

对 anchor $a=(x_a,y_a,w_a,h_a)$ 和 GT $g=(x_g,y_g,w_g,h_g)$，常见编码是：

$$
t_x=\frac{x_g-x_a}{w_a},\quad t_y=\frac{y_g-y_a}{h_a},
$$

$$
t_w=\log\frac{w_g}{w_a},\quad t_h=\log\frac{h_g}{h_a}.
$$

推理时反解：

$$
\hat x=t_x w_a + x_a,\quad \hat y=t_y h_a + y_a,
$$

$$
\hat w=e^{t_w} w_a,\quad \hat h=e^{t_h} h_a.
$$

### 4.4 GIoU 用在哪里

IoU 主要用于 assignment、NMS、评估；GIoU 常用于回归损失。

GIoU 定义：

$$
\mathrm{GIoU}(A,B)=\mathrm{IoU}(A,B)-\frac{|C\setminus(A\cup B)|}{|C|},
$$

其中 $C$ 是包住 $A$ 和 $B$ 的最小闭包框。

回归损失可写为：

$$
L_{box}=1-\mathrm{GIoU}(\hat b,b^*).
$$

### 4.5 数值例子

假设：

- anchor 宽高 $(w_a,h_a)=(64,32)$；
- GT 宽高 $(w_g,h_g)=(80,40)$；
- anchor 中心 $(x_a,y_a)=(200,120)$；
- GT 中心 $(x_g,y_g)=(212,128)$。

则：

$$
t_x=\frac{212-200}{64}=0.1875,
\quad
t_y=\frac{128-120}{32}=0.25.
$$

$$
t_w=\log(80/64)=\log(1.25)\approx 0.2231,
\quad
t_h=\log(40/32)=\log(1.25)\approx 0.2231.
$$

如果网络输出正好学到这些回归量，解码后就能回到 GT 近似位置。

## 5. Head 的详细流程

Head 是整条检测链路里最容易“看起来简单、实际最关键”的部分。

### 5.1 Head 的输入是什么

输入不是原图，而是来自 FPN 的多尺度特征：

$$
\{P_3,P_4,P_5\}
$$

每个 $P_l$ 对应一个尺度层级。

### 5.2 Head 的输出是什么

对于 anchor-based Head，每个位置通常输出：

1. 分类 logits $z \in \mathbb{R}^{C}$；
2. 框回归向量 $t$；
3. 可选的 angle、centerness、quality score。

分类概率通过 softmax：

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}.
$$

若采用 sigmoid 多标签形式，则：

$$
p_k=\sigma(z_k)=\frac{1}{1+e^{-z_k}}.
$$

### 5.3 Head 里每个位置在做什么

以某一层 $P_l$ 为例，空间大小是 $H_l\times W_l$。若每个位置放 9 个 anchors，而类别数是 15，那么每个位置要输出：

$$
9\times 15=135
$$

个分类分数，再加上回归分支的数值。

如果使用四参数回归，那就是每个 anchor 4 个数；若加角度就是 5 个数。

### 5.4 训练阶段的 Head

训练时，Head 接收 assignment 结果，对正样本算分类损失和回归损失，对负样本只算分类损失或背景损失。

分类损失可以是交叉熵：

$$
L_{cls}=-\sum_{k=1}^C y_k\log p_k.
$$

若类别不平衡严重，可以用 Focal Loss：

$$
FL(p_t)=-\alpha(1-p_t)^\gamma \log(p_t).
$$

例如 $\gamma=2,\alpha=0.25$。

### 5.5 推理阶段的 Head

推理时，Head 输出大量候选框。一般流程：

1. 解码回归量；
2. 过滤低分框，例如分数阈值 $s>0.05$；
3. 对所有层级的框做旋转 NMS；
4. 输出最终目标集合。

### 5.6 数值例子

假设某位置输出 15 类 softmax 概率中：

- aircraft: 0.03
- ship: 0.08
- vehicle: 0.72
- others: 0.17

则该 anchor 的预测类别为 vehicle，置信度为 $0.72$。

如果回归分支输出：

$$
t_x=0.1,\; t_y=-0.05,\; t_w=0.2,\; t_h=0.0,
$$

anchor 为 $(x_a,y_a,w_a,h_a)=(300,400,64,32)$，则解码得到：

$$
\hat x=0.1\times64+300=306.4,
$$

$$
\hat y=-0.05\times32+400=398.4,
$$

$$
\hat w=e^{0.2}\times64\approx 78.2,
$$

$$
\hat h=e^0\times32=32.
$$

这就得到一个最终预测框。

## 6. NMS 为什么必须在推理阶段使用

### 6.1 推理阶段是什么

推理阶段就是：模型已经训练完，参数固定，只做前向传播来生成预测，不再反向传播更新权重。

训练阶段：

$$
\theta\leftarrow \theta - \eta\nabla_\theta L
$$

推理阶段：

$$
\hat y=f(x;\theta)
$$

没有参数更新。

### 6.2 NMS 怎么去重

给定若干预测框 $\{(b_i,s_i)\}$，NMS：

1. 按分数排序；
2. 取最高分框；
3. 删除与它 IoU 大于阈值的框；
4. 重复。

如果两个框分别为：

- $b_1$ 分数 0.95
- $b_2$ 分数 0.90
- $\mathrm{IoU}(b_1,b_2)=0.72$

若阈值 $T_{nms}=0.5$，则 $b_2$ 会被删除。

如果是 Soft-NMS，那么 $b_2$ 不一定被删除，而是分数下降，例如：

$$
s_2' = s_2\cdot e^{-\mathrm{IoU}^2/\sigma}
$$

若 $\sigma=0.5$，则 $s_2$ 会显著下降。

## 7. Prompt-bank、开放提示词、VLM 的位置

### 7.1 什么是开放提示词

开放提示词在本项目中不是“随便写一句话”，而是可组织的类语义描述、层级信息、场景先验和负样本约束。

例如一个类 prompt 可以写成：

$$
p_c = [\text{class name}, \text{aliases}, \text{parent class}, \text{scene priors}, \text{geometry priors}, \text{confusing classes}, \text{negative cues}].
$$

### 7.2 Prompt-bank 是什么

Prompt-bank 是 prompt 的集合：

$$
\mathcal P = \{p_1,p_2,\dots,p_N\}
$$

它通常不是一个“单句 prompt”，而是一个带层级结构的语义库。

在项目里，prompt-bank 的作用有三个：

1. 将 taxonomy 变成类嵌入；
2. 为不同场景选择不同 prompt；
3. 给伪标签 purification 提供语义先验。

### 7.3 为什么现在看起来没有 VLM 部分

因为当前仓库还处在 scaffold / research scaffold 阶段，默认 embedder 是 hash fallback，不是真正的 CLIP/SkyCLIP/RemoteCLIP。也就是说：

- 能跑通流程，但语义质量有限；
- 适合 smoke test；
- 不足以直接支撑 paper-level claim。

真正的 VLM 部分需要：

$$
e_{text}=g_{text}(p),\quad e_{img}=g_{img}(crop)
$$

然后用相似度：

$$
s_{vlm}=\cos(e_{text},e_{img})=\frac{e_{text}^\top e_{img}}{\|e_{text}\|\,\|e_{img}\|}.
$$

### 7.4 VLM 在哪里用

VLM 不是替代 detector，而是做两件事：

1. 让 prompt/class embedding 更有语义；
2. 给伪标签 purity 提供 crop-text agreement 分数。

## 8. Scene / Context Adapter 的原理

### 8.1 为什么需要 context

遥感目标高度依赖场景。例如“白色长条”在港口可能是船，在停车场可能是车辆，在工业区可能是储罐。单看局部 patch 不够。

### 8.2 数学表示

设类 prompt 向量为 $e_c$，图像上下文向量为 $z$。Context adapter 可写为：

$$
\tilde e_c = e_c + A(z)
$$

或者门控形式：

$$
\tilde e_c = e_c \odot \sigma(Wz+b)
$$

也可以做拼接：

$$
\tilde e_c = W[e_c;z]+b.
$$

### 8.3 数值例子

如果某个类原始 prompt embedding 是 512 维向量，context 向量也是 512 维，门控输出的某些维度被放大到 1.3 倍，某些维度压到 0.7 倍，那么该类别在“港口场景”下会更偏向船类语义，而在“停车场场景”下更偏向车类语义。

## 9. 伪标签置信度如何计算

### 9.1 最基本的 detector confidence

最简单的伪标签置信度就是分类概率：

$$
s_{det}=\max_k p_k.
$$

如果分类概率分布为：

$$
p=(0.05,0.10,0.78,0.07)
$$

那么置信度就是 $0.78$。

### 9.2 为什么不能只看 detector confidence

因为 detector 有时会高置信地把相似类别混掉。比如：

- ship vs harbor vehicle
- storage tank vs circular roof
- small vehicle vs container

所以项目要求更强的 purification：

$$
s_{purify}=\lambda_1 s_{det}+\lambda_2 s_{hier}+\lambda_3 s_{vlm}+\lambda_4 s_{geo}.
$$

其中：

- $s_{det}$：检测器分数；
- $s_{hier}$：层级一致性；
- $s_{vlm}$：图文对齐分数；
- $s_{geo}$：几何合理性分数。

例如可设：

$$
\lambda_1=0.4,\; \lambda_2=0.2,\; \lambda_3=0.3,\; \lambda_4=0.1.
$$

若某伪标签有：

- $s_{det}=0.92$
- $s_{hier}=0.80$
- $s_{vlm}=0.60$
- $s_{geo}=0.90$

则：

$$
s_{purify}=0.4\times0.92+0.2\times0.80+0.3\times0.60+0.1\times0.90
=0.368+0.16+0.18+0.09=0.798.
$$

如果阈值 $\tau=0.75$，这个伪标签会被接受；如果 $\tau=0.85$，则会被拒绝。

### 9.3 阈值是人为设定的吗

是的，最初通常是人为设定的经验阈值，例如 0.7、0.8、0.9。之后可通过验证集调参：

- 阈值过低：噪声多；
- 阈值过高：伪标签太少；
- 最终目标：找到精度/召回平衡点。

更稳妥的做法是按类别设置不同阈值，因为不同类别的分数分布不同。

## 10. 模型一致性和时序教师

### 10.1 模型一致性

如果同一 crop 上，两个独立模型都预测成 ship，且框位置接近，那么这个伪标签更可信。

可定义一致性分数为：

$$
s_{cons} = \mathbb{1}[\mathrm{class}_1=\mathrm{class}_2]\cdot \mathrm{IoU}(b_1,b_2).
$$

例如两个模型：

- class 都是 ship
- 框 IoU = 0.84

则一致性分数可记为 0.84。

### 10.2 时序教师（Mean Teacher）

teacher 参数通过 EMA 更新：

$$
\theta_t^{(T)} = \alpha\theta_{t-1}^{(T)} + (1-\alpha)\theta_t^{(S)}.
$$

例如 $\alpha=0.99$。若 student 某步参数变化较大，teacher 只缓慢跟随，因此更稳定。

### 10.3 数值例子

假设 teacher 某时刻参数为 10，student 当前参数为 14，$\alpha=0.99$，则新 teacher 为：

$$
\theta^{(T)}_{new}=0.99\times10+0.01\times14=10.04.
$$

这就是“慢更新”。

## 11. batch 的含义，和硬件、效果的关系

### 11.1 batch 是什么

batch 是一次前向/反向传播同时处理的样本数。

如果 batch size = 4，就表示一次拿 4 个 tile 进网络。

### 11.2 对硬件的影响

显存大致随 batch 线性增长。假设单样本占 5 GB 激活 + 1 GB 参数和缓存，则：

- batch=1 约 6 GB；
- batch=2 约 11 GB；
- batch=4 约 21 GB。

实际还要加上中间缓存、FPN、回归分支等开销，所以检测任务里常常 batch 很小。

### 11.3 大 batch 一定更好吗

不一定。

优点：

- 梯度更稳定；
- 吞吐量高；
- 训练曲线更平滑。

缺点：

- 占显存；
- 容易需要调大学习率；
- 过大 batch 可能泛化更差。

线性缩放规则常写作：

$$
\eta' = \eta\cdot \frac{B'}{B}.
$$

例如原来 batch=4，lr=0.01；若改成 batch=16，理论上可试：

$$
\eta'=0.01\times\frac{16}{4}=0.04.
$$

但这只是初始建议，仍需实验。

## 12. 整个流程：从输入到输出

这一节把整个链路写成紧密逻辑结构。

### 12.1 S0 强基线

1. 输入原始大图；
2. 做 tile 切片；
3. 用 oriented detector 训练；
4. 输出 bbox 和类别；
5. 用 rotated NMS 得到最终结果。

S0 的意义：先证明“最基本的闭集检测管线是有效的”。

### 12.2 S1 flat class-name prompt classifier

1. 给每个类别构造 flat prompt；
2. 用文本编码器得到类语义向量；
3. 图像 crop 与类 prompt 做相似度比较；
4. 产生类分数或辅助监督。

这一步主要验证“文本描述是否有帮助”。

### 12.3 S2 hierarchical prompt bank

1. 从 taxonomy 生成层级 prompt；
2. 父类、兄弟类、负类 cues 一起进入 prompt bank；
3. 类嵌入更细粒度。

例如“vehicle”下面可分：car、truck、bus、container truck。层级上先粗后细能减少混淆。

### 12.4 S3 hierarchy + context adapter

1. 输入图像及其场景 context；
2. 通过 adapter 修改类别 embedding；
3. 场景信息影响类判别。

例如港口场景中 ship 类权重上升，停车场场景中 vehicle 类权重上升。

### 12.5 S4 hierarchy + context + VLM-assisted pseudo-label purification

1. teacher detector 对无标签数据产生候选框；
2. prompt/VLM 给候选 crop 计算图文对齐分数；
3. 结合 detector confidence、hierarchy consistency、VLM score、geometry plausibility 得到 purified score；
4. 选出高分伪标签回灌训练。

这是本项目最关键的半监督闭环。

### 12.6 一个完整数值例子

假设：

- 一张原图 6000×6000；
- tile=1024，stride=768；
- 一共有 64 个 tile；
- 每个 tile 上 teacher detector 输出 20 个候选框；
- 一共候选框数约 1280 个。

其中某个候选框：

- detector confidence = 0.93
- hierarchy consistency = 0.85
- VLM agreement = 0.74
- geometry plausibility = 0.90

采用权重：

$$
0.4, 0.2, 0.3, 0.1
$$

则 purified score 为：

$$
0.4\times0.93+0.2\times0.85+0.3\times0.74+0.1\times0.90
=0.372+0.17+0.222+0.09=0.854.
$$

若阈值 $\tau=0.8$，则接受；若 $\tau=0.9$，则拒绝。

这说明阈值不是“绝对真理”，而是实验超参。

## 13. 哪些可以直接用，哪些必须训练/实验得到

### 13.1 可以直接使用的

- tile 切片逻辑；
- anchor 生成模板；
- FPN、NMS、IoU、GIoU 等通用公式；
- 数据增强框架；
- 评估脚本；
- 现成的标注格式转换与可视化逻辑。

### 13.2 必须训练/实验得到的

- detector 权重；
- prompt-bank 的最终取法；
- context adapter 参数；
- VLM 与 detector 的融合权重；
- 伪标签阈值；
- 各类 ablation 的数值结论。

### 13.3 对本项目尤其重要的“不能直接写死”的东西

因为 `PROJECT_INSTRUCTIONS.md` 明确要求 paper-first，所以这些内容必须来自真实实验：

- baseline 的 mAP；
- prompt bank 的增益；
- context adapter 的增益；
- pseudo-label purification 的提升；
- S5 routing 是否值得进入主故事。

## 14. 一个更直观的端到端例子

假设做一次训练：

1. 原始图像 2000 张；
2. 切成 15000 个 tile；
3. 每个 batch 放 2 个 tile/卡；
4. 4 卡训练，全局 batch=8；
5. backbone 输出 C3/C4/C5；
6. FPN 输出 P3/P4/P5；
7. head 在每个位置输出 9 个 anchors，每个 anchor 15 类分数 + 4 个回归数；
8. 训练时 IoU>0.5 的作为正样本；
9. 推理时阈值 0.05 过滤，再做 NMS=0.5；
10. teacher 预测无标签数据，purified score>0.8 的作为伪标签。

最终输出：

- bbox；
- class；
- score；
- mAP；
- 伪标签统计；
- ablation 对比表。

## 15. 总结

如果只用一句话概括本项目：

> 先用一个可信的 oriented detector 把“检测”这件事做稳，再用层级提示、场景上下文和 VLM 来改善类别语义与伪标签质量，最后证明这些结构确实带来可复现实验增益。

如果你后续要继续，我建议的下一步不是再泛泛解释概念，而是直接把这份文档拆成三个可执行清单：

1. baseline 训练与评估清单；
2. prompt bank 构建清单；
3. pseudo-label purification 清单。
