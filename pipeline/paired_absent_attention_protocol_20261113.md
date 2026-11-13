# Paired Attention Analysis：absent-object 语义关联注意力量化（2026-11-13）

本协议在任何提取运行之前冻结。目标是把论文 Figure 3(a)（Object association
prior）的定性观察升级为定量配对证据：**当目标物体不存在时，模型是否对与该
物体语义相关的视觉区域表现出系统性的 attention preference**。

本实验不训练任何模型、不修改任何已发表结果、不调用任何外部付费接口；
仅使用本地 LLaVA-1.5-7B 权重与 COCO val2017 图像/标注做前向探针。

## 1. 模型与动机

- 定性 Figure 3(a) 与附录可视化均为 drawio 拼装资产，仓库与历史记录中
  没有可追溯的生成脚本或模型记录；按用户既定回退规则使用 **LLaVA-1.5-7B**。
- 该选择与论文机制实验一致：Figure 4 协议明确 `model: LLaVA-1.5-7B`。
- 权重：`llava-hf/llava-1.5-7b-hf`，snapshot `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，
  float16，原生 SDPA。CLIP-ViT-L/14-336 单裁剪，576 = 24×24 视觉 token。

## 2. 数据

- 图像：COCO val2017 全部 5,000 张（来源副本
  `/data/home/chenguanxu/SinkDetectData/paired_absent_attention_20261113/val2017`，
  与 `/data/lff/fast_models/coco_2017_eval2017` 逐字节一致）。
- 标注：官方 `instances_val2017.json` 与 `captions_val2017.json`
  （`annotations_trainval2017.zip`，2017-09-01 原版）。
- 关联关系：论文已冻结的 Figure 4 `related_graph` / `relation_reasons`
  （`LookingIsNotGrounding/figures/assets/figure4/protocol.json`），
  含 train20k 共现统计与"fixed semantic/context family"标记。
  本实验不重新推导共现，不引入新的语义资源。

## 3. 关联 pair 的机械定义（先冻结、后选样）

设 `reasons[T|R]` 为 relation_reasons 中的有序条目 `T|R`（T=target，R=related）。
`cooc(T,R)` 表示其中含 train20k 共现记录的条目（support/conditional/lift）。

**Type 1 — Contextual Co-occurrence（上下文共现）**

入选条件：`cooc(T,R)` 存在且 support≥40、conditional≥0.15、lift≥2.0。
排序键 `support × lift` 降序（平局按类别名字典序），取前 16 个有序 pair；
同一无序 pair 只保留排名最高的一个方向。
另强制纳入论文旗舰 pair `traffic light|car`（满足基础阈值；若已入选则不重复）。

**Type 2 — Category Similarity（类别相似）**

入选条件：`reasons[T|R]` 仅有 "fixed semantic/context family" 标记
（无任何共现记录），且 T、R 均为 COCO 80 类、二者同 COCO supercategory。
按无序 supercategory-pair 分层（同一层内随机取最多 2 个有序 pair），
seed 20261113，直至取满 16 个有序 pair；不足则取全部。
另强制纳入论文旗舰 pair `bicycle|motorcycle`（满足全部条件；若已入选则不重复）。

**Unrelated control U（无关对照查询目标）**

- 默认 `U = teddy bear`（与 Figure 3(a) 示例一致）；
  supercategory-pair 为动物域（animal×animal）的 pair 改用 `U = traffic light`。
- 每个样本还要逐图校验：U 在图中不存在（标准见 §4）、U 不出现在图中任何
  已出现类别（实例或 caption 提及）的 related_graph 邻域内（含反向边，
  即也不允许任何已出现类别出现在 U 的邻域内）、`cooc(U,R)` 不存在
  （双向检查）、U 与 R 不同 COCO supercategory（U 非 COCO 类时跳过该项
  校验）。任一不满足则该图不入选该 pair（计入跳过，不做替换式挑选）。

## 4. 样本入选与冻结

图像 I 入选 pair (T, R, U) 须同时满足：

1. T 在 I 中不存在：无 T 的实例标注，且 I 的全部 COCO 人写 caption 中
   无 T 的词形变体（词边界匹配；变体 = 原名、原名+s，结尾为 s 的名字
   另含去 s 单数形式）。
2. U 按同样标准不存在，且通过 §3 的无关性逐图校验。
3. R 存在：至少一个 `iscrowd=0` 实例，优先使用 segmentation polygon，
   无 polygon 时退化为 bbox。
4. R 的区域映射（§5）覆盖 ≥3 个视觉 token。
5. 配额：每个有序 pair 最多 40 张图（按 image_id 升序扫描、不足补齐）；
   同一图像在全实验中最多贡献 4 个样本。

候选 pair 先 Type 1 后 Type 2，同 Type 内按 (T,R) 字典序；图像按
`image_id` 升序确定性扫描。seed 20261113 仅用于 §3 的随机分层与平局。
选择过程不读取任何 attention、logits 或模型输出（attention-blind）。
冻结产物 `sample_list.json` 记录每样本的 image_id、T/R/U、区域 token
索引、两个 query 的首 subtoken id 与 SHA；提取阶段不得增删改样本。

## 5. 视觉区域 → token 映射

LLaVA-1.5 预处理 = shortest-edge resize 到 336 + center crop 336×336。
token(r,c)（r,c ∈ [0,24)）在裁剪坐标系覆盖
x∈[14c,14c+14)、y∈[14r,14r+14)，其**中心点** (14c+7, 14r+7) 经逆变换

    X = (x + (W_r − 336)/2) × (W / W_r)，  Y 同理

映射回原图坐标（W,H 为原图尺寸，W_r,H_r 为 resize 后尺寸）。
R 的 segmentation polygon 用 pycocotools 栅格化为原图分辨率 mask；
中心点落入任一 R 实例 mask 的 token 全体记为 M_r。
该规则只依赖标注几何与确定性预处理逆变换，不依赖模型输出。

## 6. Attention 提取约定（沿用论文已发表口径）

- 提示模板（LLaVA-1.5 原生）：`USER: <image>\nIs there {article} {name} in the image? ASSISTANT:`，
  冠词为机械规则（元音字母开头用 an，否则 a；影响 apple/orange）。
  提取尚未开始前修正冠词语法，属预注册期修订，两类 query 使用同一模板。
- Query 行：目标名词**首 subtoken** 所在位置（与 Figure 1(a)
  "first object subtoken after consumption" 一致）。related 与 unrelated
  query 共用同一模板，首 subtoken 在模板中的位置天然对齐。
- 探针：原生 SDPA forward 不变；在每个 decoder layer 内用同一 q/k 以
  FP32 重算选中行与全部 key 的因果 softmax（复刻 Figure 1(a) 的
  selected-row FP32 probe）。`A[q,v]` 为该行对 576 个 visual key 的权重。
- 聚合：先按 32 layer × 32 head 对 attention map 取算术平均得到 a_q(v)，
  再按用户公式计算
  `A(q,M_r) = Σ_{v∈M_r} a_q(v) / Σ_{v∈V} a_q(v)`；
  主指标 `ΔA = A(q_rel, M_r) − A(q_unrel, M_r)`（百分点）。
  per-(layer,head) 质量同步存档，供逐层/逐 head 细分与
  fraction-of-means 复核，但不作为主指标。
- 校验门：前 8 个样本另以 FP32 全行 dense 权重切片复算，与选中行结果
  逐位比较（max|Δ| < 1e-4）；不通过则终止，不产出正式结果。

## 7. 统计

- 每样本得 ΔA；按图像聚类 bootstrap（对图像有放回重抽 10,000 次，
  seed 20261113）给出每个 Association Type 的 mean ΔA 与 95% percentile CI。
- 次要证据：逐 pair 的 ΔA 明细表；配 Wilcoxon 符号秩检验
  （scipy 可用时）或符号翻转置换检验（numpy 实现）。
- 同时报告两类 query 的总视觉质量 `Σ_{v∈V} a_q(v)`（描述性），
  用于区分"区域内再分配"与"整体视觉 attention 上升"。

## 8. 判读边界（如实报告）

- ΔA>0 且 CI 不含 0：与 absent 目标语义相关的 query 引起相关区域更强的
  attention preference，支持 Figure 3(a) 的定量版本。
- CI 含 0 或 ΔA<0：如实记录为不支持/方向相反，不得挑选子集重新解读。
- 本实验是 attention 描述性分析：不构成幻觉检测 AUROC、不构成缓解收益、
  不支撑任何干预方法主张；单模型（LLaVA-1.5-7B）结果不外推到其他底座。
- 样本量目标：每 Type ≥300 样本（不足时如实报告实际数与原因）。

## 9. 执行

环境 `/data/miniconda3/envs/llava/bin/python`（torch 2.1.2+cu121 /
transformers 4.37.2 / pycocotools 2.0.6），GPU 0。产物写入
`detection/runs/paired_absent_attention_20261113/`（git-ignored）；
脚本与协议入 `pipeline/`；单测入 `detection/tests/`。
