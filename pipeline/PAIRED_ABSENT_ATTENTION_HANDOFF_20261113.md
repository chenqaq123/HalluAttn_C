# Paired Attention Analysis 交接包说明（2026-11-13）

本包是 SinkDetect "Paired Attention Analysis"（absent-object 语义关联注意力量化，
ICLR rebuttal 用）实验的自包含交接包。所有设计与冻结产物已就绪并通过单测；
接手机器只需：装环境 → 准备数据 → 按序执行三个脚本 → 汇报结果。

## 包内容

| 文件 | 作用 |
|---|---|
| `pipeline/paired_absent_attention_protocol_20261113.md` | 预注册协议（已冻结，勿改） |
| `pipeline/build_paired_absent_attention_samples_20261113.py` | 样本冻结脚本（仅供参考/复核，勿重跑） |
| `pipeline/run_paired_absent_attention_extract_20261113.py` | 提取 runner（核心，运行） |
| `pipeline/validate_paired_absent_attention_regions_20261113.py` | 区域映射正向控制（运行） |
| `pipeline/aggregate_paired_absent_attention_20261113.py` | 聚合统计 + bootstrap CI（运行） |
| `detection/tests/test_paired_absent_attention.py` | 单测 18 个（运行，期望全过） |
| `sample_list.json` | **已冻结的 1043 样本清单**（含逐图 SHA256、区域 token 索引；不得重新生成或修改） |

## 实验一句话

同一张图、同一片 present related-object 区域 M_r，对比 related-absent 查询
（如 "Is there a traffic light in the image?"，图中无红绿灯但有 car）
与 unrelated-absent 查询（如 "Is there a teddy bear in the image?"）
下 M_r 的 attention 占比，配对 ΔA>0 即支持 Figure 3(a) 的定量版本。
模型 LLaVA-1.5-7B（fp16，transformers 4.37.x，单卡 ≥16GB 空闲显存）。

## 环境准备

1. Python：需要 **transformers==4.37.x** + torch 2.x+CUDA + pycocotools + numpy
   （runner 复刻了 4.37 的 LlamaAttention/RoPE 内部 API，5.x 不兼容）。
2. protobuf 报错就加环境变量 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`。
3. LLaVA-1.5-7B 权重（HF 格式 llava-hf/llava-1.5-7b-hf，含 safetensors）。
4. fast tokenizer 需在目标环境重建（原快照 tokenizer.json 与新 tokenizers 库不兼容）：
   ```bash
   python -c "from transformers import LlamaTokenizerFast; import os; \
     src='<权重目录>'; dst='<数据目录>/tokenizer_llava15'; \
     os.makedirs(dst,exist_ok=True); \
     LlamaTokenizerFast.from_pretrained(src, local_files_only=True, from_slow=True).save_pretrained(dst)"
   ```
5. 数据：COCO val2017 全部 5000 张 jpg（放任意目录，用 `--image-root` 指定）；
   官方 `instances_val2017.json` + `captions_val2017.json`
   （annotations_trainval2017.zip 内，仅构建/复核脚本用，提取不需要）。

## 执行步骤（按序，均 fail-closed）

```bash
# 0) 单测（期望 18 passed）
python -m pytest detection/tests/test_paired_absent_attention.py -q

# 1) 完整提取：~2200 次 forward，单卡 20-60 分钟，支持断点续跑。
#    前 8 个样本做 selected-vs-dense FP32 审计，失败会立即终止（不得绕过）。
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python \
  pipeline/run_paired_absent_attention_extract_20261113.py \
  --samples sample_list.json \
  --model <权重目录> --tokenizer <数据目录>/tokenizer_llava15 \
  --image-root <val2017图片目录> \
  --output runs/extract --device cuda:0 \
  --audit-count 8 --checkpoint-every 25 \
  --min-free-gb 16 --wait-minutes 0 \
  > runs/extract.log 2>&1 &

# 2) 区域映射正向控制（R-present 查询应显著高于转置/随机区域）
python pipeline/validate_paired_absent_attention_regions_20261113.py \
  --samples sample_list.json --extraction runs/extract \
  --model <权重目录> --tokenizer <数据目录>/tokenizer_llava15 \
  --image-root <val2017图片目录> \
  --output runs/region_validation.json --device cuda:0

# 3) 聚合出表
python pipeline/aggregate_paired_absent_attention_20261113.py \
  --samples sample_list.json --extraction runs/extract \
  --output-json runs/results.json --output-md runs/results.md --output-tex runs/results.tex
```

注意：validate 脚本同样接受 `--image-root`；若你的 validate 副本没有该参数，
给 sample_list.json 里的 `image_file` 字段做同样的文件名映射或把图片放到
原路径结构下也可（逐样本 SHA256 会校验字节一致性，放错数据会直接报错）。

## 需要汇报的结果

1. `runs/results.md` 的两张表：每类（Contextual Co-occurrence / Category
   Similarity）的 #Samples、Related %、Unrelated %、ΔA(pp) [95% CI]、sign-flip p。
2. `runs/results.md` 的 per-pair 与 per-layer 明细。
3. `runs/region_validation.json` 的 summary（映射方向验证）。
4. `runs/extract.log` 中 `audits` 段（8 个样本 dense 审计须全过）。

## 判读与约束

- 如实报告方向：ΔA>0 且 CI 不含 0 才支持预期；不显著或反向也如实记录，
  不得挑选子集重解读。
- 这是单模型（LLaVA-1.5-7B）attention 描述性分析，不构成检测 AUROC、
  不构成缓解收益主张。
- 不要 git commit/push；不要改论文正文与既有官方结果；无任何付费 API。
