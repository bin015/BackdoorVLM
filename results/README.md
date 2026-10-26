# Prediction and metric examples

The checked-in prediction samples are copied without changing any fields from
local LLaVA-1.5-7B vLLM inference outputs:

| Example | Records | Required metadata |
| --- | ---: | --- |
| [`coco_caption_val_sample.json`](predict/examples/coco_caption_val_sample.json) | 5 | `image_id` |
| [`vqav2_val_sample.json`](predict/examples/vqav2_val_sample.json) | 5 | `image_id`, `question_id` |

Each record has `prompt`, `predict`, `label`, `image`, and `metadata` fields.
The caption and VQA labels are the task markers `<caption>` and `<answer>`;
the reference answers are read from the evaluation annotations in the
[Hugging Face dataset](https://huggingface.co/datasets/bin015/BackdoorVLM).
The `image` values retain the paths emitted by the original inference run.

The aggregate files below were recomputed from the **full local predictions**,
using the published dataset's reference annotations. They are examples of the
metrics output format, not scores calculated from the five-record samples or
a claim of complete paper reproduction:

| Source predictions | Metric file | Selected result |
| --- | --- | ---: |
| 5,000 COCO Caption records | [`caption metrics`](metrics/llava-1.5-7b-hf-coco_caption_val_vllm_predictions.json) | BLEU-4 0.288; CIDEr 1.062 |
| 200 VQAv2 records | [`VQA metrics`](metrics/llava-1.5-7b-hf-vqav2_val_vllm_predictions.json) | VQA score 81.0 |

To check the metric interface with the small examples, first download the full
dataset into `data/` as described in the root README. The reference annotation
files under `data/eval/utility/reference/` and a Java runtime for the COCO
tokenizer are required. Then run:

```bash
python -m evaluation.metrics results/predict/examples/coco_caption_val_sample.json \
  --task caption --save_dir /tmp/backdoorvlm-metrics
python -m evaluation.metrics results/predict/examples/vqav2_val_sample.json \
  --task vqa --save_dir /tmp/backdoorvlm-metrics
```

The full prediction dumps remain local and are ignored by Git. A separate
legacy `llava_instruct_0.2k` result uses the old `pred` key, so it is not used
as a schema or metrics example.
