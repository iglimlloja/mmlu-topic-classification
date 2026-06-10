# MMLU Topic Classification

Experiments in classifying academic questions into 57 MMLU subject categories (e.g. `college_physics`, `medical_genetics`, `high_school_mathematics`), with the goal of routing each question to a domain-specialized model or expert.

## Motivation

The MMLU benchmark spans 57 academic subjects. A router that can reliably identify the subject of an incoming question could be used to direct it to a specialized model or fine-tuned adapter, potentially improving accuracy over a single general-purpose LLM. This notebook surveys several classification approaches ranging from zero-shot to full fine-tuning.

## Approaches Explored

| Approach | Model | Notes |
|----------|-------|-------|
| **Zero-shot classification** | `facebook/bart-large-mnli` | NLI-based; scores all 57 topic labels without any task-specific training |
| **Few-shot fine-tuning (SetFit)** | `paraphrase-mpnet-base-v2` | Contrastive sentence-transformer fine-tuning on 40 examples per topic; checkpoint saved at step 9000 |
| **Masked LM probing** | `roberta-base` | Fills a `<mask>` token to probe what domain-specific vocabulary the model associates with a passage |
| **Direct LLM prompting** | Gemma-7B, Mistral-7B, DeepSeek-7B, Qwen-7B | System prompt asks the model to output one of the 57 topic labels; tests zero-shot instruction following |
| **Baseline generation** | Gemma-2B / Gemma-7B | Sanity-check generation tests used as a reference before distillation experiments |

## Files

```
mmlu-topic-classification/
└── topic_classification.ipynb   Full exploration notebook covering all five approaches above;
                                 includes dataset loading from MMLU dev/val/test splits,
                                 SetFit trainer setup, RoBERTa mask-filling, multi-model
                                 prompting, and per-approach accuracy evaluation
```
