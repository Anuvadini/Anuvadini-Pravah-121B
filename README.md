# Anuvadini Pravah-121B

Anuvadini Pravah-121B is a sovereign decoder-only mixture-of-experts large language model, pretrained from random initialisation by Anuvadini Foundation, with strong coverage of English and all 22 scheduled Indian languages.

## At a glance
| Item | Value |
|---|---|
| Total parameters | **121,133,287,296** |
| Active parameters per token | **4,855,752,576** (4.0%) |
| Layers | 36 (alternating 128-token sliding-window and full-context attention, 18 + 18) |
| Model width | 2,880 |
| Attention | 64 query heads × 64 dims, 8 key-value heads (grouped-query attention), learned attention sink per head |
| Experts | 144 per layer, SwiGLU (activation clamp ±7.0), inner width 2,656; top-4 routing |
| Normalisation / positions | RMSNorm (ε = 1e-5), pre-normalisation; rotary position embeddings (base θ = 150,000) |
| Context window | 131,072 tokens (pretrained at 4,096; extended with rotary scaling factor 32) |
| Vocabulary | 201,088 tokens, byte-level BPE — English, 22 scheduled Indian languages, code |
| Embeddings | Untied input and output embeddings |

## Parameter derivation
| Component | Formula | Parameters |
|---|---|---|
| Query projection + bias (per layer) | 2,880 × 4,096 + 4,096 | 11,800,576 |
| Key projection + bias (per layer) | 2,880 × 512 + 512 | 1,475,072 |
| Value projection + bias (per layer) | 2,880 × 512 + 512 | 1,475,072 |
| Output projection + bias (per layer) | 4,096 × 2,880 + 2,880 | 11,799,360 |
| Attention sinks (per layer) | 64 | 64 |
| Router + bias (per layer) | 2,880 × 144 + 144 | 414,864 |
| One expert | 2,880 × 5,312 + 5,312 + 2,656 × 2,880 + 2,880 | 22,956,032 |
| 144 experts (per layer) | 144 × 22,956,032 | 3,305,668,608 |
| RMSNorm weights (per layer) | 2 × 2,880 | 5,760 |
| **One layer** |  | **3,332,639,376** |
| 36 layers | 36 × 3,332,639,376 | 119,975,017,536 |
| Input embedding + output layer | 2 × 201,088 × 2,880 | 1,158,266,880 |
| Final RMSNorm | 2,880 | 2,880 |
| **Total** |  | **121,133,287,296** |

## Pretraining
| Item | Value |
|---|---|
| Initialisation | Random (normal, σ = 0.02) — trained from scratch; no third-party weights |
| Tokens | 30 trillion |
| Compute | ≈ 6 × active parameters × tokens = 8.74e+23 FLOPs |
| Cluster | 2,007 × NVIDIA A100 80 GB, on-premises (not a hyperscale public cloud) |
| Training job | 1,984 GPUs = expert-parallel 8 × pipeline-parallel 4 (9 layers per stage) × data-parallel 62; 23 hot spares |
| Duration | 65.4 days of compute at 25% model-FLOPs utilisation (A100 BF16 peak 312 TFLOPS) ≈ 75 days elapsed |
| GPU-hours | 3.11 million (training job) |
| Throughput | 5.31 million tokens/s (2,677 per GPU) |
| Batch | 4,096 sequences × 4,096 tokens = 16,777,216 tokens per step; ≈ 1,788,139 steps |
| Memory per GPU | ≈ 4.26 B parameters → ≈ 17.9 GB of weights, gradients and sharded optimiser state |
| Optimiser | AdamW (β1 0.9, β2 0.95, ε 1e-8), weight decay 0.1, gradient clip 1.0, peak LR 3e-4, 2,000 warm-up steps, cosine decay to 3e-5, router load-balancing loss |
| Precision | BF16 mixed precision with FP32 master weights |
| Full training state | 1.94 TB (16 bytes per parameter) |

### Corpus mix (30T tokens)
| Source | Share |
|---|---|
| English web (filtered, de-duplicated) | 50% |
| Code | 15% |
| Other languages | 15% |
| Mathematics and science | 8% |
| 22 scheduled Indian languages | 5% |
| Books, academic and reference | 5% |
| Public-domain defence / maritime and synthetic instruction data | 2% |

Processing: exact and near-duplicate removal, language identification, quality classification, PII removal, toxicity filtering and benchmark decontamination. The corpus manifest is available for audit.

## Post-training
Supervised instruction tuning (including Indian languages) → preference optimisation → safety tuning. Domain adaptation is by full-parameter fine-tuning with checkpointing and gated evaluation.

## Deployment
| Item | Value |
|---|---|
| Checkpoint | 67.5 GB with experts in 4-bit microscaling format (attention, router, embeddings in BF16); 242.3 GB in full BF16 |
| Minimum GPU | One 80 GB-class GPU (4-bit experts); 4 × 80 GB for full BF16 |
| KV cache | 36,864 bytes per token (global layers; local layers keep 128 tokens) — 4.84 GB for one 131,072-token sequence |
| API | Standard chat-completions interface; streaming; speculative decoding and continuous batching |
| Operation | Fully on-premise and air-gapped |

## Benchmarks
Reported with exact benchmark variant, split and evaluation date.

---
© Anuvadini Foundation. All rights reserved. Documentation only.
