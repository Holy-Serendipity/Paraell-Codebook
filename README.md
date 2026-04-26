# RPPG: Parallel Codebook Generative Recommendation

A generative recommendation system that represents items as semantic IDs using parallel codebooks, and predicts the next item in user sequences with a GPT-2 backbone.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (see genrec/datasets/ for format reference)
#    Data files go in /data/cache/<dataset>/raw/

# 3. Train model
python main.py --model RPG --dataset Netease

# 4. Generate recommendations
python main.py --mode generate --model RPG --dataset Netease \
  --checkpoint /output/ckpt/best_model.pth --output recommendations.json
```

See [Configuration](#configuration) for key settings. For common issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Features

- **Semantic ID Generation**: Optimized Product Quantization (OPQ) converts item text embeddings into compact discrete codebook representations
- **GPT-2 Backbone**: Autoregressive transformer predicts next-item semantic IDs from user sequences
- **Graph-Constrained Decoding**: Item-item similarity graph guides beam search for coherent recommendations
- **Swing Algorithm Enhancement**: Collaborative filtering fused with semantic IDs via 3 configurable methods (gated/graph/attention)
- **Flexible Embedding**: Supports BGE, Qwen, OpenAI, and SentenceTransformers models
- **Sparse Matrix Optimization**: Memory-efficient handling of 350k+ item catalogs

## Architecture

### Pipeline
```
Raw Data → Dataset → Tokenizer (OPQ) → Model (GPT-2) → Evaluator
                              ↑
                    Sentence Embedding Model
```

### Semantic ID Generation
1. Item descriptions → dense vectors via sentence embedding model
2. OPQ partitioning into multiple codebooks (e.g., 32 × 256)
3. Each item assigned nearest codeword per codebook → tuple of indices (semantic ID)

### Model Forward Pass
```
item_id2tokens → GPT-2 wte (mean-pool) → [Swing Enhancement] → GPT-2 → Prediction Heads
```
- Semantic IDs mapped to embeddings via GPT-2's word embedding layer
- Optional Swing enhancement blends collaborative filtering similarity into embeddings
- Separate prediction heads for each codebook position

### Swing Enhancement
Computes item-item similarity from user co-occurrence data (Jaccard or exact Swing) and enhances semantic embeddings via:
- **gated**: `σ(W·[emb;neighbor]) ⊙ emb + (1-σ) ⊙ neighbor` — learnable feature-level soft selection
- **graph**: `W_self·emb + W_neighbor·neighbor` — dual-transform analogous to GCN layer
- **attention**: `MultiHeadAttn(Q=emb, K/V=neighbors, bias=Swing_scores)` — retains full neighbor sequence, content-aware multi-head attention with Swing bias (see [Swing Attention](#swing-attention))

All types share the same precomputation pipeline (sparse similarity matrix → top-k neighbors → cached embeddings), differing only in the fusion step.

#### Swing Attention
Unlike the first four methods (which pre-aggregate neighbor embeddings into a single vector), Swing Attention:

1. **Preserves neighbor sequence** — keeps each neighbor embedding separate as `[k, emb_dim]`
2. **Computes content-aware attention** — QKV transformations let the model selectively attend to neighbors based on their content
3. **Adds Swing bias** — Swing similarity scores serve as a fixed bias to the learned attention distribution
4. **Applies residual + LayerNorm** — stable gradient flow and normalized output

```
attention_score = Q_target·K_neighbor^T / √d + swing_scale · swing_sim
enhanced = LayerNorm(emb + W_out(softmax(score)·V))
```

## Project Structure

```
├── main.py                         # Entry point
├── genrec/
│   ├── pipeline.py                 # Training/evaluation pipeline
│   ├── dataset.py                  # Abstract dataset
│   ├── tokenizer.py                # Abstract tokenizer
│   ├── model.py                    # Abstract model
│   ├── trainer.py                  # Training loop
│   ├── evaluator.py                # Metrics (NDCG, Recall)
│   ├── recommender.py              # Batch recommendation generation
│   ├── default.yaml                # Global config
│   ├── datasets/                   # Dataset implementations
│   │   └── {Netease, AmazonReviews2014, Pixel}/
│   └── models/
│       └── RPG/
│           ├── model.py            # RPG model with Swing enhancement
│           ├── tokenizer.py        # OPQ-based tokenizer
│           ├── swing.py            # Swing similarity algorithm
│           └── config.yaml         # Model-specific config
└── requirements.txt
```

## Configuration

### Global (`genrec/default.yaml`)
| Parameter | Description |
|-----------|-------------|
| `train_batch_size`, `eval_batch_size` | Batch sizes |
| `lr`, `weight_decay`, `warmup_steps` | Optimizer settings |
| `epochs`, `patience` | Training schedule |
| `topk`, `metrics` | Evaluation (e.g., [5,10], [ndcg,recall]) |
| `wandb_project`, `wandb_entity` | Weights & Biases |

### Model (`genrec/models/RPG/config.yaml`)
| Parameter | Description |
|-----------|-------------|
| `n_codebook`, `codebook_size` | Codebook structure (e.g., 32 × 256) |
| `sent_emb_model` | Embedding model path/name |
| `n_embd`, `n_layer`, `n_head` | GPT-2 architecture |
| `temperature` | Contrastive loss softmax temperature |
| `num_beams`, `n_edges`, `propagation_steps` | Decoding parameters |

### Swing Enhancement (`config.yaml` - Swing section)
| Parameter | Description |
|-----------|-------------|
| `use_swing_enhancement` | Enable/disable Swing enhancement |
| `swing_enhance_weight` | Enhancement strength (0-1) for gated/graph |
| `swing_neighbors` | Number of neighbor items to aggregate |
| `swing_enhance_type` | Fusion: `gated` / `graph` / `attention` |
| `swing_attention_n_head` | Attention heads (only for `attention` type; must divide `n_embd`) |
| `swing_attention_dropout` | Dropout rate for attention weights |
| `use_item_id_embedding` | Legacy option; disable when using Swing |

## Usage

### Training
```bash
python main.py --model RPG --dataset Netease [--checkpoint PATH]
```
Automatically evaluates on test set after training. Best checkpoint saved to `ckpt_dir`.

### Evaluation Only
```bash
python main.py --mode evaluate --model RPG --dataset Netease --checkpoint /path/to/model.pth
```

### Generate Recommendations
```bash
# Basic generation
python main.py --mode generate --model RPG --dataset Netease \
  --checkpoint /path/to/model.pth --output recommendations.json

# With graph-constrained decoding (first run builds adjacency cache)
python main.py --mode generate --checkpoint /path/to/model.pth \
  --use_graph_decoding --output recommendations.json

# For specific users (one ID per line in file)
python main.py --mode generate --checkpoint /path/to/model.pth \
  --user_list users.txt --output recommendations.json

# Post-training generation
python main.py --mode train --model RPG --dataset Netease \
  --generate_recommendations --recommendations_output recs.json
```

**Output**: JSON with `metadata` (model, dataset, params) and `recommendations` (per-user ranked list with confidence scores). Scores are log probabilities — negative values are normal, higher (closer to 0) is better.

### Swing Enhancement Examples
```bash
# Gated (learnable feature-level selection; default)
python main.py --use_swing_enhancement=true --swing_enhance_type=gated

# Graph (dual-transform, analogous to GCN)
python main.py --use_swing_enhancement=true --swing_enhance_type=graph

# Attention (multi-head content-aware, highest capacity)
python main.py --use_swing_enhancement=true --swing_enhance_type=attention \
  --swing_attention_n_head=4
```

### Embedding Models
Supported: BGE (`BAAI/bge-large-zh-v1.5`), Qwen (`/data/models/Qwen3-Embedding-4B`), OpenAI (`text-embedding-3-large`), SentenceTransformers.

## Performance Tips

- **100k+ items**: Use `swing_sim_type: jaccard` and `use_sparse: true`
- **500k+ items**: Ensure sufficient swap space
- **GPU memory**: Reduce `train_batch_size` or `swing_enhancement` batch size
- **Attention memory**: `swing_enhance_type=attention` caches `[n_items, k, emb_dim]` on CPU (~6 GB for 350k items, k=10). Reduce `swing_neighbors` if RAM constrained
- **Caching**: Adjacency matrices and embeddings cached in `./cache/` for fast reload
- **Overhead**: attention type ~2-3x slower per forward than multiplicative; the other three types have negligible overhead after precomputation

## Citation

```bibtex
@article{rppg2025,
  title={RPPG: Parallel Codebook Generative Recommendation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.
