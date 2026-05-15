# FuceCF: Parallel Codebook Generative Recommendation

A generative recommendation system that represents items as semantic IDs using parallel codebooks, and predicts the next item in user sequences with a GPT-2 backbone. Multimodal data (text + image) is unified via image captioning into a shared textual representation space. Collaborative signals are injected through a Swing Attention mechanism.

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

## Architecture

### Pipeline
```
Raw Data → Dataset → Tokenizer (OPQ) → Model (GPT-2) → Evaluator
                              ↑
                    Sentence Embedding Model
```

### Model Forward Pass
```
item_id2tokens → GPT-2 wte (mean-pool) → ⊕ → GPT-2 → Prediction Heads
                                          ↑
[Swing Attention] ────────────────────────┘
```

- **Semantic path**: item content → OPQ tokens → GPT-2 wte mean-pool → Swing Attention enhancement
- **Swing Attention**: multi-head content-aware attention over top-k Swing neighbors, with Swing similarity scores as bias
- **Parallel prediction heads**: one independent head per codebook digit, all predicted simultaneously
- **Auxiliary losses**: cross-codebook group contrastive learning with Swing-aware positive pair construction

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
│           ├── model.py            # RPG model with Swing Attention
│           ├── tokenizer.py        # OPQ-based tokenizer
│           ├── swing.py            # Swing similarity algorithm
│           └── config.yaml         # Model-specific config
└── requirements.txt
```

## Key Configuration

### Global (`genrec/default.yaml`)
| Parameter | Description |
|-----------|-------------|
| `train_batch_size`, `eval_batch_size` | Batch sizes |
| `lr`, `weight_decay`, `warmup_steps` | Optimizer settings |
| `epochs`, `patience` | Training schedule |
| `topk`, `metrics` | Evaluation (e.g., [5,10], [ndcg,recall]) |

### Model (`genrec/models/RPG/config.yaml`)
| Parameter | Description |
|-----------|-------------|
| `n_codebook`, `codebook_size` | Codebook structure (e.g., 32 × 256) |
| `sent_emb_model` | Embedding model path/name |
| `n_embd`, `n_layer`, `n_head` | GPT-2 architecture |
| `temperature` | Contrastive loss softmax temperature |
| `num_beams`, `n_edges`, `propagation_steps` | Decoding parameters |
| `use_swing_enhancement` | Enable/disable Swing Attention |
| `swing_neighbors` | Number of neighbor items to aggregate |
| `swing_attention_n_head` | Attention heads for Swing Attention |
| `swing_enhance_type` | Enhancement type: `gated` / `graph` / `attention` |
| `use_img_embedding` | Enable CLIP cover image embeddings |

## Usage

### Training
```bash
python main.py --model RPG --dataset Netease [--checkpoint PATH]
```

### Generation
```bash
python main.py --mode generate --model RPG --dataset Netease \
  --checkpoint /path/to/model.pth --output recommendations.json
```

### Swing Attention
```bash
python main.py --use_swing_enhancement=true --swing_enhance_type=attention \
  --swing_attention_n_head=4
```

### Multimodal (text + image)
```bash
python main.py --use_img_embedding=true
```

## Citation

```bibtex
@article{fucecf2025,
  title={FuceCF},
  author={Li Jiaming},
  journal={arXiv preprint},
  year={2026}
}
```
