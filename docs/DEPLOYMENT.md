# RPPG Production Deployment Guide

## Overview

This document covers the production deployment strategy for the RPG model with daily item updates (天更). The current architecture performs all computations globally — there is no incremental update path for new items without code changes.

## Architecture: Incremental Update Feasibility

| Component | Method | Global / Incremental | Why |
|-----------|--------|---------------------|-----|
| Sentence Embedding | `_encode_sent_emb()` (tokenizer.py:120) | **Global** | Encodes ALL items at once; no per-item caching. Cache is a single `.sent_emb` binary file. |
| PCA (optional) | `PCA.fit_transform()` (tokenizer.py:443) | **Global** | PCA is fit on all embeddings; `PCA` object is **never persisted** to disk. |
| OPQ Semantic IDs | `_generate_semantic_id_opq()` (tokenizer.py:252) | **Global** | FAISS `index.train()` on all items. FAISS index is **never saved** standalone — only the `.sem_ids` mapping is cached. |
| Swing/Jaccard Similarity | `SwingSimilarity._compute_jaccard_similarity()` (swing.py:220) | **Global** | Full co-occurrence matrix across all user-item interactions. New interactions change existing item pairs. |
| Decoding Graph (semantic) | `build_ii_topk_adjacency()` (model.py:1341) | **Approximately incremental** | New rows can be computed independently, but existing rows may change. Uses `O(n_items^2)` block iteration. |
| Decoding Graph (fusion) | `_build_fused_adjacency()` (model.py:1573) | **Global** | Depends on Swing, which is global. See above. |

## Option A: Full Daily Rebuild (Recommended, Zero Code Changes)

### Feasibility

At 350k items, a full daily rebuild completes within ~3-5 hours, well within a nightly window:

| Stage | Est. Time | Key Parameters | Cache Path |
|-------|-----------|----------------|------------|
| Sentence Embedding | ~30 min | `sent_emb_batch_size` | `{cache_dir}/processed/{model_name}.sent_emb` |
| PCA + OPQ Training | ~10 min | `n_codebook`, `codebook_size`, `faiss_omp_num_threads` | `{cache_dir}/processed/{n_codebook}-{bits}/{model_name}_{index_factory}.sem_ids` |
| Swing/Jaccard | ~10 min | `swing_alpha`, `swing_min_cooccurrence`, `swing_sim_type` | `{cache_dir}/processed/swing_*.pt` |
| Decoding Graph | ~2 min | `sim_type`, `n_edges` | `./cache/adjacency_{dataset}_items{n_items}_{config}.pt` |
| GPT-2 Training | 2-4 h | `n_layer`, `n_head`, `n_embd`, `epochs` | Checkpoint at `ckpt_dir` |
| Recommendation Gen | ~1 h | `num_beams`, `top_k`, batch_size | — |
| **Total** | **~3-5 h** | | |

### Daily Pipeline Script

```bash
#!/bin/bash
# daily_rebuild.sh — run as nightly cron job

CACHE_DIR=/data/cache12/
DATE=$(date +%Y-%m-%d)
CKPT_DIR=/output/ckpt/daily/$DATE
OUTPUT_DIR=/output/recommendations/$DATE

python main.py \
  --n_codebook=128 \
  --opq_use_gpu=False \
  --cache_dir=$CACHE_DIR \
  --mode=train \
  --dataset=Netease \
  --epochs=200 \
  --patience=40 \
  \
  --sent_emb_batch_size=32 \
  \
  --use_swing=true \
  --swing_sim_type=jaccard \
  --use_swing_enhancement=true \
  --swing_enhance_type=attention \
  --use_swing_contrastive_labels=true \
  --swing_contrastive_weight=0.7 \
  \
  --use_graph_decoding \
  --sim_type=fusion \
  --num_beams=500 \
  --top_k=300 \
  \
  --generate_recommendations \
  --recommendations_output=$OUTPUT_DIR/recommendations.json

echo "Daily rebuild complete: $DATE"
```

### Cache Reuse Strategy

On the **first run**, caches are built from scratch:
- `{cache_dir}/processed/*.sent_emb` — sentence embeddings (rebuilt only if embeddings change)
- `{cache_dir}/processed/*.sem_ids` — OPQ semantic IDs
- `./cache/adjacency_*.pt` — decoding graph
- `{cache_dir}/processed/swing_*.pt` — Swing similarity matrix

On **subsequent runs** (same item set), all caches are reused — only training progress is lost (no checkpoint carry-over).

**Important**: If the item set changes (adds/removes), `.sem_ids` and `.sent_emb` caches must be **manually cleared**. These cache files do NOT encode `n_items` in their filename and may silently return stale data.

---

## Option B: Freeze OPQ Codebook (Code Changes Required)

For environments where semantic ID stability across daily updates is critical (e.g., to reuse previous checkpoints), the OPQ codebook can be frozen and reused.

### Required Code Changes

#### 1. Save FAISS Index After Training

In `tokenizer.py:_generate_semantic_id_opq()` (line 307), after `index.train()` and `index.add()`:

```python
# Save trained FAISS index for incremental use
faiss.write_index(index, os.path.join(
    dataset.cache_dir, 'processed',
    f'opq_{self.index_factory}.faiss'
))
```

#### 2. Save PCA Object

In `tokenizer.py:_init_tokenizer()` (line 443), after `pca.fit_transform()`:

```python
import joblib
joblib.dump(pca, os.path.join(
    dataset.cache_dir, 'processed',
    f'pca_{self.config["sent_emb_pca"]}.pkl'
))
```

#### 3. Incremental Semantic ID Assignment

New method to add a single item:

```python
def _assign_semantic_id_for_new_item(self, item_id, item_meta):
    """Assign semantic ID to a single new item using pre-trained OPQ."""
    # Load saved FAISS index
    index = faiss.read_index(os.path.join(
        self.dataset.cache_dir, 'processed',
        f'opq_{self.index_factory}.faiss'
    ))
    # Load saved PCA
    pca = joblib.load(os.path.join(
        self.dataset.cache_dir, 'processed',
        f'pca_{self.config["sent_emb_pca"]}.pkl'
    ))
    # Encode single item
    emb = self._encode_single_item(item_meta)
    # Apply PCA
    emb = pca.transform(emb.reshape(1, -1))
    # Add to index and decode
    index.add(emb)
    sem_id = decode_pq_codes(index, ...)
    return sem_id
```

#### 4. Modify `_init_tokenizer()` Load Path

Check for existing FAISS index before deciding to train from scratch:

```python
faiss_path = os.path.join(cache_dir, f'opq_{self.index_factory}.faiss')
if os.path.exists(faiss_path) and self._is_incremental_mode():
    # Load FAISS + PCA, incrementally assign IDs to NEW items only
    # Keep existing item2tokens, append new items
else:
    # Full training (original code path)
```

### Benefits
- **Checkpoint reuse**: Semantic IDs don't change, so previous daily checkpoints can be used for warm-start training
- **Faster daily rebuild**: Skip OPQ training (~10 min saved)
- **Compatible with decoding graph**: Semantic similarity still valid because token space is stable

### Trade-offs
- ~200 lines of code changes across `tokenizer.py` and `model.py`
- OPQ codebook quality may degrade if new items shift the embedding distribution significantly
- Periodic full retrain (weekly/monthly) recommended to refresh the codebook

---

## Cold Start: Intra-Day New Items

New items that appear between daily rebuilds cannot be handled by the main model (no semantic ID, no graph edge). A **fallback strategy** is needed:

### Content-Based Vector Retrieval

```
New Item -> Sentence Embedding -> FAISS ANN Search (pre-built index) -> Top-K similar items -> Serve
```

No code changes to the RPG model — this is a separate serving layer.

### Implementation Sketch

```bash
# 1. Build ANN index from existing item embeddings (done once after daily rebuild)
python -c "
import faiss, numpy as np
embs = np.fromfile('{cache_dir}/processed/{model}.sent_emb').reshape(-1, dim)
index = faiss.index_factory(dim, 'IVF100,PQ32', faiss.METRIC_INNER_PRODUCT)
index.train(embs)
index.add(embs)
faiss.write_index(index, './cache/ann_index.faiss')
"

# 2. At serving time, for each cold-start item:
#    a. Encode item text via sentence encoder
#    b. index.search(emb, K=50) -> similar existing items
#    c. Return those items' recommendations as cold-start candidates
```

### Integration Points

| Layer | Where to Add | What It Does |
|-------|-------------|--------------|
| **Serving API** | Before calling RPG model | Detect OOB item IDs, route to ANN fallback |
| **Recommendation output** | Post-processing | Merge or replace cold-start results |
| **Monitoring** | Logging layer | Track cold-start hit rate, latency |

### When to Bypass Cold-Start

After the next daily full rebuild, the cold-start item has a proper semantic ID, Swing scores, and graph edges — it is served by the main model. The ANN fallback is only for **intra-day** gaps.

---

## Production Risk Table

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `.sem_ids` cache does not encode `n_items` | Silent dimension mismatch on `item_id2tokens` init | Medium (item set changes) | Add `n_items` to cache filename; clear cache on data change |
| PCA object not persisted | Cannot transform new items without full refit | High (any incremental use) | Save with `joblib.dump()` alongside FAISS index |
| No OOB handling in `forward()` (model.py:872) | Crash on unknown item ID: `item_id2tokens[batch['input_ids']]` | High (generation on new/test users) | Add `min(item_id, n_items-1)` clamp or filter |
| Sentence embedding is all-or-nothing | Single new item requires re-encoding entire catalog | Low (daily batch) | Acceptable for nightly rebuild; pre-encode for ANN fallback |
| Swing matrix recompute on every rebuild | ~10 min overhead per run | Low (within nightly window) | Cache with file hash or timestamp of interaction data |
| Decoding graph cache stale on config change | Old graph silently used | Low | Cache filename encodes all hyperparameters (already implemented) |
| New user sequences contain OOB items during training | Training crash | Medium | Filter training sequences: only keep items in `item_id2tokens` |

## Daily Operations Checklist

- [ ] Raw data ETL completed and validated
- [ ] `.sem_ids` cache cleared if item set changed
- [ ] `.sent_emb` cache cleared if embedding model changed
- [ ] Sufficient disk space: ~50GB (embeddings + caches + checkpoint)
- [ ] Sufficient RAM: 64GB+ (Swing + graph building)
- [ ] GPU memory: 24GB+ for training (reduce batch size if OOM)
- [ ] Monitor training loss, NDCG@K, Recall@K for regression
- [ ] Validate recommendation output format before serving

## References

- **Tokenizer**: `genrec/models/RPG/tokenizer.py`
- **Swing Similarity**: `genrec/models/RPG/swing.py`
- **Model (forward, graph)**: `genrec/models/RPG/model.py`
- **Pipeline**: `genrec/pipeline.py`
- **Configuration**: `genrec/models/RPG/config.yaml`
