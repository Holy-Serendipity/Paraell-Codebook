# RPPG Troubleshooting Guide

## Setup Issues

### Circular Import Error
```
ImportError: cannot import name 'Pipeline' from partially initialized module 'genrec.pipeline'
```
**Cause**: Circular dependency between `pipeline.py` and `recommender.py`.

**Fix**: Ensure `genrec/recommender.py` does not import `Pipeline` from `genrec.pipeline`. Clear cache:
```bash
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +
python -c "from genrec.pipeline import Pipeline; print('OK')"
```

### ModuleNotFoundError: No module named 'genrec'
**Fix**: Run from project root or add to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/Paraell-Codebook:$PYTHONPATH
cd /path/to/Paraell-Codebook && python main.py ...
```

### Missing Embedding Model
```
ModuleNotFoundError: No module named 'FlagEmbedding'
```
Config specifies `sent_emb_model: /data/models/Qwen3-Embedding-4B` by default.

**Fixes**:
- Download the model to the configured path
- Switch to an auto-downloading model: `sent_emb_model: BAAI/bge-large-zh-v1.5`
- Install dependencies: `pip install sentence-transformers FlagEmbedding`

## Data Issues

### FileNotFoundError for Data Files
**Required structure** for Netease:
```
/data/cache/Netease/raw/
├── data_likes.csv    # role_id,work_id,ts
└── data_items.csv    # item_id,metadata
```

**For testing**, create minimal samples:
```bash
mkdir -p /data/cache/Netease/raw/
echo -e "role_id,work_id,ts\nuser1,item1,2023-01-01" > /data/cache/Netease/raw/data_likes.csv
echo -e "item_id,metadata\nitem1,Test Song" > /data/cache/Netease/raw/data_items.csv
```

## Training Issues

### CUDA Unknown Error / DataLoader Freeze
```
RuntimeError: CUDA error: unknown error
— or —
AcceleratorError: CUDA error: unknown error
```
**Cause**: On Linux, PyTorch's DataLoader uses `fork` by default. Forking after CUDA initialization creates child processes with corrupted CUDA context. This occurs when `num_workers > 0` with CUDA.

**Fixes** (pick one):
1. **No multiprocessing** (simple, no data parallelism):
   ```python
   DataLoader(..., num_workers=0, pin_memory=False)
   ```
2. **Spawn mode** (data parallelism works):
   ```python
   import torch.multiprocessing
   DataLoader(..., num_workers=4, multiprocessing_context='spawn', pin_memory=False)
   ```
3. **Set `num_workers=0`** in `genrec/pipeline.py` (already the default in current code).

### CUDA Out of Memory
**Fixes**:
- Reduce `train_batch_size` in `config.yaml`
- Reduce `eval_batch_size`
- Lower `n_embd` or `n_layer` in model config

### Training Stalls at 0% (CPU 98%, GPU 0%)
**Cause**: Swing enhancement precomputation on first forward pass (one-time cost: ~55s for 212k items).

**Fix**: Wait for precompletion, or manually trigger before training:
```python
model.swing_enhancement.precompute_topk_neighbors()
```

### No Improvement with Swing Enhancement
- Start with small weight (`swing_enhance_weight: 0.1`)
- Verify similarity matrix contains meaningful values
- Check logs for `[SwingEnhancement]` messages confirming enhancement is active

## Generation Issues

### Missing Model Checkpoint
```
FileNotFoundError: Checkpoint file not found: /path/to/model.pth
```
**Fixes**: Train first (`python main.py --mode train --model RPG --dataset Netease`), then use the checkpoint saved to `ckpt_dir`.

### Scores Are All Negative
**Expected**: Scores are log probabilities from `log_softmax` — always ≤ 0. Higher (closer to 0) is better. Convert to probability: `exp(score)`.

### Output Shows Small Numbers (1, 2, 3) Instead of Item IDs
**Cause**: Token IDs not converted back to original item IDs.

**Fix**: Check that `_token_id_to_item_id()` in `genrec/recommender.py` maps correctly.

### Slow First Run with Graph Decoding
Building the item-item similarity matrix is expensive for large datasets. The adjacency matrix is cached to `./cache/` after first build — subsequent runs load in seconds.

### too many values to unpack (expected 2)
**Cause**: Graph-constrained decoding returns 3 values, standard returns 2. Latest code handles both — update to the current version.

## Swing Algorithm Issues

### Memory Exhaustion
```
Killed
```
- Use Jaccard approximation: `swing_sim_type: jaccard`
- Ensure `use_sparse: true` in config
- Increase swap space for 500k+ items

### Slow Similarity Computation
Jaccard is 10-100x faster than exact Swing. Similarity matrices are cached to disk automatically after first computation.

### Device Mismatch
```
RuntimeError: Expected all tensors to be on the same device
```
Latest code handles device transfer automatically. If persisting, force CPU for similarity matrices: set `device='cpu'` in `swing.py` initialization.

## GPU Issues

### GPU Not Utilized
- Check `torch.cuda.is_available()` returns True
- Default batch size 8192 for precomputation — may need tuning for your GPU
- Code falls back to CPU automatically if GPU memory insufficient

### CUDA Out of Memory During Precomputation
- Reduce `batch_size` in `precompute_topk_neighbors()` in `genrec/models/RPG/model.py`
- Force CPU: set `use_gpu = False` in the same method
- Clear cache: `torch.cuda.empty_cache()`

## Diagnostics

```bash
# Check environment
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from genrec.pipeline import Pipeline; print('Import OK')"

# Verify data
ls -la /data/cache/Netease/raw/ 2>/dev/null || echo "Data not found"

# Find checkpoints
find . -name "*.pth" -o -name "*.pt" 2>/dev/null | head -5
```

## Need More Help?

1. Check console logs for detailed error messages and stack traces
2. Verify paths in `genrec/default.yaml` and `genrec/models/RPG/config.yaml`
3. Ensure read/write permissions for data and output directories
4. Pull the latest code: `git pull origin main`
