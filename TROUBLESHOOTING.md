# RPPG Troubleshooting Guide

## Common Issues and Solutions

### 1. Circular Import Error

**Error Message:**
```
Traceback (most recent call last):
  File "/root/PycharmProjects/parade/main.py", line 5, in <module>
    from genrec.pipeline import Pipeline
  File "/root/PycharmProjects/parade/genrec/pipeline.py", line 15, in <module>
    from genrec.recommender import Recommender
  File "/root/PycharmProjects/parade/genrec/recommender.py", line 13, in <module>
    from genrec.pipeline import Pipeline
ImportError: cannot import name 'Pipeline' from partially initialized module 'genrec.pipeline' (most likely due to a circular import)
```

**Cause:**
Circular dependency between `genrec/pipeline.py` and `genrec/recommender.py`.

**Solution:**
1. **Verify the fix is applied:** Ensure `genrec/recommender.py` does NOT contain `from genrec.pipeline import Pipeline` on line 13.
2. **Check for __init__.py:** Ensure `genrec/__init__.py` exists.
3. **Clean Python cache:**
   ```bash
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -type d -exec rm -rf {} +
   ```
4. **Test the fix:**
   ```bash
   python -c "from genrec.pipeline import Pipeline; print('Import successful')"
   ```

**Code Fix Applied:**
- Removed unnecessary import of `Pipeline` from `genrec/recommender.py`
- Added `genrec/__init__.py` to make `genrec` a proper Python package

### 2. Missing Data Files

**Error Symptoms:**
- `FileNotFoundError` when trying to load dataset
- Dataset processing fails

**Required Files for Netease Dataset:**
```
/data/cache/Netease/raw/
├── data_likes.csv    # User-item interactions
└── data_items.csv    # Item metadata
```

**File Formats:**

**data_likes.csv:**
```csv
role_id,work_id,ts
user1,item1,2023-01-01 10:30:00
user1,item2,2023-01-02 14:20:00
user2,item3,2023-01-03 09:15:00
```

**data_items.csv:**
```csv
item_id,metadata
item1,Song Name - Artist Name
item2,Movie Title - Director Name
item3,Product Description
```

**Solutions:**
1. **Place data files:** Copy your data files to the correct directory
2. **Create test data:** For testing, create minimal sample files:
   ```bash
   mkdir -p /data/cache/Netease/raw/
   echo "role_id,work_id,ts" > /data/cache/Netease/raw/data_likes.csv
   echo "user1,item1,2023-01-01 10:30:00" >> /data/cache/Netease/raw/data_likes.csv
   echo "user1,item2,2023-01-02 14:20:00" >> /data/cache/Netease/raw/data_likes.csv
   
   echo "item_id,metadata" > /data/cache/Netease/raw/data_items.csv
   echo "item1,Test Song - Test Artist" >> /data/cache/Netease/raw/data_items.csv
   echo "item2,Test Movie - Test Director" >> /data/cache/Netease/raw/data_items.csv
   ```

### 3. Missing Model Checkpoint

**Error Message:**
```
FileNotFoundError: Checkpoint file not found: /path/to/model.pth
```

**Solutions:**
1. **Use existing checkpoint:** Specify correct path to trained model
   ```bash
   python main.py --mode generate --checkpoint /output/ckpt/best_model.pth
   ```
2. **Train a model first:**
   ```bash
   python main.py --mode train --model RPG --dataset Netease
   ```
3. **Check checkpoint directory:** Models are saved to `ckpt_dir` in config (default: `/output/ckpt/`)

### 4. Missing Embedding Model

**Error Symptoms:**
- `ModuleNotFoundError` for sentence-transformers, FlagEmbedding, etc.
- Error loading embedding model

**Required Models:**
The configuration (`genrec/models/RPG/config.yaml`) specifies:
```yaml
sent_emb_model: /data/models/Qwen3-Embedding-4B
```

**Solutions:**
1. **Download model:** Download the embedding model to specified path
2. **Change model:** Edit config to use a different model:
   ```yaml
   # Use BGE model (automatically downloads)
   sent_emb_model: BAAI/bge-large-zh-v1.5
   
   # Use SentenceTransformer model
   sent_emb_model: all-MiniLM-L6-v2
   ```
3. **Install dependencies:**
   ```bash
   pip install sentence-transformers FlagEmbedding
   ```

### 5. Memory Issues (GPU/CPU)

**Error Symptoms:**
- `CUDA out of memory`
- Process killed due to memory limits

**Solutions:**
1. **Reduce batch size:**
   ```bash
   python main.py --mode generate --batch_size 64
   ```
2. **Use CPU:**
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   python main.py --mode generate
   ```
3. **Limit users:** Generate for subset of users
   ```bash
   echo "user1\nuser2\nuser3" > users.txt
   python main.py --mode generate --user_list users.txt
   ```

### 6. Command Line Arguments Error

**Error Symptoms:**
- `ArgumentError` for invalid arguments
- Missing required arguments

**Correct Usage:**
```bash
# Minimum required for generate mode
python main.py --mode generate \
  --checkpoint /path/to/model.pth \
  --output recommendations.json

# All available options
python main.py --mode generate \
  --model RPG \
  --dataset Netease \
  --checkpoint /path/to/model.pth \
  --top_k 10 \
  --include_scores \
  --use_graph_decoding \
  --batch_size 256 \
  --user_list users.txt \
  --output recommendations.json
```

### 7. Recommendation Generation Issues

#### Issue 7.1: "too many values to unpack (expected 2)"
**Error Message:**
```
ERROR: Failed to generate recommendations: too many values to unpack (expected 2)
```

**Cause:**
Graph-constrained decoding returns 3 values `(preds, scores, visited_counts)` while standard decoding returns 2 values `(preds, scores)`.

**Solution:**
Code already handles this automatically. If you see this error, ensure you have the latest version with the fix in `genrec/recommender.py` lines 439-458.

#### Issue 7.2: Negative Confidence Scores
**Observation:**
Recommendation scores are negative values (e.g., -5.5, -5.6).

**Explanation:**
This is **normal and expected**. Scores are log probabilities from `log_softmax`:
- Negative values: log probabilities are always ≤ 0
- Higher is better: -5.5 is better than -5.6 (closer to 0)
- Probability: `exp(score)` gives actual probability (e.g., -5.5 ≈ 0.004)

**Usage for A/B testing:**
You can directly use these scores for ranking and comparison.

#### Issue 7.3: Slow First Run with Graph Decoding
**Observation:**
First run with `--use_graph_decoding` takes hours.

**Explanation:**
Building the item-item similarity matrix is computationally expensive for large datasets.

**Solution:**
- Matrix is cached to disk after first build
- Subsequent runs load from cache
- Cache location: `cache_dir` in config (default: `./cache/`)
- Cache file: `adjacency_{dataset}_items{n_items}.pt`

#### Issue 7.4: Token ID vs Item ID Confusion
**Observation:**
Output contains very small numbers (1, 2, 3) instead of original item IDs.

**Cause:**
Model outputs token IDs which need conversion to original item IDs.

**Solution:**
Ensure `_token_id_to_item_id()` method in `genrec/recommender.py` is working correctly. Latest code handles this automatically.

#### Issue 7.5: CUDA Memory Errors During Generation
**Error Message:**
```
CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch_size 64`
2. Process fewer users at once: use `--user_list` with subset
3. Use CPU: `export CUDA_VISIBLE_DEVICES=""`
4. Clear GPU cache: `torch.cuda.empty_cache()`

### 8. Import Errors for genrec Module

**Error Message:**
```
ModuleNotFoundError: No module named 'genrec'
```

**Solutions:**
1. **Add project root to PYTHONPATH:**
   ```bash
   export PYTHONPATH=/path/to/Paraell-Codebook:$PYTHONPATH
   cd /path/to/Paraell-Codebook
   python main.py --mode generate ...
   ```
2. **Run from project root:**
   ```bash
   cd /path/to/Paraell-Codebook
   python main.py --mode generate ...
   ```
3. **Install as package:** (Advanced)
   ```bash
   pip install -e .
   ```

### 9. Version Compatibility Issues

**Check Python and Package Versions:**
```bash
python --version
pip list | grep -E "(torch|transformers|sentence-transformers|faiss)"
```

**Required Versions:**
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.36+

### 10. Quick Diagnostics

Run these commands to check your setup:

```bash
# 1. Check Python
python --version

# 2. Test basic import
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 3. Test project import
python -c "from genrec.pipeline import Pipeline; print('Pipeline import OK')"

# 4. Check data files
ls -la /data/cache/Netease/raw/ 2>/dev/null || echo "Data directory not found"

# 5. Check model files
find . -name "*.pth" -o -name "*.pt" 2>/dev/null | head -5

# 6. Check dependencies
pip install -r requirements.txt
```

### 11. Getting Help

If issues persist:

1. **Check logs:** Look for detailed error messages in console output
2. **Verify file paths:** Ensure all paths in config files are correct
3. **Check permissions:** Ensure read/write permissions for data directories
4. **Update code:** Pull latest changes if using git repository

**Common Configuration Files to Check:**
- `genrec/default.yaml` - Global settings
- `genrec/models/RPG/config.yaml` - Model-specific settings
- `genrec/datasets/Netease/config.yaml` - Dataset settings