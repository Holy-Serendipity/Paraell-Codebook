# RPPG: Parallel Codebook Generative Recommendation

A generative recommendation system that represents items as semantic IDs using parallel codebooks, and learns to predict the next item in user sequences with a GPT-2 based model.

## Overview

RPPG (Recommendation with Parallel Codebook Generative Modeling) is a novel approach for sequential recommendation that converts item textual descriptions into discrete semantic IDs via product quantization (OPQ), and trains a transformer model to autoregressively predict the next item's semantic IDs given user interaction history. This method combines the benefits of dense semantic representations with efficient discrete codes, enabling high-quality recommendation with reduced computational cost.

Key Features:
- **Semantic ID Generation**: Uses Optimized Product Quantization (OPQ) to quantize item text embeddings into multiple codebooks, creating compact discrete representations.
- **Generative Modeling**: Employs a GPT-2 backbone to model user sequences and predict next-item semantic IDs.
- **Graph-Constrained Decoding**: Incorporates item-item similarity graphs to guide generation and improve recommendation quality.
- **Flexible Embedding Models**: Supports various sentence embedding models (BGE, Qwen, OpenAI, SentenceTransformers) for semantic representation.
- **Modular Design**: Clean separation of dataset processing, tokenization, model architecture, and training pipeline.

## Architecture

The system consists of several key components:

1. **Dataset Processing**: Raw interaction data and item metadata are processed into user-item sequences with textual descriptions.

2. **Semantic ID Tokenizer**:
   - Item descriptions are encoded using a sentence embedding model
   - Embeddings are quantized using FAISS OPQ into multiple codebooks (e.g., 32 codebooks × 256 entries each)
   - Each item is represented as a tuple of codebook indices (semantic IDs)

3. **RPG Model**:
   - **Backbone**: GPT-2 transformer adapted for recommendation tasks
   - **Input**: Mean-pooled embeddings of previous items' semantic IDs
   - **Prediction Heads**: Separate residual blocks for each codebook position
   - **Graph Decoding**: Optional item-item similarity graph for constrained beam search

4. **Training Pipeline**:
   - Leave-one-out sequential split
   - Cross-entropy loss over codebook predictions
   - Evaluation with standard metrics (NDCG@k, Recall@k)

## Project Structure

```
├── main.py                    # Entry point with CLI arguments
├── scripts/                   # Standalone utility scripts
│   └── generate_recommendations.py  # Batch recommendation generation tool
├── genrec/
│   ├── pipeline.py           # Main training/evaluation pipeline
│   ├── dataset.py            # Abstract dataset class
│   ├── tokenizer.py          # Abstract tokenizer class
│   ├── model.py              # Abstract model class
│   ├── trainer.py            # Training loop and evaluation
│   ├── evaluator.py          # Recommendation metrics
│   ├── utils.py              # Utilities (config, logging, etc.)
│   ├── recommender.py        # Batch recommendation generation
│   ├── default.yaml          # Global training configuration
│   ├── datasets/             # Dataset implementations
│   │   ├── Netease/         # Netease Cloud Music dataset
│   │   ├── AmazonReviews2014/ # Amazon review dataset
│   │   ├── Pixel/           # Pixel dataset
│   │   └── __init__.py
│   └── models/               # Model implementations
│       ├── RPG/             # Parallel Codebook model
│       │   ├── model.py     # RPG model implementation
│       │   ├── tokenizer.py # OPQ-based tokenizer
│       │   └── config.yaml  # RPG-specific configuration
│       └── __init__.py
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Paraell-Codebook
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure FAISS is installed with GPU support (if needed):
   ```bash
   pip install faiss-gpu
   ```

4. (Optional) Create cache directory for faster recommendation generation:
   ```bash
   mkdir -p ./cache
   ```
   This directory will store cached adjacency matrices and embeddings for faster subsequent runs, especially important when using graph-constrained decoding.

## Usage

### 1. Prepare Dataset

Place your dataset in the appropriate directory under `genrec/datasets/`. Each dataset should include:
- `config.yaml`: Dataset configuration (paths, metadata)
- `dataset.py`: Dataset loading and processing implementation

Example dataset configurations are provided for Netease, AmazonReviews2014, and Pixel datasets.

### 2. Configure Settings

Edit configuration files as needed:
- `genrec/default.yaml`: Global training parameters (batch size, learning rate, etc.)
- `genrec/models/RPG/config.yaml`: Model-specific settings (codebook size, embedding model, etc.)

### 3. Run Training

```bash
python main.py --model RPG --dataset Netease [--checkpoint PATH]
```

Arguments:
- `--model`: Model name (default: 'RPG')
- `--dataset`: Dataset name (default: 'Netease')
- `--checkpoint`: Optional path to checkpoint for resuming training

### 4. Evaluate Model

The pipeline automatically evaluates on test set after training. Metrics include NDCG@k and Recall@k (k=5,10 by default).

### 5. Generate Recommendations for Online Testing

RPPG provides flexible options for generating recommendations for online A/B testing:

#### Generate Mode (Standalone)
Generate recommendations from a trained checkpoint:

```bash
# Generate recommendations from test set
python main.py --mode generate --model RPG --dataset Netease \
  --checkpoint /path/to/model.pth --output recommendations.json

# Specify number of recommendations per user
python main.py --mode generate --checkpoint /path/to/model.pth \
  --top_k 20 --output recommendations.json

# Include confidence scores
python main.py --mode generate --checkpoint /path/to/model.pth \
  --include_scores --output recommendations.json

# Use graph-constrained decoding (caches adjacency matrix for faster subsequent runs)
python main.py --mode generate --checkpoint /path/to/model.pth \
  --use_graph_decoding --output recommendations.json

# Generate for specific user list (one user ID per line)
python main.py --mode generate --checkpoint /path/to/model.pth \
  --user_list users.txt --output recommendations.json

# Advanced: combine multiple options
python main.py --mode generate --checkpoint /path/to/model.pth \
  --top_k 10 --include_scores --use_graph_decoding \
  --batch_size 256 --output recommendations.json
```

#### Post-Training/Evaluation Generation
Generate recommendations automatically after training or evaluation:

```bash
# Train model and generate recommendations for test users
python main.py --mode train --model RPG --dataset Netease \
  --generate_recommendations --recommendations_output train_recommendations.json

# Evaluate existing model and generate recommendations
python main.py --mode evaluate --model RPG --dataset Netease \
  --checkpoint /path/to/model.pth --generate_recommendations \
  --recommendations_output eval_recommendations.json
```

**Note on Graph-Decoding**: When using `--use_graph_decoding`, the item-item similarity matrix is cached to disk (in the cache directory) for faster subsequent runs. The first run may take several hours for large datasets.

**Output Format:**
The generated JSON file contains:

**Metadata Section:**
- `model`: Model name (e.g., "RPG")
- `dataset`: Dataset name (e.g., "Netease")
- `checkpoint`: Checkpoint path used for generation
- `generation_time`: ISO timestamp of generation
- `top_k`: Number of recommendations per user
- `include_scores`: Whether scores are included
- `use_graph_decoding`: Whether graph-constrained decoding was used
- `total_users`: Number of users in the recommendations

**Recommendations Section:**
List of recommendations per user, each containing:
- `user_id`: User identifier (string)
- `user_history`: User's interaction history (list of original item IDs)
- `recommendations`: List of recommended items, each with:
  - `item_id`: Original item ID (not token ID)
  - `score`: Confidence score (log probability, negative values expected)
  - `rank`: Position in recommendation list (1-based)

**Score Interpretation:**
- **Negative values are normal**: Scores are log probabilities from `log_softmax`
- **Higher values are better**: -5.5 is better than -5.6 (closer to 0)
- **Probability conversion**: `probability = exp(score)` (e.g., -5.5 ≈ 0.004 probability)
- **Range**: Typically between -10 and 0 for well-trained models

**Example Output:**
```json
{
  "metadata": {
    "model": "RPG",
    "dataset": "Netease",
    "checkpoint": "/path/to/model.pth",
    "generation_time": "2026-04-16T18:16:36",
    "top_k": 10,
    "include_scores": true,
    "use_graph_decoding": true,
    "total_users": 1000
  },
  "recommendations": [
    {
      "user_id": "72509001127",
      "user_history": [3063102, 3148501, 3543592, ...],
      "recommendations": [
        {"item_id": 3297408, "score": -5.537, "rank": 1},
        {"item_id": 3499714, "score": -5.541, "rank": 2},
        ...
      ]
    }
  ]
}
```

**Weights & Biases Integration:**
Recommendation generation automatically logs to wandb when enabled, including:
- Generation progress and timing
- GPU memory usage
- Score statistics (mean, std, distribution)
- Recommendation diversity metrics
- Output file as wandb artifact

**Standalone Script:**
For batch recommendation generation, you can also use the standalone script:
```bash
python scripts/generate_recommendations.py \
  --checkpoint /path/to/model.pth \
  --output recommendations.json \
  --top_k 10 \
  --include_scores
```

## Configuration

### Key Parameters

**Global Training (`default.yaml`):**
- `train_batch_size`, `eval_batch_size`: Batch sizes for training and evaluation
- `lr`, `weight_decay`: Optimizer settings
- `epochs`: Number of training epochs
- `topk`, `metrics`: Evaluation metrics
- `wandb_project`, `wandb_entity`: Weights & Biases integration

**RPG Model (`models/RPG/config.yaml`):**
- `n_codebook`, `codebook_size`: Codebook structure (e.g., 32×256)
- `sent_emb_model`: Sentence embedding model path or name
- `max_item_seq_len`: Maximum sequence length
- `n_embd`, `n_layer`, `n_head`: GPT-2 architecture dimensions
- `temperature`: Softmax temperature for contrastive loss
- `num_beams`, `n_edges`, `propagation_steps`: Graph decoding parameters

**Recommendation Generation (`default.yaml` - recommendation section):**
- `top_k`: Number of recommendations per user (default: 10)
- `include_scores`: Whether to include confidence scores in output (default: true)
- `output_format`: Output format for recommendations (default: json)
- `use_graph_decoding`: Use graph-constrained decoding for generation (default: false)
- `batch_size`: Batch size for recommendation generation (default: 256)
- `cache_dir`: Directory for caching adjacency matrices and embeddings (default: `./cache/`)
- `wandb_mode`: Weights & Biases mode for generation tracking (`online`, `offline`, `disabled`)
- `wandb_project`: W&B project name for generation runs
- `wandb_entity`: W&B entity (team/username) for generation runs

### Embedding Models

Supported embedding models:
- **BGE Models**: `BAAI/bge-large-zh-v1.5`, `BAAI/bge-m3`
- **Qwen Embeddings**: `Qwen/Qwen2.5-7B-Instruct`, local paths
- **OpenAI**: `text-embedding-3-large` (requires API key)
- **SentenceTransformers**: `all-MiniLM-L6-v2`, etc.

## How It Works

### Semantic ID Generation

1. **Text Encoding**: Item descriptions are encoded into dense vectors using a sentence embedding model.
2. **Product Quantization**: The embedding space is partitioned into multiple codebooks using OPQ.
3. **Discretization**: Each item is assigned to the nearest codeword in each codebook, creating a tuple of indices (semantic ID).

### Recommendation Generation

#### Online Testing Pipeline
RPPG includes a comprehensive recommendation generation system for online A/B testing:

1. **Batch Processing**: Generates recommendations for test users or specified user lists in batches
2. **Flexible Output**: JSON format with metadata, user history, and scored recommendations
3. **Progress Tracking**: Real-time progress display and wandb integration for monitoring
4. **Caching System**: Adjacency matrices are cached to disk for fast subsequent runs

#### Generation Process
1. **Sequence Encoding**: User interaction history is converted to semantic IDs and mean-pooled.
2. **Transformer Processing**: The GPT-2 model processes the sequence to predict the next item's semantic IDs.
3. **Decoding**: Either:
   - **Direct Decoding**: Select items with highest probability over semantic IDs
   - **Graph-Constrained Decoding**: Use item-item similarity graph to guide beam search
4. **Post-processing**: Convert token IDs back to original item IDs, compute confidence scores
5. **Output Generation**: Format results as JSON with metadata and wandb logging

#### Performance Optimizations
- **Adjacency Matrix Caching**: Graph structures saved to `cache_dir` for reuse
- **Batch Processing**: Configurable batch size for memory-efficient generation
- **GPU Memory Management**: Automatic handling of large recommendation batches
- **Wandb Integration**: Comprehensive metrics tracking for generation runs

### Graph Construction

Item-item similarity is computed based on semantic ID agreement, forming a graph used during decoding to improve recommendation coherence. The adjacency matrix is:
- **Computed once** for each dataset configuration
- **Cached to disk** for fast loading in subsequent runs
- **Memory efficient** using sparse tensor representation
- **Automatically loaded** when `--use_graph_decoding` is enabled

## Results

The model achieves competitive performance on sequential recommendation benchmarks by leveraging semantic information from item descriptions while maintaining efficiency through discrete representations.

## Future Work

- Extend to multi-modal item representations (images, attributes)
- Incorporate user profiles and contextual information
- Explore different quantization techniques (VQ-VAE, residual quantization)
- Deploy to production with optimized inference

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rppg2025,
  title={RPPG: Parallel Codebook Generative Recommendation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Troubleshooting

For common issues and solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## License

[Specify your license here]