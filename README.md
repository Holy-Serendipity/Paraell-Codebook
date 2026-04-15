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

After training, you can generate batch recommendations for online A/B testing:

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

# Use graph-constrained decoding
python main.py --mode generate --checkpoint /path/to/model.pth \
  --use_graph_decoding --output recommendations.json

# Generate for specific user list (one user ID per line)
python main.py --mode generate --checkpoint /path/to/model.pth \
  --user_list users.txt --output recommendations.json
```

**Output Format:**
The generated JSON file contains:
- `metadata`: Model information, generation parameters, and timestamp
- `recommendations`: List of recommendations per user, each with:
  - `user_id`: User identifier
  - `user_history`: User's interaction history
  - `recommendations`: List of recommended items with scores and ranks

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

1. **Sequence Encoding**: User interaction history is converted to semantic IDs and mean-pooled.
2. **Transformer Processing**: The GPT-2 model processes the sequence to predict the next item's semantic IDs.
3. **Decoding**: Either:
   - **Direct Decoding**: Select items with highest probability over semantic IDs
   - **Graph-Constrained Decoding**: Use item-item similarity graph to guide beam search

### Graph Construction

Item-item similarity is computed based on semantic ID agreement, forming a graph used during decoding to improve recommendation coherence.

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

## License

[Specify your license here]