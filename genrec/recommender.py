import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
from logging import getLogger
from datetime import datetime

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.dataset import AbstractDataset
from genrec.utils import get_config, init_seed, init_logger, init_device, log

import wandb
class Recommender:
    """
    Recommender class for generating batch recommendations from trained RPPG models.

    This class handles loading trained models, generating recommendations for users,
    and exporting results in JSON format for online testing.

    Args:
        config (dict): Configuration dictionary
        model (AbstractModel): Trained model instance
        tokenizer (AbstractTokenizer): Tokenizer instance
        dataset (AbstractDataset): Dataset instance

    Attributes:
        config (dict): Configuration parameters
        model (AbstractModel): Loaded model
        tokenizer (AbstractTokenizer): Tokenizer
        dataset (AbstractDataset): Dataset
        logger (Logger): Logger instance
        device (torch.device): Device for computation
    """

    def __init__(
        self,
        config: dict,
        model: AbstractModel,
        tokenizer: AbstractTokenizer,
        dataset: AbstractDataset
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.logger = getLogger()

        # Set device
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()

        # Enable/disable graph decoding based on config
        if self.config.get('use_graph_decoding', False):
            self.model.generate_w_decoding_graph = True
            self.logger.info("Graph-constrained decoding enabled")
        else:
            self.model.generate_w_decoding_graph = False
            self.logger.info("Standard decoding enabled")

    def _log_to_wandb(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics to wandb if wandb is initialized.

        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number for logging
        """
        if self.config.get('wandb_run') is not None:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to wandb: {e}")

    def _record_generation_metrics(self, batch_idx: int, total_batches: int,
                                   batch_size: int, batch_time: float,
                                   scores: Optional[torch.Tensor] = None,
                                   predictions: Optional[torch.Tensor] = None):
        """
        Record batch-level generation metrics to wandb.

        Args:
            batch_idx (int): Current batch index
            total_batches (int): Total number of batches
            batch_size (int): Size of current batch
            batch_time (float): Time taken for current batch in seconds
            scores (torch.Tensor, optional): Confidence scores for predictions
            predictions (torch.Tensor, optional): Predicted item IDs
        """
        metrics = {
            'generation/batch_idx': batch_idx,
            'generation/progress': batch_idx / max(total_batches, 1),
            'generation/batch_size': batch_size,
            'generation/batch_time': batch_time,
            'generation/users_per_second': batch_size / max(batch_time, 1e-6),
        }

        # Add GPU memory usage if available
        if torch.cuda.is_available():
            metrics.update({
                'generation/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 ** 3,  # GB
                'generation/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024 ** 3,  # GB
                'generation/gpu_memory_percent': torch.cuda.memory_allocated() / max(torch.cuda.max_memory_allocated(), 1),
            })

        # Add score statistics if available
        if scores is not None:
            # Squeeze any extra dimensions
            scores_flat = scores.squeeze().cpu()
            if scores_flat.dim() == 0:
                scores_flat = scores_flat.unsqueeze(0)
            scores_np = scores_flat.numpy()
            metrics.update({
                'scores/mean': float(scores_np.mean()),
                'scores/std': float(scores_np.std()),
                'scores/min': float(scores_np.min()),
                'scores/max': float(scores_np.max()),
            })

        # Add prediction statistics if available
        if predictions is not None:
            # Squeeze any extra dimensions
            preds_flat = predictions.squeeze().cpu()
            if preds_flat.dim() == 0:
                preds_flat = preds_flat.unsqueeze(0)
            preds_np = preds_flat.numpy()
            unique_items = len(np.unique(preds_np))
            metrics.update({
                'predictions/unique_items': unique_items,
                'predictions/diversity': unique_items / (preds_np.size if preds_np.size > 0 else 1),
            })

        self._log_to_wandb(metrics)

    def _calculate_recommendation_quality(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate recommendation quality metrics.

        Args:
            recommendations (List[dict]): List of recommendation results

        Returns:
            dict: Dictionary of quality metrics
        """
        if not recommendations:
            return {}

        all_scores = []
        all_item_ids = []

        for rec in recommendations:
            for item in rec['recommendations']:
                all_scores.append(item['score'])
                all_item_ids.append(item['item_id'])

        if not all_scores:
            return {}

        scores_np = np.array(all_scores)
        item_ids_np = np.array(all_item_ids)

        metrics = {
            'quality/score_mean': float(scores_np.mean()),
            'quality/score_std': float(scores_np.std()),
            'quality/score_min': float(scores_np.min()),
            'quality/score_max': float(scores_np.max()),
            'quality/unique_items': len(np.unique(item_ids_np)),
            'quality/diversity': len(np.unique(item_ids_np)) / len(item_ids_np) if len(item_ids_np) > 0 else 0,
            'quality/total_recommendations': len(all_scores),
            'quality/total_users': len(recommendations),
        }

        # Calculate score distribution percentiles
        for p in [10, 25, 50, 75, 90]:
            metrics[f'quality/score_p{p}'] = float(np.percentile(scores_np, p))

        return metrics
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = 'RPG',
        dataset_name: str = 'Netease',
        config_dict: Optional[dict] = None
    ) -> 'Recommender':
        """
        Create a Recommender instance from a checkpoint file.

        Args:
            checkpoint_path (str): Path to model checkpoint
            model_name (str): Name of the model (default: 'RPG')
            dataset_name (str): Name of the dataset (default: 'Netease')
            config_dict (dict, optional): Additional configuration overrides

        Returns:
            Recommender: Initialized recommender instance
        """
        # Get configuration
        config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=None,
            config_dict=config_dict
        )

        # Initialize device and accelerator
        config['device'], config['use_ddp'] = init_device()
        #config['accelerator'] = None  # No accelerator for inference-only
        class SimpleAccelerator:
            def __init__(self):
                self.is_main_process = True
                self.device = config['device']
                self.num_processes = 1

            def prepare(self, *args, **kwargs):
                # Return arguments as-is for inference
                if len(args) == 1:
                    return args[0]
                return args

            def unwrap_model(self, model):
                return model

            def wait_for_everyone(self):
                pass

            def log(self, *args, **kwargs):
                pass

            def gather_for_metrics(self, *args, **kwargs):
                if args and len(args) == 1:
                    return args[0]
                return args

            def end_training(self):
                pass

            def main_process_first(self):
                # Simple context manager for inference mode
                import contextlib
                @contextlib.contextmanager
                def dummy_context():
                    yield

                return dummy_context()

        config['accelerator'] = SimpleAccelerator()

        # Initialize logger
        init_logger(config)
        logger = getLogger()

        # Initialize wandb for recommendation generation tracking
        if config.get('wandb_mode', 'online') != 'disabled':
            # Set up project directory for wandb
            project_dir = os.path.join(
                config.get('tensorboard_log_dir', config.get('log_dir', '/tmp/')),
                config.get('dataset', 'unknown_dataset'),
                config.get('model', 'unknown_model')
            )
            os.makedirs(project_dir, exist_ok=True)

            # Set wandb run name if not provided
            wandb_run_name = config.get('wandb_run_name')
            if wandb_run_name is None:
                from datetime import datetime
                wandb_run_name = f"generate-{config.get('model', 'unknown')}-{config.get('dataset', 'unknown')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                config['wandb_run_name'] = wandb_run_name

            # Initialize wandb run
            wandb.init(
                project=config.get('wandb_project', 'RPG-Netease'),
                name=wandb_run_name,
                config=config,  # Record all configuration parameters
                dir=project_dir,  # Set wandb directory
                resume="allow",  # Allow resume
                mode=config.get('wandb_mode', 'online'),  # Online or offline mode
                entity=config.get('wandb_entity'),  # Team or user name
                tags=config.get('wandb_tags', []) + ['generation']  # Add generation tag
            )

            # Save wandb run info to config
            config['wandb_run'] = wandb.run
            config['wandb_run_id'] = wandb.run.id

            # Record configuration to wandb
            wandb.config.update(config)

            # Create wandb config file
            config_file_path = os.path.join(project_dir, 'config_generate.yaml')
            with open(config_file_path, 'w') as f:
                import yaml
                yaml.dump(config, f)

            # Upload config file to wandb
            wandb.save(config_file_path, base_path=os.path.dirname(config_file_path))

            logger.info(f"Initialized wandb run: {wandb_run_name} (ID: {wandb.run.id})")
        else:
            config['wandb_run'] = None
            config['wandb_run_id'] = None

        # Load dataset
        from genrec.utils import get_dataset
        dataset_class = get_dataset(dataset_name)
        dataset = dataset_class(config)
        logger.info(f"Loaded dataset: {dataset}")

        # Load tokenizer
        from genrec.utils import get_tokenizer
        tokenizer_class = get_tokenizer(model_name)
        tokenizer = tokenizer_class(config, dataset)

        # Load model
        from genrec.utils import get_model
        model_class = get_model(model_name)
        model = model_class(config, dataset, tokenizer)

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=config['device']), strict=False)
            logger.info(f"Loaded model checkpoint from {checkpoint_path} (strict=False due to architecture changes)")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        return cls(config, model, tokenizer, dataset)

    def generate_for_user(
        self,
        user_history: List[int],
        top_k: int = 10,
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Generate recommendations for a single user.

        Args:
            user_history (List[int]): List of item IDs in user's interaction history
            top_k (int): Number of recommendations to generate
            include_scores (bool): Whether to include confidence scores

        Returns:
            dict: Recommendation result for the user
        """
        # Convert user history to model input format
        batch = self._prepare_batch([user_history])

        with torch.no_grad():
            # Generate recommendations
            if include_scores:
                try:
                    # Use return_scores if available, otherwise generate and compute scores separately
                    if hasattr(self.model, 'generate_with_scores'):
                        preds, scores = self.model.generate_with_scores(batch, n_return_sequences=top_k)
                    elif hasattr(self.model.generate,
                                 '__code__') and 'return_scores' in self.model.generate.__code__.co_varnames:
                        # Try standard unpacking first
                        try:
                            preds, scores = self.model.generate(batch, n_return_sequences=top_k, return_scores=True)
                        except ValueError as e:
                            if "too many values to unpack" in str(e):
                                # Graph decoding returns 3 values: preds, scores, visited_counts
                                preds, scores, visited_counts = self.model.generate(batch, n_return_sequences=top_k,
                                                                                    return_scores=True)
                                self.logger.debug(
                                    f"Graph decoding: received visited_counts shape {visited_counts.shape}")
                            else:
                                raise e
                    else:
                        # Fallback: generate without scores
                        preds = self.model.generate(batch, n_return_sequences=top_k)
                        scores = torch.ones_like(preds, dtype=torch.float32)  # Placeholder scores
                except Exception as e:
                    self.logger.error(f"Error generating recommendations with scores: {e}")
                    # Fallback without scores
                    preds = self.model.generate(batch, n_return_sequences=top_k)
                    scores = torch.ones_like(preds, dtype=torch.float32)
            else:
                preds = self.model.generate(batch, n_return_sequences=top_k)
                scores = None

        # Convert tensors to Python lists
        recommendations = preds.squeeze(-1).cpu().tolist()[0]  # Shape: (1, top_k) -> list
        if scores is not None:
            score_list = scores.squeeze(-1).cpu().tolist()[0]
        else:
            score_list = [1.0] * len(recommendations)  # Default scores

        # Format recommendations - convert token IDs back to item IDs
        rec_items = []
        for i, (token_id, score) in enumerate(zip(recommendations, score_list)):
            # Convert token ID back to item ID
            item_id = self._token_id_to_item_id(int(token_id))
            rec_items.append({
                "item_id": item_id,
                "score": float(score),
                "rank": i + 1
            })

        return {
            "user_history": user_history,
            "recommendations": rec_items
        }

    def generate_for_users(
        self,
        user_histories: List[List[int]],
        top_k: int = 10,
        include_scores: bool = True,
        batch_size: int = 256,
        user_ids: Optional[List[Union[str, int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for multiple users in batch.

        Args:
            user_histories (List[List[int]]): List of user histories
            top_k (int): Number of recommendations per user
            include_scores (bool): Whether to include confidence scores
            batch_size (int): Batch size for inference
            user_ids (List[Union[str, int]], optional): User IDs corresponding to histories

        Returns:
            List[dict]: Recommendation results for all users
        """
        if user_ids is None:
            user_ids = list(range(len(user_histories)))

        if len(user_histories) != len(user_ids):
            raise ValueError(f"Number of user histories ({len(user_histories)}) "
                           f"must match number of user IDs ({len(user_ids)})")

        all_results = []

        # Calculate total batches
        total_batches = (len(user_histories) + batch_size - 1) // batch_size

        # Process in batches
        # for i in tqdm(range(0, len(user_histories), batch_size), desc="Generating recommendations"):
        for batch_idx, i in enumerate(tqdm(range(0, len(user_histories), batch_size), desc="Generating recommendations")):
            batch_histories = user_histories[i:i+batch_size]
            batch_user_ids = user_ids[i:i+batch_size]

            # Record batch start time
            import time
            batch_start_time = time.time()

            # Prepare batch
            batch = self._prepare_batch(batch_histories)

            with torch.no_grad():
                # Generate recommendations for batch
                if include_scores:
                    try:
                        # Try to get scores - handle both standard and graph decoding
                        if hasattr(self.model, 'generate_with_scores'):
                            preds, scores = self.model.generate_with_scores(batch, n_return_sequences=top_k)
                        elif hasattr(self.model.generate,
                                     '__code__') and 'return_scores' in self.model.generate.__code__.co_varnames:
                            # Try standard unpacking first
                            try:
                                preds, scores = self.model.generate(batch, n_return_sequences=top_k, return_scores=True)
                            except ValueError as e:
                                if "too many values to unpack" in str(e):
                                    # Graph decoding returns 3 values: preds, scores, visited_counts
                                    preds, scores, visited_counts = self.model.generate(batch, n_return_sequences=top_k,
                                                                                        return_scores=True)
                                    self.logger.debug(
                                        f"Graph decoding: received visited_counts shape {visited_counts.shape}")
                                else:
                                    raise e
                        else:
                            preds = self.model.generate(batch, n_return_sequences=top_k)
                            scores = torch.ones_like(preds, dtype=torch.float32)
                    except Exception as e:
                        self.logger.error(f"Error generating recommendations with scores: {e}")
                        # Fallback without scores
                        preds = self.model.generate(batch, n_return_sequences=top_k)
                        scores = torch.ones_like(preds, dtype=torch.float32)
                else:
                    preds = self.model.generate(batch, n_return_sequences=top_k)
                    scores = None
            # Calculate batch time and record metrics
            batch_time = time.time() - batch_start_time
            current_batch_size = len(batch_histories)
            # Record generation metrics to wandb
            self._record_generation_metrics(
                batch_idx=batch_idx,
                total_batches=total_batches,
                batch_size=current_batch_size,
                batch_time=batch_time,
                scores=scores,
                predictions=preds
            )
            # Process batch results
            batch_preds = preds.squeeze(-1).cpu().tolist()  # Shape: (batch_size, top_k)
            if scores is not None:
                batch_scores = scores.squeeze(-1).cpu().tolist()
            else:
                batch_scores = [[1.0] * top_k for _ in range(len(batch_preds))]

            for j, (user_id, history, pred_list, score_list) in enumerate(
                zip(batch_user_ids, batch_histories, batch_preds, batch_scores)
            ):
                rec_items = []
                for rank, (item_id, score) in enumerate(zip(pred_list, score_list), 1):
                    # Convert token ID to original item ID
                    original_item_id = self._token_id_to_item_id(int(item_id))
                    rec_items.append({
                        "item_id": original_item_id if isinstance(original_item_id, int) else int(original_item_id),
                        "score": float(score),
                        "rank": rank
                    })

                all_results.append({
                    "user_id": str(user_id),
                    "user_history": history,
                    "recommendations": rec_items
                })

        return all_results

    def generate_from_test_set(
        self,
        top_k: int = 10,
        include_scores: bool = True,
        batch_size: int = 256,
        user_subset: Optional[List[Union[str, int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for users in the test set.

        Args:
            top_k (int): Number of recommendations per user
            include_scores (bool): Whether to include confidence scores
            batch_size (int): Batch size for inference
            user_subset (List[Union[str, int]], optional): Subset of users to generate for

        Returns:
            List[dict]: Recommendation results for test users
        """
        # Get test dataset
        split_datasets = self.dataset.split()
        test_dataset = split_datasets['test']

        # Filter users if subset provided
        if user_subset is not None:
            # Convert user IDs to indices
            user_indices = []
            user_histories = []
            for user_id in user_subset:
                if isinstance(user_id, str):
                    # Try to find user in dataset
                    if user_id in self.dataset.user2id:
                        idx = self.dataset.user2id[user_id] - 1  # Convert to 0-based
                    else:
                        self.logger.warning(f"User {user_id} not found in dataset, skipping")
                        continue
                else:
                    idx = user_id - 1 if user_id > 0 else user_id

                if idx < len(test_dataset):
                    user_indices.append(idx)
                else:
                    self.logger.warning(f"User index {idx} out of range, skipping")

            if user_indices:
                test_dataset = test_dataset.select(user_indices)
            else:
                self.logger.warning("No valid users in subset, using all test users")

        # Prepare user histories from test set
        user_histories = []
        user_ids = []

        for idx, example in enumerate(test_dataset):
            if idx < 3:  # Debug first 3 examples
                self.logger.info(f"[DEBUG] Example {idx} keys: {list(example.keys())}")

            # item_seq may be nested: [[item1, item2, ...]] as in tokenizer
            item_seq_data = example['item_seq']
            if idx < 3:
                self.logger.info(f"[DEBUG] item_seq type: {type(item_seq_data)}, content preview: {str(item_seq_data)[:100]}")
            # Handle various item_seq formats
            user_seq = []
            if isinstance(item_seq_data, list):
                if len(item_seq_data) > 0:
                    if isinstance(item_seq_data[0], list):
                        # Nested list format: [[item1, item2, ...]]
                        user_seq = item_seq_data[0]
                        if idx < 3:
                            self.logger.info(f"[DEBUG] Nested list detected, using first element, len: {len(user_seq)}")
                    else:
                        # Flat list format: [item1, item2, ...]
                        user_seq = item_seq_data
                        if idx < 3:
                            self.logger.info(f"[DEBUG] Flat list detected, len: {len(user_seq)}")
            elif isinstance(item_seq_data, str):
                # Try to parse as JSON string
                try:
                    parsed = json.loads(item_seq_data)
                    if isinstance(parsed, list):
                        user_seq = parsed
                        if idx < 3:
                            self.logger.info(f"[DEBUG] Parsed JSON string, len: {len(user_seq)}")
                    else:
                        self.logger.warning(f"Parsed JSON is not a list: {type(parsed)}")
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse item_seq as JSON: {item_seq_data[:50]}...")
            else:
                self.logger.warning(f"Unexpected item_seq format: {type(item_seq_data)}")

            # Ensure all items are integers
            cleaned_seq = []
            for item_idx, item in enumerate(user_seq):
                if isinstance(item, (int, np.integer)):
                    cleaned_seq.append(int(item))
                elif isinstance(item, str):
                    try:
                        cleaned_seq.append(int(item))
                    except ValueError:
                        self.logger.warning(f"Failed to convert item '{item}' (position {item_idx}) to int, skipping")
                elif isinstance(item, float):
                    # Try to convert float to int if it's actually an integer
                    if item.is_integer():
                        cleaned_seq.append(int(item))
                    else:
                        self.logger.warning(f"Float item {item} at position {item_idx} is not integer, skipping")
                else:
                    self.logger.warning(f"Unexpected item type {type(item)} at position {item_idx}: {item}, skipping")

            if idx < 3 and cleaned_seq:
                self.logger.info(f"[DEBUG] Cleaned seq first 5 items: {cleaned_seq[:5]}")
                self.logger.info(f"[DEBUG] Cleaned seq types: {[type(x) for x in cleaned_seq[:5]]}")
            user_id = example['user'] if 'user' in example else len(user_ids) + 1
            user_histories.append(cleaned_seq)
            user_ids.append(user_id)

        self.logger.info(f"Generating recommendations for {len(user_histories)} test users")

        # Generate recommendations using the main generation function
        return self.generate_for_users(
            user_histories=user_histories,
            user_ids=user_ids,
            top_k=top_k,
            include_scores=include_scores,
            batch_size=batch_size
        )

    def _token_id_to_item_id(self, token_id: int) -> int:
        """
        Convert token ID back to original item ID.
        Args:
            token_id (int): Token ID from model output (1-based token IDs)
        Returns:
            int: Original item ID
        """
        # token_id 0 is padding, should not appear in predictions
        if token_id == 0:
            self.logger.warning(f"Padding token (0) found in predictions")
            return 0
        # Check if tokenizer has id2item mapping
        if hasattr(self.tokenizer, 'id2item') and self.tokenizer.id2item is not None:
            # id2item is 0-indexed list mapping token_id to original item ID
            # Model outputs 1-based token IDs (1 to n_items)
            # So we need to subtract 1 to index into id2item
            idx = token_id - 1
            if 0 <= idx < len(self.tokenizer.id2item):
                item = self.tokenizer.id2item[idx]
                # Try to convert to integer if possible
                try:
                    return int(item)
                except (ValueError, TypeError):
                    return item
            else:
                self.logger.warning(f"Token ID {token_id} (idx={idx}) out of range for id2item (len={len(self.tokenizer.id2item)})")
                # Fallback: try direct lookup if token_id is already original item ID
                return token_id
        else:
            # Fallback: assume token_id is item_id (for backward compatibility)
            self.logger.debug(f"No id2item mapping available, using token_id as item_id")
            return token_id

    def save_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save recommendations to JSON file.

        Args:
            recommendations (List[dict]): Recommendation results
            output_path (str): Path to output JSON file
            metadata (dict, optional): Additional metadata to include
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}

        # Add default metadata
        default_metadata = {
            "model": self.config.get('model', 'RPG'),
            "dataset": self.config.get('dataset', 'Netease'),
            "checkpoint": self.config.get('checkpoint_path', 'unknown'),
            "generation_time": datetime.now().isoformat(),
            "top_k": self.config.get('top_k', 10),
            "include_scores": self.config.get('include_scores', True),
            "use_graph_decoding": self.config.get('use_graph_decoding', False),
            "total_users": len(recommendations)
        }

        # Update with provided metadata
        default_metadata.update(metadata)

        # Prepare output structure
        output_data = {
            "metadata": default_metadata,
            "recommendations": recommendations
        }

        # Save to file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved recommendations to {output_path}")
        self.logger.info(f"Total users: {len(recommendations)}")

        # Log generation summary to wandb
        if self.config.get('wandb_run') is not None:
            try:
                # Calculate recommendation quality metrics
                quality_metrics = self._calculate_recommendation_quality(recommendations)

                # Record quality metrics
                wandb.log(quality_metrics)

                # Record summary metrics
                summary_metrics = {
                    'generation/total_users': len(recommendations),
                    'generation/total_recommendations': len(recommendations) * default_metadata.get('top_k', 10),
                    'generation/output_path': output_path,
                    'generation/checkpoint': default_metadata.get('checkpoint', 'unknown'),
                }
                wandb.log(summary_metrics)

                # Update wandb run summary
                for key, value in quality_metrics.items():
                    wandb.run.summary[key] = value
                wandb.run.summary['generation/total_users'] = len(recommendations)
                wandb.run.summary['generation/output_path'] = output_path
                wandb.run.summary['generation/completed_at'] = default_metadata.get('generation_time',
                                                                                    datetime.now().isoformat())

                # Save recommendations file as wandb artifact
                artifact = wandb.Artifact(
                    name=f"recommendations-{os.path.basename(output_path).replace('.json', '')}",
                    type="recommendations",
                    description=f"Generated recommendations for {len(recommendations)} users"
                )
                artifact.add_file(output_path)
                wandb.log_artifact(artifact)

                self.logger.info(f"Logged generation summary to wandb (run: {wandb.run.name})")

            except Exception as e:
                self.logger.warning(f"Failed to log generation summary to wandb: {e}")

    def _prepare_batch(self, user_histories: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for model inference.

        Args:
            user_histories (List[List[int]]): List of user histories

        Returns:
            dict: Batch dictionary for model input
        """
        # Debug: check input format and tokenizer mapping
        if len(user_histories) > 0:
            self.logger.info(f"[DEBUG] _prepare_batch: processing {len(user_histories)} histories")

            # Check tokenizer mapping
            if hasattr(self.tokenizer, 'item2id'):
                self.logger.info(f"[DEBUG] Tokenizer item2id type: {type(self.tokenizer.item2id)}")
                # Check first few keys
                keys = list(self.tokenizer.item2id.keys())
                if len(keys) > 0:
                    self.logger.info(f"[DEBUG] First 3 item2id keys: {keys[:3]}, types: {[type(k) for k in keys[:3]]}")
                    self.logger.info(f"[DEBUG] First 3 item2id values: {[self.tokenizer.item2id[k] for k in keys[:3]]}")
                # Check if tokenizer has vocab_size
                if hasattr(self.tokenizer, 'vocab_size'):
                    self.logger.info(f"[DEBUG] Tokenizer vocab_size: {self.tokenizer.vocab_size}")
            for i, history in enumerate(user_histories[:3]):  # Check first 3
                self.logger.info(f"[DEBUG]   History {i}: type={type(history)}, len={len(history) if hasattr(history, '__len__') else 'N/A'}")
                if len(history) > 0:
                    self.logger.info(f"[DEBUG]     First element: type={type(history[0])}, value={history[0]}")
                    # Check if all elements are int
                    non_int = [item for item in history if not isinstance(item, (int, np.integer))]
                    if non_int:
                        self.logger.warning(f"[DEBUG]     Non-integer elements found: {non_int[:5]}...")
        max_seq_len = self.config.get('max_item_seq_len', 50)

        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []

        for history in user_histories:
            # Use the tokenizer's tokenization logic
            # For inference, we want to predict next item given history
            # Convert item IDs to token IDs using tokenizer mapping
            tokenized_history = []
            for item_id in history:
                # Map item_id to token_id
                # Note: tokenizer expects item IDs as keys, not indices
                item_str = str(item_id)  # tokenizer可能使用字符串item ID
                token_id = None

                # Try string key first (most common)
                if item_str in self.tokenizer.item2id:
                    token_id = self.tokenizer.item2id[item_str]
                # Fallback: try integer key
                elif item_id in self.tokenizer.item2id:
                    token_id = self.tokenizer.item2id[item_id]
                else:
                    # Unknown item, use padding token (0) as fallback
                    self.logger.warning(f"Item {item_id} not found in tokenizer.item2id, using padding token")
                    token_id = 0

                # Check if token_id is within valid range for items
                # token_id should be between 0 (padding) and len(id2item)-1 (last item)
                if hasattr(self.tokenizer, 'id2item') and self.tokenizer.id2item is not None:
                    max_item_id = len(self.tokenizer.id2item)  # n_items, id2item indices: 0 to n_items-1
                    if token_id >= max_item_id and token_id != 0:
                        self.logger.error(
                            f"Token ID {token_id} exceeds max item ID {max_item_id - 1}, using padding token")
                        token_id = 0
                tokenized_history.append(token_id)
            if len(tokenized_history) >= max_seq_len + 1:
                # Truncate to max_seq_len + 1 (last item is target)
                truncated_history = tokenized_history[-(max_seq_len + 1):]
            else:
                truncated_history = tokenized_history

            # Prepare input sequence (all but last item)
            input_seq = truncated_history[:-1]
            seq_len = len(input_seq)

            # Pad input sequence
            padded_input = input_seq + [0] * (max_seq_len - seq_len)
            attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)

            # For inference, labels are ignored (set to -100)
            labels = [-100] * max_seq_len
            # Last position is the target (but we're predicting it)
            if seq_len > 0:
                labels[seq_len - 1] = truncated_history[-1]  # Last item as target

            all_input_ids.append(padded_input)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_seq_lens.append(seq_len)

        # Convert to tensors with validation
        try:
            input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Failed to create input_ids tensor: {e}")
            self.logger.error(
                f"all_input_ids sample (first 3): {all_input_ids[:3] if len(all_input_ids) >= 3 else all_input_ids}")
            # Check for non-integer values
            for i, seq in enumerate(all_input_ids[:5]):
                non_int = [item for item in seq if not isinstance(item, (int, np.integer))]
                if non_int:
                    self.logger.error(
                        f"  Sequence {i} has non-integer values at indices: {[j for j, item in enumerate(seq) if not isinstance(item, (int, np.integer))][:10]}")
            raise

        try:
            attention_mask_tensor = torch.tensor(all_attention_mask, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Failed to create attention_mask tensor: {e}")
            raise

        try:
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Failed to create labels tensor: {e}")
            raise

        try:
            seq_lens_tensor = torch.tensor(all_seq_lens, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Failed to create seq_lens tensor: {e}")
            raise
        batch = {
            'input_ids': input_ids_tensor.to(self.device),
            'attention_mask': attention_mask_tensor.to(self.device),
            'labels': labels_tensor.to(self.device),
            'seq_lens': seq_lens_tensor.to(self.device)
        }

        return batch