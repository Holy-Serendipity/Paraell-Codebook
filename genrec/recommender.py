import os
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
from logging import getLogger
from datetime import datetime

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.dataset import AbstractDataset
from genrec.utils import get_config, init_seed, init_logger, init_device, log


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
            model.load_state_dict(torch.load(checkpoint_path, map_location=config['device']))
            logger.info(f"Loaded model checkpoint from {checkpoint_path}")
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
                # Use return_scores if available, otherwise generate and compute scores separately
                if hasattr(self.model, 'generate_with_scores'):
                    preds, scores = self.model.generate_with_scores(batch, n_return_sequences=top_k)
                elif hasattr(self.model.generate, '__code__') and 'return_scores' in self.model.generate.__code__.co_varnames:
                    preds, scores = self.model.generate(batch, n_return_sequences=top_k, return_scores=True)
                else:
                    # Fallback: generate without scores
                    preds = self.model.generate(batch, n_return_sequences=top_k)
                    scores = torch.ones_like(preds, dtype=torch.float32)  # Placeholder scores
            else:
                preds = self.model.generate(batch, n_return_sequences=top_k)
                scores = None

        # Convert tensors to Python lists
        recommendations = preds.squeeze(-1).cpu().tolist()[0]  # Shape: (1, top_k) -> list
        if scores is not None:
            score_list = scores.squeeze(-1).cpu().tolist()[0]
        else:
            score_list = [1.0] * len(recommendations)  # Default scores

        # Format recommendations
        rec_items = []
        for i, (item_id, score) in enumerate(zip(recommendations, score_list)):
            rec_items.append({
                "item_id": int(item_id),
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

        # Process in batches
        for i in tqdm(range(0, len(user_histories), batch_size), desc="Generating recommendations"):
            batch_histories = user_histories[i:i+batch_size]
            batch_user_ids = user_ids[i:i+batch_size]

            # Prepare batch
            batch = self._prepare_batch(batch_histories)

            with torch.no_grad():
                # Generate recommendations for batch
                if include_scores:
                    if hasattr(self.model, 'generate_with_scores'):
                        preds, scores = self.model.generate_with_scores(batch, n_return_sequences=top_k)
                    elif hasattr(self.model.generate, '__code__') and 'return_scores' in self.model.generate.__code__.co_varnames:
                        preds, scores = self.model.generate(batch, n_return_sequences=top_k, return_scores=True)
                    else:
                        preds = self.model.generate(batch, n_return_sequences=top_k)
                        scores = torch.ones_like(preds, dtype=torch.float32)
                else:
                    preds = self.model.generate(batch, n_return_sequences=top_k)
                    scores = None

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
                    rec_items.append({
                        "item_id": int(item_id),
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

        for example in test_dataset:
            user_seq = example['item_seq']
            user_id = example['user'] if 'user' in example else len(user_ids) + 1
            user_histories.append(user_seq)
            user_ids.append(user_id)

        self.logger.info(f"Generating recommendations for {len(user_histories)} test users")

        return self.generate_for_users(
            user_histories=user_histories,
            user_ids=user_ids,
            top_k=top_k,
            include_scores=include_scores,
            batch_size=batch_size
        )

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

    def _prepare_batch(self, user_histories: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for model inference.

        Args:
            user_histories (List[List[int]]): List of user histories

        Returns:
            dict: Batch dictionary for model input
        """
        max_seq_len = self.config.get('max_item_seq_len', 50)

        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []

        for history in user_histories:
            # Use the tokenizer's tokenization logic
            # For inference, we want to predict next item given history
            if len(history) >= max_seq_len + 1:
                # Truncate to max_seq_len + 1 (last item is target)
                truncated_history = history[-(max_seq_len + 1):]
            else:
                truncated_history = history

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

        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long).to(self.device),
            'labels': torch.tensor(all_labels, dtype=torch.long).to(self.device),
            'seq_lens': torch.tensor(all_seq_lens, dtype=torch.long).to(self.device)
        }

        return batch