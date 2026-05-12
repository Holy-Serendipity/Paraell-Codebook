from logging import getLogger
from typing import Union
import torch
import os
from datetime import datetime
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log
from genrec.recommender import Recommender
import wandb

class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device() 
        self.checkpoint_path = checkpoint_path

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with=['wandb'], project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator

        # 在主进程中初始化 wandb
        if self.accelerator.is_main_process:
            wandb_run_name = self.config.get('wandb_run_name')
            if wandb_run_name is None:
                wandb_run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.config['cache_dir'][6:12]}"
                self.config['wandb_run_name'] = wandb_run_name

            wandb.init(
                project=self.config.get('wandb_project', 'default-project'),
                name=wandb_run_name,
                config=self.config,
                dir=self.project_dir,
                resume="allow",
                mode=self.config.get('wandb_mode', 'online'),
                entity=self.config.get('wandb_entity'),
                tags=self.config.get('wandb_tags', [])
            )

            self.config['wandb_run'] = wandb.run
            self.config['wandb_run_id'] = wandb.run.id

            config_file_path = os.path.join(self.project_dir, 'config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(self.config, f)
            wandb.save(config_file_path, base_path=os.path.dirname(config_file_path))
        else:
            self.config['wandb_run'] = None
            self.config['wandb_run_id'] = None

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device']),
                                           strict=False)
                self.log(f'Loaded model checkpoint from {checkpoint_path} (strict=False due to architecture changes)')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        # DataLoader
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.tokenizer.collate_fn['train']
        )
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val']
        )
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        if self.checkpoint_path is None:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt), strict=False)

            if self.accelerator.is_main_process and self.config.get('wandb_run'):
                try:
                    artifact = wandb.Artifact(
                        name=f"best-model-{wandb.run.id}",
                        type="model",
                        description=f"Best model from epoch {best_epoch} with val_score {best_val_score:.4f}"
                    )
                    artifact.add_file(self.trainer.saved_model_ckpt)
                    wandb.log_artifact(artifact)
                except Exception as e:
                    self.log(f"Failed to save model artifact to wandb: {e}", level='warning')

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process and self.checkpoint_path is None:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        # Enable graph-constrained decoding for model inference
        self.trainer.model.generate_w_decoding_graph = True
        test_results = self.trainer.evaluate(test_dataloader)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

            if self.config.get('wandb_run'):
                summary = wandb.run.summary
                for key, value in test_results.items():
                    summary[f"test/{key}"] = value
                summary.update({
                    "best_epoch": best_epoch,
                    "best_val_score": best_val_score,
                    "model_parameters": self.model.n_parameters,
                    "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        self.log(f'Test Results: {test_results}')
        self.trainer.end()

        if self.accelerator.is_main_process and self.config.get('wandb_run'):
            wandb.finish()

        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)

    def generate_recommendations(self, output_path, top_k=10, include_scores=True,
                                use_graph_decoding=False, batch_size=256,
                                user_subset=None):
        """
        Generate recommendations using the trained model.

        Args:
            output_path (str): Path to output JSON file
            top_k (int): Number of recommendations per user
            include_scores (bool): Whether to include confidence scores
            use_graph_decoding (bool): Whether to use graph-constrained decoding
            batch_size (int): Batch size for inference
            user_subset (list): Optional list of user IDs to generate for

        Returns:
            dict: Dictionary containing generation statistics
        """
        self.log(
            f"Generating recommendations with top_k={top_k}, include_scores={include_scores}, use_graph_decoding={use_graph_decoding}")

        # Update config for graph decoding
        self.config['use_graph_decoding'] = use_graph_decoding
        # Also set model flag directly
        self.model.generate_w_decoding_graph = use_graph_decoding

        # Create recommender instance using current pipeline components
        recommender = Recommender(
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.raw_dataset
        )

        # Generate recommendations
        recommendations = recommender.generate_from_test_set(
            top_k=top_k,
            include_scores=include_scores,
            batch_size=batch_size,
            user_subset=user_subset
        )

        # Save recommendations
        metadata = {
            "generated_by": "Pipeline.generate_recommendations",
            "model": self.config.get('model', 'RPG'),
            "dataset": self.config.get('dataset', 'Netease'),
            "checkpoint": self.checkpoint_path,
            "top_k": top_k,
            "include_scores": include_scores,
            "use_graph_decoding": use_graph_decoding,
        }

        recommender.save_recommendations(recommendations, output_path, metadata)

        stats = {
            "total_users": len(recommendations),
            "top_k": top_k,
            "output_path": output_path,
            "include_scores": include_scores,
            "use_graph_decoding": use_graph_decoding,
        }

        self.log(f"Generated recommendations for {len(recommendations)} users")
        self.log(f"Output saved to: {output_path}")

        return stats

    def evaluate_only(self):
        """
        Evaluate the model on test set without training.
        Useful for evaluating a pre-trained checkpoint.

        Returns:
            dict: Evaluation results
        """
        # Prepare test dataloader
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        # Load model checkpoint if provided
        if self.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.config['device']),
                                       strict=False)
            self.log(f'Loaded model checkpoint from {self.checkpoint_path} (strict=False due to architecture changes)')

        # Prepare model and dataloader with accelerator
        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )

        # Enable graph-constrained decoding for model inference
        self.trainer.model.generate_w_decoding_graph = True

        # Evaluate
        test_results = self.trainer.evaluate(test_dataloader)

        # Log results
        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

            if self.config.get('wandb_run'):
                for key, value in test_results.items():
                    wandb.run.summary[f"test/{key}"] = value

        self.log(f'Test Results: {test_results}')

        if self.config.get('wandb_run'):
            wandb.finish()

        return {
            'test_results': test_results,
        }