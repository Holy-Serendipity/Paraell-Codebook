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
            # 设置 wandb 运行名称（如果没有提供）
            wandb_run_name = self.config.get('wandb_run_name')
            if wandb_run_name is None:
                wandb_run_name = f"{self.config['model']}-{self.config['dataset']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                self.config['wandb_run_name'] = wandb_run_name

            # 初始化 wandb
            wandb.init(
                project=self.config.get('wandb_project', 'default-project'),
                name=wandb_run_name,
                config=self.config,  # 记录所有配置参数
                dir=self.project_dir,  # 设置 wandb 目录
                resume="allow",  # 允许恢复运行
                mode=self.config.get('wandb_mode', 'online'),  # 在线或离线模式
                entity=self.config.get('wandb_entity'),  # 团队或用户名称
                tags=self.config.get('wandb_tags', [])  # 可选的标签
            )

            # 将 wandb 运行信息保存到配置中
            self.config['wandb_run'] = wandb.run
            self.config['wandb_run_id'] = wandb.run.id

            # 记录配置信息到 wandb
            wandb.config.update(self.config)

            # 创建 wandb 的配置文件
            config_file_path = os.path.join(self.project_dir, 'config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(self.config, f)

            # 将配置文件上传到 wandb
            wandb.save(config_file_path, base_path=os.path.dirname(config_file_path))

        else:
            # 非主进程设置 None
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
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device']))
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # 记录模型信息到 wandb
        if self.accelerator.is_main_process and self.config.get('wandb_run'):
            wandb.log({
                'model/parameters': self.model.n_parameters,
                'model/architecture': str(type(self.model)),
                # 'dataset/size': self.config['dataset_size'],
                # 'dataset/classes': len(self.config['dataset_classes'])
            })

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

        if self.accelerator.is_main_process and self.config.get('wandb_run'):
            wandb.log({
                'data/train_batches': len(train_dataloader),
                'data/val_batches': len(val_dataloader),
                'data/test_batches': len(test_dataloader),
                'data/train_batch_size': self.config['train_batch_size'],
                'data/eval_batch_size': self.config['eval_batch_size']
            })

        best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)

        if self.accelerator.is_main_process and self.config.get('wandb_run'):
            # 保存到 wandb 的摘要
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_val_score"] = best_val_score

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        if self.checkpoint_path is None:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

            if self.accelerator.is_main_process and self.config.get('wandb_run'):
                wandb.log({'info/best_model_loaded': 1})
                # 保存最佳模型到 wandb（可选）
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
                    # wandb's abstract
                    wandb.run.summary[f"test/{key}"] = test_results[key]

        self.log(f'Test Results: {test_results}')
        #记录完整的训练摘要到wandb
        if self.accelerator.is_main_process and self.config.get('wandb_run'):
            for key, value in test_results.items():
                wandb.run.summary[f"best_test/{key}"] = value

            # 记录完整的训练摘要
            wandb.run.summary.update({
                "best_epoch": best_epoch,
                "best_val_score": best_val_score,
                "train_batch_size": self.config['train_batch_size'],
                "eval_batch_size": self.config['eval_batch_size'],
                "model_parameters": self.model.n_parameters,
                "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        self.trainer.end()

        if self.accelerator.is_main_process and self.config.get('wandb_run') and wandb.run is not None:
            wandb.finish()

        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
