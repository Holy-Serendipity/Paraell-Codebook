import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.evaluator import Evaluator
from genrec.utils import get_file_name, get_total_steps, config_for_log, log

import wandb
class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        # self.accelerator.init_trackers(
        #     project_name=get_file_name(self.config, suffix=''),
        #     config=config_for_log(self.config),
        #     init_kwargs={"tensorboard": {"flush_secs": 60}},
        # )

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        # 记录训练开始信息
        if self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get('wandb_run'):
            wandb.log({
                'info/training_started': 1,
                'training/total_epochs': n_epochs,
                'training/total_steps': total_n_steps
            })

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )

            # 记录学习率
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.config['lr']

            grad_norms=[]
            grad_means=[]
            for batch_idx, batch in enumerate(train_progress_bar):
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                # 计算梯度统计信息
                batch_grad_norm = 0.0
                batch_grad_means = []

                for param in self.model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        param_mean = param.grad.data.mean().item()
                        batch_grad_norm += param_norm ** 2
                        batch_grad_means.append(param_mean)
                # 计算当前batch的梯度统计
                batch_grad_norm = batch_grad_norm ** 0.5
                batch_grad_mean = sum(batch_grad_means) / len(batch_grad_means) if batch_grad_means else 0

                # 存储用于epoch统计
                grad_norms.append(batch_grad_norm)
                grad_means.append(batch_grad_mean)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

                # 记录批次级别的指标（可选，避免太频繁）
                if batch_idx % 100 == 0 and self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get('wandb_run'):
                    wandb.log({
                        'batch/loss': loss.item(),
                        'batch/lr': current_lr,
                        'batch/step': epoch * len(train_dataloader) + batch_idx,
                        'batch/grad_norm': batch_grad_norm,
                        'batch/grad_mean': batch_grad_mean
                    })
            # 计算整个epoch的平均梯度统计
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
            avg_grad_mean = sum(grad_means) / len(grad_means) if grad_means else 0
            avg_train_loss = total_loss / len(train_dataloader)
            self.accelerator.log({
                "Loss/train_loss": avg_train_loss,
                "LearningRate/lr": current_lr,
                "Gradient/norm": avg_grad_norm,
                "Gradient/mean": avg_grad_mean
            }, step=epoch + 1)

            # 记录到 wandb
            if self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get('wandb_run'):
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': avg_train_loss,
                    'train/learning_rate': current_lr,
                    'train/epoch_progress': (epoch + 1) / n_epochs,
                    'train/grad_norm': avg_grad_norm,
                    'train/grad_mean': avg_grad_mean
                })

            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    # 记录到 wandb
                    if 'wandb_run' in self.config and self.config.get('wandb_run'):
                        wandb_log_data = {'epoch': epoch + 1}
                        for key, value in all_results.items():
                            wandb_log_data[f'val/{key}'] = value
                        wandb.log(wandb_log_data)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1

                    # 记录最佳结果到 wandb
                    if self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get(
                            'wandb_run'):
                        wandb.log({
                            'best/val_score': best_val_score,
                            'best/epoch': best_epoch
                        })

                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)

                        # 记录模型保存到 wandb
                        if 'wandb_run' in self.config and self.config.get('wandb_run'):
                            wandb.log({'info/model_saved': 1})

                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')

                    # 记录早停信息到 wandb
                    if self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get(
                            'wandb_run'):
                        wandb.log({'info/early_stopping': epoch + 1})

                    break

        # 记录训练完成信息
        if self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get('wandb_run'):
            wandb.log({
                'info/training_completed': 1,
                'best/final_val_score': best_val_score,
                'best/final_epoch': best_epoch
            })

        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        return best_epoch, best_val_score

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    if isinstance(preds, tuple):
                        preds, n_visited_items = preds
                        all_preds, all_labels, all_n_visited_items = self.accelerator.gather_for_metrics((preds, batch['labels'], n_visited_items))
                        all_preds = (all_preds, all_n_visited_items)
                    else:
                        all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        output_results['n_visited_items'] = torch.cat(all_results['n_visited_items']).mean().item()

        # 记录评估完成
        if self.accelerator.is_main_process and 'wandb_run' in self.config and self.config.get('wandb_run'):
            wandb.log({f'info/{split}_evaluation_completed': 1})

        return output_results

    def case_evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        diff2gap = defaultdict(list)

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                outputs = self.model.forward(batch, return_loss=False)
                states = outputs.final_states.gather(
                    dim=1,
                    index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.model.n_pred_head, self.model.config['n_embd'])
                )
                states = F.normalize(states, dim=-1)

                token_emb = self.model.gpt2.wte.weight[1:-1]
                token_emb = F.normalize(token_emb, dim=-1)
                token_embs = torch.chunk(token_emb, self.model.n_pred_head, dim=0)
                logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.model.temperature for i in range(self.model.n_pred_head)]
                logits = [F.log_softmax(logit, dim=-1) for logit in logits]
                token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

                sampled_items = torch.randint(1, self.model.item_id2tokens.shape[0], (token_logits.shape[0], 10))

                item_logits = torch.gather(
                    input=token_logits.unsqueeze(-2).expand(-1, sampled_items.shape[1], -1),              # (batch_size, n_items, n_tokens)
                    dim=-1,
                    index=(self.model.item_id2tokens[sampled_items,:] - 1)  # (batch_size, n_items, code_dim)
                ).mean(dim=-1)

                for batch_id in range(item_logits.shape[0]):
                    logit_list = item_logits[batch_id].cpu().tolist()
                    for i in range(len(logit_list)):
                        for j in range(i + 1, len(logit_list)):
                            item_a = sampled_items[batch_id, i]
                            item_b = sampled_items[batch_id, j]
                            gap = abs(logit_list[i] - logit_list[j])
                            diff = (self.model.item_id2tokens[item_a] != self.model.item_id2tokens[item_b]).sum().item()
                            diff2gap[diff].append(gap)
        return diff2gap

    def evaluate_cold_start(self, dataloader, token2item, item2group, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        group2results = {
            '0': defaultdict(list),
            '1': defaultdict(list),
            '2': defaultdict(list),
            '3': defaultdict(list),
            '4': defaultdict(list)
        }
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for i, label in enumerate(batch['labels'].cpu().tolist()):
                    if self.config['model'] == 'TIGER':
                        item_id = token2item[' '.join(list(map(str, tuple(label[:-1]))))]
                    else:
                        item_id = token2item[str(label[0])]
                    if item_id not in item2group:
                        continue
                    group = item2group[item_id]
                    for key, value in results.items():
                        group2results[group][key].append(value.cpu().tolist()[i])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        return output_results, group2results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
