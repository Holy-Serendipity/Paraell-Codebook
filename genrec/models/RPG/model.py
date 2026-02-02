import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # Initialize as an identity mapping
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        self.item_id_embedding=nn.Embedding(
            num_embeddings=self.dataset.n_items+1,
            embedding_dim=config['n_embd'],
            padding_idx=0
        )
        self.fusion_gate = nn.Linear(config['n_embd'] * 2, config['n_embd'])
        self.id_scale = nn.Parameter(torch.tensor(0.1))
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            # pred_head_list.append(ResBlock(self.config['n_embd']))
            if i<64:
                head=ResBlock(config['n_embd'])
            else:
                head=nn.Linear(config['n_embd'], config['n_embd'])
            pred_head_list.append(head)
        self.pred_heads = nn.ModuleList(pred_head_list)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        # Group by layer
        self.num_groups = config['num_groups']
        self.group_temperature = self.config['group_temperature']
        self.group_loss_weight = self.config['group_loss_weight']
        self.consistency_loss_weight = self.config['consistency_loss_weight']
        #Group attention
        self.group_attention = nn.Linear(config['n_embd'], self.num_groups)
        #Group prototype
        self.group_prototype = nn.Parameter(torch.randn(self.num_groups, config['n_embd']))
        nn.init.normal_(self.group_prototype, mean=0.0, std=0.02)
        #Group representation project network
        # self.group_projection = nn.Linear(config['n_embd'], config['n_embd'])
        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'Embedding parameters: {emb_params}\n' \
                f'Non-embedding parameters: {total_params - emb_params}\n' \
                f'Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_semantic_tokens = self.item_id2tokens[batch['input_ids']]
        semantic_embs = self.gpt2.wte(input_semantic_tokens).mean(dim=-2)

        id_embs = self.item_id_embedding(batch['input_ids'])

        attention_mask = batch['attention_mask']
        semantic_embs=semantic_embs*attention_mask.unsqueeze(-1)
        id_embs=id_embs*attention_mask.unsqueeze(-1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([semantic_embs, id_embs], dim=-1)))
        input_embs = gate * semantic_embs + (1 - gate) * id_embs

        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        # final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        combine_states = outputs.last_hidden_state+self.id_scale*id_embs
        final_states=[self.pred_heads[i](combine_states).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        bs, seq_len, num_heads, h_dim = final_states.shape
        outputs.final_states = final_states
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, num_heads, h_dim)[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, num_heads, dim=1)
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, num_heads, dim=0)
            token_logits = [torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                for i in range(self.n_pred_head)
            ]
            # outputs.loss = torch.mean(torch.stack(losses))
            #token level losses
            token_level_loss = torch.mean(torch.stack(losses))
            #contrastive group losses

            # a. calculate degree of
            attn_weights = self.group_attention(final_states.view(-1, h_dim))
            attn_weights = attn_weights.view(bs, seq_len, num_heads, self.num_groups)
            attn_weights = F.softmax(attn_weights, dim=-1)

            #b. create group representation
            group_reps = torch.einsum('bsng, bsnd->bsgd', attn_weights, final_states)
            group_reps = F.normalize(group_reps, dim=-1)

            #c.calculate group contrastive loss
            # 同一批次中，门控值相近的物品，组表示应更相似
            # 将每个序列最后一个位置（预测位置）的组表示取出来
            last_positions = (batch['attention_mask']!=0).sum(dim=1)-1
            last_group_reps = group_reps[torch.arange(bs), last_positions]
            #calculate the gate similarity among all items in batch （cos）
            # last_gate = gate[torch.arange(bs), last_positions].squeeze()
            # gate_similarity = torch.matmul(F.normalize(last_gate.unsqueeze(1), dim=1),
            #                                F.normalize(last_group_reps.unsqueeze(0), dim=1))
            # 门控相似度高于阈值0.9作为正样本
            # pos_mask = (gate_similarity > 0.9).float()
            # pos_mask.fill_diagonal_(0)
            overall_rep = torch.mean(last_group_reps, dim=1)
            # overall_sim = torch.matmul(overall_rep, overall_rep.T)
            # gate value and similarity
            last_gate = gate[torch.arange(bs), last_positions]
            # last_gate_scalar = last_gate.mean(dim=1, keepdim=True)
            gate_norm = F.normalize(last_gate, dim=-1)
            gate_similarity = gate_norm @ gate_norm.T
            # id embedding
            last_id_emb = id_embs[torch.arange(bs), last_positions]
            id_similarity = torch.matmul(F.normalize(last_id_emb, dim=-1),
                                         F.normalize(last_id_emb, dim=-1).T)
            # combine weight
            combined_sim = 0.4*gate_similarity + 0.6*id_similarity

            temperature = self.group_temperature
            logits = torch.matmul(overall_rep, overall_rep.T) / temperature
            # 根据阈值构建标签
            eye_mask = torch.eye(bs, device=logits.device).bool()
            logits_masked = logits.masked_fill(eye_mask, float('-inf'))
            pos_threshold = self.config['pos_threshold']
            # labels = torch.zeros(bs, bs, device=overall_rep.device)
            # labels[combined_sim>pos_threshold] = 1.0
            labels = (combined_sim > pos_threshold).float()

            #排除对角线
            # mask=torch.eye(bs, device=overall_rep.device).bool()
            labels[eye_mask] = 0.0

            # k = max(1, bs // 10)
            # top_idx = torch.topk(combined_sim, k=k, dim=1)
            # pos_mask = torch.zeros_like(combined_sim, dtype=torch.bool)
            # pos_mask.scatter(1, top_idx, True)
            # pos_mask[eye_mask] = False

            #计算对比损失
            pos_mask = labels.bool()
            neg_mask = ~pos_mask & ~eye_mask
            log_probs = F.log_softmax(logits_masked, dim=1)
            probs = log_probs.exp()
            eps = 1e-8
            #正样本损失
            # pos_logits = logits[pos_mask]
            # pos_loss = -torch.log(torch.exp(pos_logits)/
            #                       torch.sum(torch.exp(logits), dim=1)[pos_mask.any(dim=1)]).mean()
            if pos_mask.any():
                pos_loss = -(log_probs[pos_mask].mean())
            else:
                pos_loss = torch.tensor(0.0, device=logits.device)
            #负样本损失
            # neg_logits = logits[neg_mask]
            # neg_loss = -torch.log(1-torch.exp(neg_logits)/
            #                       torch.sum(torch.exp(logits), dim=1)[neg_mask.any(dim=1)]).mean()
            if neg_mask.any():
                neg_loss = -(torch.log(torch.clamp(1.0 - probs[neg_mask], min=eps)).mean())
            else:
                neg_loss = torch.tensor(0.0, device=logits.device)
            group_contrast_loss = pos_loss + neg_loss*0.1

            #组间多样性损失
            # group_cov = torch.matmul(last_group_reps.transpose(1,2), last_group_reps) # [bs, num_groups, num_groups]
            # group_std = torch.std(last_group_reps, dim=[0,1])
            # diversity_loss = -torch.mean(group)
            group_similarity = torch.matmul(last_group_reps, last_group_reps.transpose(1,2))#[b,n,n]
            eye_mask=torch.eye(self.num_groups,device=last_group_reps.device).bool()
            eye_mask =eye_mask.unsqueeze(0).expand(bs, -1, -1)

            off_diag_similarity = group_similarity[~eye_mask].view(bs, self.num_groups*(self.num_groups-1))
            diversity_loss = off_diag_similarity.mean()
            #组内一致性损失
            attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
            consistency_loss = -attn_entropy.mean()

            # print(f"Token-level loss: {token_level_loss.item():.4f}")
            # print(f"Group contrast loss: {group_contrast_loss.item():.4f}")
            # print(f"Consistency loss: {consistency_loss.item():.4f}")
            # print(f"Diversity loss: {diversity_loss.item():.4f}")
            # print(f"Group loss weight: {self.group_loss_weight}")
            # print(
            #     f"Contribution: token={token_level_loss.item():.4f}, group={self.group_loss_weight * group_contrast_loss.item():.4f}")
            #
            # print(f"\n梯度检查:")
            # print(f"- token_loss requires_grad: {token_level_loss.requires_grad}")
            # print(f"- group_loss requires_grad: {group_contrast_loss.requires_grad}")

            # # 3. 检查分组注意力是否有意义
            # # 计算注意力权重的熵（平均每个码本的分配明确程度）
            # attn_entropy_per_head = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10),
            #                                    dim=-1)  # [bs, seq_len, 128]
            # avg_entropy = attn_entropy_per_head.mean().item()
            # max_entropy = torch.log(torch.tensor(self.num_groups, dtype=torch.float)).item()
            # print(f"\n注意力熵: {avg_entropy:.4f} (最大可能值: {max_entropy:.4f})")
            # print(f"注意力明确度: {(1 - avg_entropy / max_entropy) * 100:.1f}%")
            #
            # # 4. 检查组表示是否有区分度
            # group_std = torch.std(last_group_reps, dim=[0, 1]).mean().item()  # 组表示的方差
            # print(f"\n组表示标准差: {group_std:.4f}")
            #
            # # 5. 检查正负样本比例
            # pos_ratio = pos_mask.float().mean().item()
            # print(f"正样本比例: {pos_ratio * 100:.2f}%")
            # print("=" * 40)

            # if self.training and random.random() < 0.01:  # 随机采样1%的批次打印，避免日志过多
            #     print("\n=== combined_sim 统计监控 ===")
            #
            #     # 基础统计
            #     print(f"combined_sim 形状: {combined_sim.shape}")
            #     print(f"最小值: {combined_sim.min().item():.6f}")
            #     print(f"平均值: {combined_sim.mean().item():.6f}")
            #     print(f"中位数: {combined_sim.median().item():.6f}")
            #     print(f"最大值: {combined_sim.max().item():.6f}")
            #     print(f"标准差: {combined_sim.std().item():.6f}")
            #
            #     # 分布直方图（分桶统计）
            #     bins = 20
            #     hist = torch.histc(combined_sim.flatten(), bins=bins, min=-1.0, max=1.0)
            #     hist_normalized = hist / hist.sum()
            #     print("\n分布直方图 (-1.0 到 1.0):")
            #     for i in range(bins):
            #         bin_min = -1.0 + i * (2.0 / bins)
            #         bin_max = -1.0 + (i + 1) * (2.0 / bins)
            #         print(f"  [{bin_min:.2f}, {bin_max:.2f}): {hist_normalized[i].item() * 100:5.1f}%")
            #
            #     # 门控相似度统计
            #     print(f"\ngate_similarity 统计:")
            #     print(f"  最小值: {gate_similarity.min().item():.6f}")
            #     print(f"  平均值: {gate_similarity.mean().item():.6f}")
            #     print(f"  最大值: {gate_similarity.max().item():.6f}")
            #
            #     # ID相似度统计
            #     print(f"\nid_similarity 统计:")
            #     print(f"  最小值: {id_similarity.min().item():.6f}")
            #     print(f"  平均值: {id_similarity.mean().item():.6f}")
            #     print(f"  最大值: {id_similarity.max().item():.6f}")
            #
            #     # 检查正样本比例（使用当前阈值）
            #     current_threshold = pos_threshold  # 假设pos_threshold已定义
            #     pos_ratio_current = (combined_sim > current_threshold).float().mean().item()
            #     print(f"\n当前阈值 {current_threshold} 下的正样本比例: {pos_ratio_current * 100:.2f}%")
            #
            #     # 测试多个阈值下的正样本比例
            #     print("不同阈值下的正样本比例:")
            #     thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
            #     for thresh in thresholds:
            #         pos_ratio = (combined_sim > thresh).float().mean().item()
            #         print(f"  阈值 {thresh}: {pos_ratio * 100:5.1f}%")
            #
            #     # 排除对角线后的统计
            #     diag_mask = torch.eye(bs, dtype=torch.bool, device=combined_sim.device)
            #     combined_sim_no_diag = combined_sim[~diag_mask].view(bs, bs - 1)
            #     print(f"\n排除对角线后:")
            #     print(f"  最小值: {combined_sim_no_diag.min().item():.6f}")
            #     print(f"  平均值: {combined_sim_no_diag.mean().item():.6f}")
            #     print(f"  最大值: {combined_sim_no_diag.max().item():.6f}")
            #
            #     print("=" * 50)

            total_loss = token_level_loss + \
                         self.group_loss_weight * group_contrast_loss + \
                         self.consistency_loss_weight * consistency_loss + \
                         0.05 * diversity_loss
            outputs.loss = total_loss
            # 可选：记录各项损失用于监控
            outputs.token_loss = token_level_loss
            outputs.group_contrast_loss = group_contrast_loss
            outputs.consistency_loss = consistency_loss
        return outputs

    def build_ii_sim_mat(self, threshold=0.5, use_half=False):

        # 获取模型所在的设备
        device = self.gpt2.device if hasattr(self.gpt2, "device") else next(self.parameters()).device

        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size
        chunk_size=self.chunk_size
        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1)
        if use_half:
            token_embs = token_embs.to(torch.float16)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        indices=[]
        values=[]
        item_tokens=self.item_id2tokens
        # 4) Prepare an output similarity matrix
        # item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = item_tokens[i_start:i_end].to(device)  # sub-block for items i
            bi = i_end-i_start
            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = item_tokens[j_start:j_end].to(device)  # sub-block for items j
                bj=j_end-j_start

                dtype_block = torch.float16 if use_half else torch.float32
                sum_block = torch.zeros((bi, bj), device=device, dtype=dtype_block)
                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                # block_size_i = i_end - i_start
                # block_size_j = j_end - j_start
                # sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                avg_block = sum_block / n_digit.to(torch.float32)
                j_offsets = torch.arange(bj, device=device)
                global_j = j_start + j_offsets
                for local_i in range(bi):
                    global_i = i_start + local_i
                    mask = (avg_block[local_i] > threshold) & (global_j >= global_i)
                    if mask.any():
                        js = global_j[mask]
                        vs = avg_block[local_i, mask]
                        # 收集 CPU 列表（避免GPU大占用）
                        indices.extend([[global_i, int(j.item())] for j in js])
                        values.extend([float(v.item()) for v in vs])
                # mask=(avg_block > threshold) & (torch.arange(block_size_i, device=device).unsqueeze(1) <= torch.arange(block_size_j,device=device).unsqueeze(0))
                # i_indices,j_indices = torch.where(mask)
                # global_i=i_start+i_indices
                # global_j=j_start+j_indices
                # sim_values=avg_block[mask]
                # for idx in range(len(global_i)):
                #     indices.append([global_i[idx].item(),global_j[idx].item()])
                #     values.append(sim_values[idx].item())
                del tokens_j, avg_block, sum_block
            del tokens_i
        if indices:
            idx_t = torch.tensor(indices, dtype=torch.int64).t()  # (2,nnz)
            val_t = torch.tensor(values, dtype=torch.float32)
            return torch.sparse_coo_tensor(idx_t, val_t, (n_items, n_items)).coalesce()
        else:
            return torch.sparse_coo_tensor(torch.empty((2,0),dtype=torch.int64),
                                           torch.empty((0,), dtype=torch.float32),
                                           size=(n_items, n_items)).coalesce()
                # Write back into the final item_item_sim
        #         item_item_sim[i_start:i_end, j_start:j_end] = avg_block
        #
        # return item_item_sim

    def build_adjacency_list(self, item_item_sim=None):
        K=self.n_edges
        device=self.gpt2.device if hasattr(self.gpt2, "device") else next(self.parameters()).device
        if item_item_sim is None:
            adj_idx, _ = self.build_ii_topk_adjacency(use_threshold=False, threshold=0.5, use_half=False)
            return adj_idx
        if item_item_sim.is_sparse:
        #     item_item_sim_dense=item_item_sim.todense()
        # else:
        #     item_item_sim_dense = item_item_sim
        # return torch.topk(item_item_sim_dense, k=self.n_edges, dim=-1).indices
            coo = item_item_sim.coalesce()
            indices = coo.indices().cpu()  # (2, nnz)
            values = coo.values().cpu()  # (nnz,)
            n_items = coo.size(0)

            buckets = [[] for _ in range(n_items)]
            for e in range(values.numel()):
                i = int(indices[0, e])
                j = int(indices[1, e])
                v = float(values[e])
                buckets[i].append((j, v))
                if i != j:
                    buckets[j].append((i, v))

            adjacency_indices = torch.zeros((n_items, K), dtype=torch.int64)
            for i in range(n_items):
                if not buckets[i]:
                    adjacency_indices[i] = i
                    continue
                js, vs = zip(*buckets[i])
                js_t = torch.tensor(js, dtype=torch.int64)
                vs_t = torch.tensor(vs, dtype=torch.float32)
                if js_t.numel() < K:
                    pad = K - js_t.numel()
                    js_t = torch.cat([js_t, torch.full((pad,), i, dtype=torch.int64)])
                    vs_t = torch.cat([vs_t, torch.ones((pad,), dtype=torch.float32)])
                _, topi = torch.topk(vs_t, k=K, largest=True, sorted=True)
                adjacency_indices[i] = js_t[topi]
            return adjacency_indices.to(device)
        else:
            return torch.topk(item_item_sim, k=K, dim=-1).indices
    def build_ii_topk_adjacency(self, use_threshold=False, threshold=0.5, use_half=False):
        device= self.gpt2.device if hasattr(self, 'gpt2') else next(self.parameters()).device

        n_items = self.dataset.n_items
        n_digit=self.tokenizer.n_digit
        codebook_size=self.tokenizer.codebook_size
        K=self.n_edges
        chunk_size=self.chunk_size

        # 1) 预处理token相似度表（n_digit, codebook_size, codebook_size）
        token_embs=self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        if use_half:
            token_embs=token_embs.to(torch.float16)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)

        # 2）在CPU上准备候选收集容器
        adjacency_js = torch.full((n_items, K), -1, dtype=torch.int64)  # 邻居索引
        adjacency_vs = torch.full((n_items, K), float("-inf"), dtype=torch.float32)  # 相似度
        item_tokens=self.item_id2tokens.cpu()
        # row_candidates_idx=[None]*n_items
        # row_candidates_val=[None]*n_items

        # 3) 上三角分块计算并收集候选
        for i_start in range(1, n_items, chunk_size):
            i_end = min(i_start + chunk_size, n_items)
            bi = i_end - i_start
            tokens_i = item_tokens[i_start:i_end]
            tokens_i_dev=tokens_i.to(device, non_blocking=True)

            for j_start in range(i_start, n_items, chunk_size):
                j_end = min(j_start + chunk_size, n_items)
                bj=j_end - j_start
                tokens_j=item_tokens[j_start:j_end]
                tokens_j_dev=tokens_j.to(device, non_blocking=True)

                #计算块内相似度
                dtype_block=torch.float16 if use_half else torch.float32
                sum_block=torch.zeros((bi,bj), device=device, dtype=dtype_block)

                for k in range(n_digit):
                    # 映射到局部索引并计算相似度
                    row_inds = tokens_i_dev[:, k] - k * codebook_size - 1
                    col_inds = tokens_j_dev[:, k] - k * codebook_size - 1
                    temp = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                    sum_block += temp
                avg_block = (sum_block / n_digit).to(torch.float32)

                # 应用阈值过滤
                if use_threshold:
                    mask = avg_block > threshold
                else:
                    mask = None
                j_offsets = torch.arange(bj, device=device)
                global_j = j_start + j_offsets
                # 收集候选对
                with torch.no_grad():
                    for local_i in range(bi):
                        global_i = i_start + local_i

                        row_vals=avg_block[local_i]
                        row_js=global_j
                        tri_mask=(row_js>=global_i)
                        if mask is not None:
                            tri_mask = tri_mask & mask[local_i]

                        if tri_mask.any():
                            row_vals = row_vals[tri_mask]
                            row_js = row_js[tri_mask]
                        else:
                            continue
                        # row_mask = mask[local_i]
                        # j_offsets = torch.arange(bj, device=device)
                        # global_j = j_start + j_offsets
                        # 限制到上三角（j >= i）
                        # row_mask = row_mask & (global_j >= global_i)
                        # if not row_mask.any():
                        #     continue
                        # global_js = global_j[row_mask].to(torch.int64)
                        # sim_vals = avg_block[local_i, row_mask].to(torch.float32)
                        # # 保存到候选列表
                        # cpu_js = global_js.detach().cpu()
                        # cpu_vals = sim_vals.detach().cpu()
                        # if row_candidates_idx[global_i] is None:
                        #     row_candidates_idx[global_i] = cpu_js
                        #     row_candidates_val[global_i] = cpu_vals
                        # else:
                        #     row_candidates_idx[global_i] = torch.cat([row_candidates_idx[global_i], cpu_js], dim=0)
                        #     row_candidates_val[global_i] = torch.cat([row_candidates_val[global_i], cpu_vals], dim=0)
                            # 块内局部 top-K（限制最多 K 个）
                        if row_vals.numel() > K:
                            blk_top_vals, blk_top_idx = torch.topk(row_vals, k=K, largest=True)
                            blk_top_js = row_js[blk_top_idx]
                        else:
                            blk_top_vals = row_vals
                            blk_top_js = row_js
                        # 与全局缓冲归并：拼接再取 top-K（最多 2K）
                        merge_js = torch.cat([adjacency_js[global_i], blk_top_js.cpu()], dim=0)
                        merge_vs = torch.cat([adjacency_vs[global_i], blk_top_vals.cpu()], dim=0)
                        top_vals, top_idx = torch.topk(merge_vs, k=K, largest=True)
                        adjacency_js[global_i] = merge_js[top_idx]
                        adjacency_vs[global_i] = top_vals
                for local_j in range(bj):
                    global_j = j_start + local_j
                    col_vals = avg_block[:, local_j]  # (bi,)
                    col_is = torch.arange(bi, device=device) + i_start
                    tri_mask2 = (col_is <= global_j)
                    if use_threshold:
                        tri_mask2 = tri_mask2 & (avg_block[:, local_j] > threshold)
                    if tri_mask2.any():
                        col_vals = col_vals[tri_mask2]
                        col_is = col_is[tri_mask2]
                    else:
                        continue
                    if col_vals.numel() > K:
                        blk_top_vals2, blk_top_idx2 = torch.topk(col_vals, k=K, largest=True)
                        blk_top_is2 = col_is[blk_top_idx2]
                    else:
                        blk_top_vals2 = col_vals
                        blk_top_is2 = col_is
                    merge_js2 = torch.cat([adjacency_js[global_j], blk_top_is2.cpu()], dim=0)
                    merge_vs2 = torch.cat([adjacency_vs[global_j], blk_top_vals2.cpu()], dim=0)
                    top_vals2, top_idx2 = torch.topk(merge_vs2, k=K, largest=True)
                    adjacency_js[global_j] = merge_js2[top_idx2]
                    adjacency_vs[global_j] = top_vals2
                # 清理中间变量释放显存
                del tokens_j_dev, avg_block, sum_block
            del tokens_i_dev
        # 4) 对称化：将 (i,j) 添加到 j 的候选列表
        # for i in range(n_items):
        #     js = row_candidates_idx[i]
        #     vals = row_candidates_val[i]
        #
        #     if js is None:
        #         continue
        #
        #     for idx in range(js.numel()):
        #         j = int(js[idx])
        #         if j == i:
        #             continue
        #
        #         v = vals[idx:idx + 1]
        #         j_tensor = torch.tensor([i], dtype=torch.int64)
        #
        #         if row_candidates_idx[j] is None:
        #             row_candidates_idx[j] = j_tensor
        #             row_candidates_val[j] = v
        #         else:
        #             row_candidates_idx[j] = torch.cat([row_candidates_idx[j], j_tensor], dim=0)
        #             row_candidates_val[j] = torch.cat([row_candidates_val[j], v], dim=0)
        # 5) 每行进行 top-K 选择，生成最终的邻接矩阵
        # adjacency_indices = torch.zeros((n_items, K), dtype=torch.int64)
        # adjacency_values = torch.zeros((n_items, K), dtype=torch.float32)
        # for i in range(n_items):
        #     js = row_candidates_idx[i]
        #     vals = row_candidates_val[i]
        #
        #     # 处理没有候选的情况
        #     if js is None or js.numel() == 0:
        #         adjacency_indices[i] = i
        #         adjacency_values[i] = 1.0
        #         continue
        #
        #     # 候选不足时填充自身
        #     if js.numel() < K:
        #         pad_n = K - js.numel()
        #         js = torch.cat([js, torch.full((pad_n,), i, dtype=torch.int64)], dim=0)
        #         vals = torch.cat([vals, torch.ones((pad_n,), dtype=torch.float32)], dim=0)
        #
        #     # 选择 top-K
        #     topv, topi = torch.topk(vals, k=K, largest=True, sorted=True)
        #     adjacency_indices[i] = js[topi]
        #     adjacency_values[i] = topv

            # 转移到目标设备
        adjacency_indices = adjacency_js.to(device, non_blocking=True)
        adjacency_values = adjacency_vs.to(device, non_blocking=True)

        return adjacency_indices, adjacency_values
    # def init_graph(self):
    #     self.tokenizer.log("Building item-item similarity matrix...")
    #     item_item_sim = self.build_ii_sim_mat()
    #     self.adjacency = self.build_adjacency_list(item_item_sim)
    #     self.tokenizer.log("Graph initialized.")
    def init_graph(self):
        self.tokenizer.log("Building item-item similarity matrix...")
        adj_idx,_=self.build_ii_topk_adjacency(use_threshold=False, threshold=0.5, use_half=False)
        self.adjacency = adj_idx
        self.tokenizer.log("Graph initialized.")
    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]

        # Initialize visited nodes tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Add initial nodes to visited set
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        for sid in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add neighbors to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)

                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        else:
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),              # (batch_size, n_items, n_tokens)
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, code_dim)
            ).mean(dim=-1)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)
