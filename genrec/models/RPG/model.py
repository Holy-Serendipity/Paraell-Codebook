import torch
import torch.nn as nn
import torch.nn.functional as F
import os
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

class SwingEnhancement(nn.Module):
    """Swing算法增强语义embedding"""
    def __init__(self, config, dataset, tokenizer, embedding_layer=None):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer  # GPT-2的wte层，用于计算物品embedding

        # 增强配置
        self.swing_enhance_weight = config.get('swing_enhance_weight', 0.3)
        self.swing_neighbors = config.get('swing_neighbors', 10)
        self.enhance_type = config.get('swing_enhance_type', 'additive')

        # 可学习的增强门控（如果使用gated类型）
        if self.enhance_type == 'gated':
            self.enhance_gate = nn.Linear(config['n_embd'] * 2, config['n_embd'])

        # Swing相似度计算器
        self.swing_sim = None
        self.sim_matrix = None  # 缓存相似度矩阵
        self.all_item_embeddings = None  # 缓存所有物品的语义embedding
        self.topk_neighbors_cache = None  # 缓存每个物品的top-k邻居（索引和权重）
        self.topk_precomputed = False  # 是否已预计算top-k邻居

    def init_swing_sim(self):
        """延迟初始化Swing相似度计算器"""
        if self.swing_sim is None:
            from .swing import SwingSimilarity
            self.swing_sim = SwingSimilarity(
                dataset=self.dataset,
                alpha=self.config.get('swing_alpha', 1.0),
                min_cooccurrence=self.config.get('swing_min_cooccurrence', 2),
                device='cpu'  # 相似度矩阵在CPU上计算，避免GPU内存问题
            )

    def precompute_topk_neighbors(self):
        """预计算每个物品的top-k邻居（索引和权重）并缓存，避免前向传播中重复计算"""
        if self.topk_precomputed and self.topk_neighbors_cache is not None:
            return

        # 获取相似度矩阵（稀疏，保持在CPU）
        sim_matrix = self.get_similarity_matrix(sparse=True)

        n_items = self.dataset.n_items
        k = self.swing_neighbors
        print(f"[SwingEnhancement] Precomputing top-{k} neighbors for {n_items} items...")

        # 初始化缓存：每个物品存储(neighbor_indices, neighbor_weights)
        topk_cache = {}

        # 对于稀疏矩阵，逐行处理
        if sim_matrix.is_sparse:
            sim_matrix = sim_matrix.coalesce()
            indices = sim_matrix.indices()  # [2, nnz]
            values = sim_matrix.values()  # [nnz]

            # 获取每行的起始位置（假设已按行排序）
            row_counts = torch.bincount(indices[0], minlength=n_items)
            row_starts = torch.cumsum(row_counts, dim=0) - row_counts

            for item_id in range(n_items):
                if item_id % 50000 == 0:
                    print(f"[SwingEnhancement] Processed {item_id}/{n_items} items")

                start = row_starts[item_id].item()
                end = start + row_counts[item_id].item()

                if start == end:
                    # 该行没有非零元素（除了对角线可能为0）
                    topk_cache[item_id] = ([], [])
                    continue

                row_indices = indices[1][start:end]  # 列索引
                row_values = values[start:end]  # 相似度值

                # 排除自身（如果存在）
                mask = row_indices != item_id
                row_indices = row_indices[mask]
                row_values = row_values[mask]

                # 取top-k
                if len(row_values) > 0:
                    k_actual = min(k, len(row_values))
                    topk_vals, topk_idx = torch.topk(row_values, k=k_actual)
                    neighbor_indices = row_indices[topk_idx].cpu().tolist()
                    neighbor_weights = topk_vals.cpu().tolist()

                    # 归一化权重
                    weight_sum = sum(neighbor_weights)
                    if weight_sum > 0:
                        neighbor_weights = [w / weight_sum for w in neighbor_weights]
                else:
                    neighbor_indices, neighbor_weights = [], []

                topk_cache[item_id] = (neighbor_indices, neighbor_weights)
        else:
            # 稠密矩阵情况（内存消耗大，尽量避免）
            for item_id in range(n_items):
                if item_id % 1000 == 0:
                    print(f"[SwingEnhancement] Processed {item_id}/{n_items} items")

                row = sim_matrix[item_id]
                # 排除自身并取top-k
                mask = torch.arange(n_items, device=sim_matrix.device) != item_id
                row_masked = row[mask]
                indices_masked = torch.arange(n_items, device=sim_matrix.device)[mask]

                if len(row_masked) > 0:
                    k_actual = min(k, len(row_masked))
                    topk_vals, topk_idx = torch.topk(row_masked, k=k_actual)
                    neighbor_indices = indices_masked[topk_idx].cpu().tolist()
                    neighbor_weights = topk_vals.cpu().tolist()

                    # 归一化权重
                    weight_sum = sum(neighbor_weights)
                    if weight_sum > 0:
                        neighbor_weights = [w / weight_sum for w in neighbor_weights]
                else:
                    neighbor_indices, neighbor_weights = [], []

                topk_cache[item_id] = (neighbor_indices, neighbor_weights)

        self.topk_neighbors_cache = topk_cache
        self.topk_precomputed = True
        print(f"[SwingEnhancement] Top-{k} neighbors precomputation completed")

    def get_similarity_matrix(self, sparse=True):
        """获取相似度矩阵，如果未计算则自动计算

        Args:
            sparse: 是否返回稀疏矩阵，默认为True以节省内存

        Returns:
            相似度矩阵（稠密或稀疏格式）
        """
        if self.sim_matrix is None:
            self.init_swing_sim()
            try:
                # 计算相似度矩阵（会自动调用compute_similarity_matrix）
                if sparse:
                    # 使用稀疏相似度矩阵
                    self.sim_matrix = self.swing_sim.get_sparse_similarity_matrix()
                else:
                    # 使用稠密相似度矩阵（可能内存过大）
                    self.sim_matrix = self.swing_sim.get_similarity_matrix(normalized=True)
                print(f"[SwingEnhancement] Similarity matrix computed: {self.sim_matrix.shape}")
            except Exception as e:
                print(f"[WARNING] Failed to compute swing similarity matrix: {e}")
                # 返回一个稀疏单位矩阵作为后备（始终在CPU上）
                n_items = self.dataset.n_items
                device = 'cpu'  # 始终在CPU上，避免GPU内存问题
                if sparse:
                    # 创建稀疏单位矩阵
                    indices = torch.arange(n_items, device=device).repeat(2, 1)
                    values = torch.ones(n_items, device=device)
                    self.sim_matrix = torch.sparse_coo_tensor(indices, values, (n_items, n_items))
                else:
                    self.sim_matrix = torch.eye(n_items, device=device)
                print(f"[SwingEnhancement] Using {'sparse ' if sparse else ''}identity matrix as fallback: {self.sim_matrix.shape}")
        # 确保相似度矩阵在CPU上（避免GPU内存问题）
        if self.sim_matrix is not None and self.sim_matrix.device.type != 'cpu':
            self.sim_matrix = self.sim_matrix.cpu()
        return self.sim_matrix

    def set_similarity_matrix(self, sim_matrix):
        """设置相似度矩阵（用于测试）"""
        self.sim_matrix = sim_matrix

    def get_similarity_row(self, sim_matrix, item_id):
        """获取相似度矩阵的指定行（支持稀疏和稠密矩阵）"""
        if sim_matrix.is_sparse:
            # 稀疏矩阵：提取指定行的非零元素
            sim_matrix = sim_matrix.coalesce()
            indices = sim_matrix.indices()
            values = sim_matrix.values()

            # 找到该行的非零元素
            row_mask = indices[0] == item_id
            if not row_mask.any():
                # 该行没有非零元素（除了对角线可能为0）
                return torch.zeros(sim_matrix.size(1), device=sim_matrix.device)

            row_indices = indices[1][row_mask]
            row_values = values[row_mask]

            # 创建稠密行
            dense_row = torch.zeros(sim_matrix.size(1), device=sim_matrix.device)
            dense_row[row_indices] = row_values

            # 确保对角线元素（自身相似度）存在
            # 对于单位矩阵后备，对角线为1；对于Swing矩阵，对角线可能为0或未存储
            # 我们添加一个小的自相似度以确保自身可被排除
            current_val = dense_row[item_id]
            dense_row[item_id] = torch.max(current_val, torch.tensor(1e-6, device=dense_row.device))

            return dense_row
        else:
            # 稠密矩阵：直接返回行
            return sim_matrix[item_id]
    def get_batch_similarity_submatrix(self, sim_matrix, batch_items):
        """从相似度矩阵提取batch内物品的子矩阵

        Args:
            sim_matrix: 相似度矩阵（稀疏或稠密）
            batch_items: batch中的物品ID列表/张量 [bs]

        Returns:
            submatrix: batch内物品的相似度子矩阵 [bs, bs]
        """
        # 获取batch_items的原始设备（如果它是张量）
        if isinstance(batch_items, torch.Tensor):
            original_device = batch_items.device
            # 将batch_items转移到sim_matrix的设备（通常是CPU）进行计算
            batch_items_cpu = batch_items.cpu()
        else:
            original_device = sim_matrix.device
            batch_items_cpu = torch.tensor(batch_items, device='cpu')

        bs = len(batch_items_cpu)

        if sim_matrix.is_sparse:
            # 稀疏矩阵：逐行提取并构建子矩阵（在CPU上计算）
            submatrix = torch.zeros((bs, bs), device='cpu')

            # 获取每个物品的相似度行
            for i, item_i in enumerate(batch_items):
                item_i = item_i.item()
                row_i = self.get_similarity_row(sim_matrix, item_i)  # [n_items]

                # 提取batch内物品对应的列
                for j in range(bs):
                    item_j = batch_items_cpu[j].item()
                    submatrix[i, j] = row_i[item_j]
        else:
            # 稠密矩阵：直接索引
            submatrix = sim_matrix[batch_items_cpu][:, batch_items_cpu]

        # 将结果转移回原始设备
        if original_device != submatrix.device:
            submatrix = submatrix.to(original_device)

        return submatrix
    def set_embedding_layer(self, embedding_layer):
        """设置embedding层（GPT-2的wte层）"""
        self.embedding_layer = embedding_layer
        # 当embedding层设置后，清空缓存，以便重新计算
        self.all_item_embeddings = None

    def compute_all_item_embeddings(self):
        """计算所有物品的语义embedding并缓存（优化版本：结果存储在CPU上以节省GPU内存）"""
        if self.embedding_layer is None:
            raise RuntimeError("Embedding layer not set. Call set_embedding_layer() first.")

        n_items = self.dataset.n_items
        emb_dim = self.config['n_embd']
        compute_device = self.embedding_layer.weight.device  # 计算设备（可能是GPU）
        store_device = 'cpu'  # 存储设备（始终为CPU）

        # 根据计算设备调整批量大小
        if compute_device.type == 'cuda':
            batch_size = 100  # GPU上使用较小的批量
        else:
            batch_size = 500  # CPU上可以使用较大的批量

        # 结果存储在CPU上
        all_embs = torch.zeros((n_items, emb_dim), device=store_device)

        # 收集所有物品的token和对应的物品ID
        item_ids_list = []
        tokens_list = []

        # 首先收集所有数据
        for item_str, tokens in self.tokenizer.item2tokens.items():
            item_id = self.dataset.item2id.get(item_str)
            if item_id is None:
                continue
            item_ids_list.append(item_id)
            tokens_list.append(tokens)

        # 转换为张量并分批处理
        total_items = len(item_ids_list)
        print(
            f"[SwingEnhancement] Computing embeddings for {total_items} items in batches of {batch_size} (compute_device: {compute_device}, store_device: {store_device})")

        # 使用no_grad避免梯度计算和内存占用
        with torch.no_grad():
            for start_idx in range(0, total_items, batch_size):
                end_idx = min(start_idx + batch_size, total_items)
                batch_item_ids = item_ids_list[start_idx:end_idx]
                batch_tokens = tokens_list[start_idx:end_idx]
                # 创建批量token张量并移到计算设备
                batch_token_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=compute_device)

                # 通过embedding层获取embedding，然后取平均
                # embedding_layer期望形状 [batch_size, n_digit]，返回 [batch_size, n_digit, emb_dim]
                batch_embs = self.embedding_layer(batch_token_tensor).mean(dim=-2)  # [batch_size, emb_dim] (在计算设备上)

                # 将embedding移回CPU并存储
                batch_embs_cpu = batch_embs.to(store_device)
                for i, item_id in enumerate(batch_item_ids):
                    all_embs[item_id] = batch_embs_cpu[i]

                # 清理中间张量以释放内存
                del batch_token_tensor, batch_embs, batch_embs_cpu
                if compute_device.type == 'cuda':
                    torch.cuda.empty_cache()

                if start_idx % 5000 == 0  or end_idx == total_items:
                    print(f"[SwingEnhancement] Processed {end_idx}/{total_items} items")
        print(f"[SwingEnhancement] Completed computing embeddings for {total_items} items")
        self.all_item_embeddings = all_embs
        return all_embs

    def get_all_item_embeddings(self):
        """获取所有物品的语义embedding，如果未缓存则计算"""
        if self.all_item_embeddings is None:
            self.compute_all_item_embeddings()
        return self.all_item_embeddings

    def set_all_item_embeddings(self, embeddings):
        """设置所有物品的语义embedding（用于测试）"""
        self.all_item_embeddings = embeddings

    def forward(self, semantic_embs, item_ids):
        """增强语义embedding（优化版本：使用预计算的top-k邻居，避免设备间传输）"""
        if not self.config.get('use_swing_enhancement', False):
            return semantic_embs

        # 获取目标设备（输入embedding所在的设备）
        target_device = semantic_embs.device

        # 预计算top-k邻居（如果尚未计算）
        try:
            self.precompute_topk_neighbors()
        except Exception as e:
            print(f"[WARNING] Failed to precompute top-k neighbors, skipping enhancement: {e}")
            return semantic_embs
        # 获取所有物品的embedding（保持在CPU上以节省GPU内存）
        try:
            all_item_embs = self.get_all_item_embeddings()  # [n_items, emb_dim] (CPU)
        except Exception as e:
            print(f"[WARNING] Cannot compute all item embeddings, skipping enhancement: {e}")
            return semantic_embs

        # 确保增强权重参数在正确设备上（如果使用门控增强）
        if self.enhance_type == 'gated' and self.enhance_gate.weight.device != target_device:
            self.enhance_gate = self.enhance_gate.to(target_device)

        batch_size, seq_len, emb_dim = semantic_embs.shape
        enhanced_embs = torch.zeros_like(semantic_embs)

        # 将物品ID展平，以便批处理
        flat_item_ids = item_ids.view(-1)  # [batch_size * seq_len]
        flat_semantic_embs = semantic_embs.view(-1, emb_dim)  # [batch_size * seq_len, emb_dim]
        flat_enhanced = torch.zeros_like(flat_semantic_embs)

        # 缓存已处理的物品ID的结果，避免重复计算
        cache = {}

        for idx in range(len(flat_item_ids)):
            item_id = flat_item_ids[idx].item()
            if item_id == 0:  # padding
                flat_enhanced[idx] = flat_semantic_embs[idx]
                continue

            ## 检查缓存
            if item_id in cache:
                flat_enhanced[idx] = cache[item_id]
                continue

            # 从缓存中获取邻居信息
            if item_id >= len(self.topk_neighbors_cache):
                # 物品ID超出范围（可能是新物品）
                flat_enhanced[idx] = flat_semantic_embs[idx]
                cache[item_id] = flat_semantic_embs[idx]
                continue

            neighbor_indices_list, neighbor_weights_list = self.topk_neighbors_cache[item_id]

            if not neighbor_indices_list:
                # 没有邻居，返回原始embedding
                flat_enhanced[idx] = flat_semantic_embs[idx]
                cache[item_id] = flat_semantic_embs[idx]
                continue

            # 将邻居索引和权重转换为张量（CPU）
            neighbor_indices = torch.tensor(neighbor_indices_list, dtype=torch.long, device='cpu')
            neighbor_weights = torch.tensor(neighbor_weights_list, dtype=torch.float32, device='cpu')

            # 获取邻居embedding（CPU）
            neighbor_embs = all_item_embs[neighbor_indices]  # [neighbors, emb_dim] (CPU)

            # 计算加权平均（CPU）
            weighted_neighbor_emb = torch.sum(neighbor_weights.unsqueeze(1) * neighbor_embs, dim=0)  # [emb_dim] (CPU)

            # 将加权邻居embedding转移到目标设备
            weighted_neighbor_emb = weighted_neighbor_emb.to(target_device)

            # 应用增强
            if self.enhance_type == 'additive':
                enhanced = flat_semantic_embs[idx] + self.swing_enhance_weight * weighted_neighbor_emb
            elif self.enhance_type == 'multiplicative':
                enhanced = flat_semantic_embs[idx] * (1 + self.swing_enhance_weight * weighted_neighbor_emb)
            elif self.enhance_type == 'gated':
                # 门控融合
                combined = torch.cat([flat_semantic_embs[idx], weighted_neighbor_emb], dim=-1)
                gate = torch.sigmoid(self.enhance_gate(combined))
                enhanced = gate * flat_semantic_embs[idx] + (1 - gate) * weighted_neighbor_emb
            else:
                enhanced = flat_semantic_embs[idx]

            flat_enhanced[idx] = enhanced
            cache[item_id] = enhanced

        # 恢复原始形状
        enhanced_embs = flat_enhanced.view(batch_size, seq_len, emb_dim)
        return enhanced_embs

class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        # self.item_id_embedding=nn.Embedding(
        #     num_embeddings=self.dataset.n_items+1,
        #     embedding_dim=config['n_embd'],
        #     padding_idx=0
        # )
        # self.fusion_gate = nn.Linear(config['n_embd'] * 2, config['n_embd'])
        # self.id_scale = nn.Parameter(torch.tensor(0.1))

        # Swing算法相关配置
        self.use_swing = config.get('use_swing', False)
        self.swing_weight = config.get('swing_weight', 0.5)
        self.swing_alpha = config.get('swing_alpha', 1.0)
        self.swing_min_cooccurrence = config.get('swing_min_cooccurrence', 2)

        # Swing语义ID增强配置
        self.use_swing_enhancement = config.get('use_swing_enhancement', False)
        self.swing_enhance_weight = config.get('swing_enhance_weight', 0.3)
        self.swing_neighbors = config.get('swing_neighbors', 10)
        self.swing_enhance_type = config.get('swing_enhance_type', 'additive')

        if self.use_swing:
            from .swing import SwingSimilarity
            self.swing_sim = SwingSimilarity(
                dataset=self.dataset,
                alpha=self.swing_alpha,
                min_cooccurrence=self.swing_min_cooccurrence,
                device='cpu'  # 相似度矩阵在CPU上计算，避免GPU内存问题
            )

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

        # Swing增强组件（在gpt2初始化后，以便传递embedding层）
        if self.use_swing_enhancement:
            self.swing_enhancement = SwingEnhancement(config, dataset, tokenizer, embedding_layer=self.gpt2.wte)
            # 预计算相似度矩阵和所有物品embedding（使用no_grad避免梯度计算和内存占用）
            try:
                with torch.no_grad():
                    print(f"[RPG] Precomputing swing similarity matrix for {dataset.n_items} items...")
                    sim_matrix = self.swing_enhancement.get_similarity_matrix(sparse=True)
                    print(f"[RPG] Swing similarity matrix computed: {sim_matrix.shape}")

                    print(f"[RPG] Precomputing all item embeddings...")
                    all_item_embs = self.swing_enhancement.get_all_item_embeddings()
                    print(f"[RPG] All item embeddings computed: {all_item_embs.shape}")
            except Exception as e:
                print(f"[WARNING] Failed to precompute swing data: {e}")
                print(f"[WARNING] Swing enhancement will be disabled or use fallback")
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            # pred_head_list.append(ResBlock(self.config['n_embd']))
            if i<128:
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

        # Swing增强语义embedding（如果启用）
        if self.use_swing_enhancement:
            semantic_embs = self.swing_enhancement(semantic_embs, batch['input_ids'])

        attention_mask = batch['attention_mask']
        semantic_embs = semantic_embs * attention_mask.unsqueeze(-1)
        input_embs = semantic_embs  # 直接使用增强后的语义embedding

        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        # final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        combine_states = outputs.last_hidden_state  # 移除+self.id_scale*id_embs
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
            # 使用语义embedding相似度替代gate和id_embedding相似度
            # 获取最后一个位置的语义embedding
            last_semantic_embs = semantic_embs[torch.arange(bs), last_positions]
            semantic_similarity = torch.matmul(
                F.normalize(last_semantic_embs, dim=-1),
                F.normalize(last_semantic_embs, dim=-1).T
            )

            # 如果启用Swing增强，可以结合Swing相似度
            if self.use_swing_enhancement:
                try:
                    # 获取Swing相似度矩阵
                    swing_sim_matrix = self.swing_enhancement.get_similarity_matrix()
                    # 提取batch内物品间的Swing相似度
                    batch_items = batch['input_ids'][torch.arange(bs), last_positions]
                    # 使用新方法提取子矩阵（支持稀疏张量）
                    swing_sim = self.swing_enhancement.get_batch_similarity_submatrix(swing_sim_matrix, batch_items)
                    combined_sim = 0.5 * semantic_similarity + 0.5 * swing_sim
                except Exception as e:
                    print(f"[WARNING] Failed to get swing similarity submatrix, using semantic similarity only: {e}")
                    combined_sim = semantic_similarity
            else:
                combined_sim = semantic_similarity

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

    def _build_base_sim_mat(self, threshold=0.5, use_half=False):
        """计算基础相似度矩阵（基于item_id_embedding）

        Args:
            threshold: 相似度阈值，低于此值的相似度设为0
            use_half: 是否使用半精度计算

        Returns:
            稀疏相似度矩阵 (torch.sparse_coo_tensor)
        """
        n_items = self.dataset.n_items
        device = self.gpt2.device if hasattr(self.gpt2, "device") else next(self.parameters()).device

        # 获取item embedding并归一化
        item_embs = self.item_id_embedding.weight[1:]  # 跳过padding_idx=0
        item_embs = F.normalize(item_embs, dim=-1)

        if use_half:
            item_embs = item_embs.to(torch.float16)

        # 分块计算余弦相似度
        chunk_size = self.chunk_size
        indices = []
        values = []

        for i_start in range(1, n_items, chunk_size):
            i_end = min(i_start + chunk_size, n_items)
            bi = i_end - i_start

            # 获取当前块的embedding
            embs_i = item_embs[i_start - 1:i_end - 1]  # 注意：item_embs索引从0开始对应item_id=1

            for j_start in range(i_start, n_items, chunk_size):
                j_end = min(j_start + chunk_size, n_items)
                bj = j_end - j_start

                # 获取j块的embedding
                embs_j = item_embs[j_start - 1:j_end - 1]

                # 计算余弦相似度 (bi, bj)
                sim_block = torch.matmul(embs_i, embs_j.T)

                # 应用阈值并收集上三角部分 (j >= i)
                j_offsets = torch.arange(bj, device=device)
                global_j = j_start + j_offsets

                for local_i in range(bi):
                    global_i = i_start + local_i
                    row_vals = sim_block[local_i]
                    row_mask = (row_vals > threshold) & (global_j >= global_i)

                    if row_mask.any():
                        js = global_j[row_mask]
                        vs = row_vals[row_mask]
                        # 收集到CPU列表
                        indices.extend([[global_i, int(j.item())] for j in js])
                        values.extend([float(v.item()) for v in vs])

        # 构建稀疏矩阵
        if indices:
            idx_t = torch.tensor(indices, dtype=torch.int64).t()
            val_t = torch.tensor(values, dtype=torch.float32)
            return torch.sparse_coo_tensor(idx_t, val_t, (n_items, n_items)).coalesce()
        else:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64),
                torch.empty((0,), dtype=torch.float32),
                size=(n_items, n_items)
            ).coalesce()

    def build_ii_sim_mat(self, threshold=0.5, use_half=False, sim_type='semantic'):
        """构建物品-物品相似度矩阵

        Args:
            threshold: 相似度阈值，低于此值的相似度设为0
            use_half: 是否使用半精度计算
            sim_type: 相似度类型，可选 'semantic'（语义相似度）、'embedding'（nn.Embedding相似度）、
                     'fusion'（融合语义和swing相似度）

        Returns:
            稀疏相似度矩阵 (torch.sparse_coo_tensor)
        """
        n_items = self.dataset.n_items
        device = self.gpt2.device if hasattr(self.gpt2, "device") else next(self.parameters()).device

        if sim_type == 'embedding':
            # 使用nn.Embedding计算相似度
            return self._build_base_sim_mat(threshold, use_half)

        elif sim_type == 'semantic':
            # 原有的语义相似度计算（基于token embedding）
            # 获取模型所在的设备
            n_digit = self.tokenizer.n_digit
            codebook_size = self.tokenizer.codebook_size
            chunk_size = self.chunk_size

            # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
            token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

            # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
            token_embs = F.normalize(token_embs, dim=-1)
            if use_half:
                token_embs = token_embs.to(torch.float16)
            token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

            # 3) Convert [-1, 1] to [0, 1] range
            token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

            indices = []
            values = []
            item_tokens = self.item_id2tokens

            # 4) Fill the item-item matrix in chunks
            for i_start in range(1, n_items, chunk_size):
                i_end = min(i_start + chunk_size, n_items)
                tokens_i = item_tokens[i_start:i_end].to(device)
                bi = i_end - i_start

                for j_start in range(1, n_items, chunk_size):
                    j_end = min(j_start + chunk_size, n_items)
                    tokens_j = item_tokens[j_start:j_end].to(device)
                    bj = j_end - j_start

                    dtype_block = torch.float16 if use_half else torch.float32
                    sum_block = torch.zeros((bi, bj), device=device, dtype=dtype_block)

                    for k in range(n_digit):
                        row_inds = tokens_i[:, k] - k * codebook_size - 1
                        col_inds = tokens_j[:, k] - k * codebook_size - 1
                        temp = token_sims_01[k].index_select(0, row_inds)
                        temp = temp.index_select(1, col_inds)
                        sum_block += temp

                    avg_block = sum_block / n_digit.to(torch.float32)
                    j_offsets = torch.arange(bj, device=device)
                    global_j = j_start + j_offsets

                    for local_i in range(bi):
                        global_i = i_start + local_i
                        mask = (avg_block[local_i] > threshold) & (global_j >= global_i)
                        if mask.any():
                            js = global_j[mask]
                            vs = avg_block[local_i, mask]
                            indices.extend([[global_i, int(j.item())] for j in js])
                            values.extend([float(v.item()) for v in vs])

                    del tokens_j, avg_block, sum_block
                del tokens_i

            if indices:
                idx_t = torch.tensor(indices, dtype=torch.int64).t()
                val_t = torch.tensor(values, dtype=torch.float32)
                return torch.sparse_coo_tensor(idx_t, val_t, (n_items, n_items)).coalesce()
            else:
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.int64),
                    torch.empty((0,), dtype=torch.float32),
                    size=(n_items, n_items)
                ).coalesce()

        elif sim_type == 'fusion':
            # 融合语义相似度和swing相似度
            if not self.use_swing:
                raise ValueError("Swing算法未启用，无法进行融合。请设置use_swing=True")

            # 计算语义相似度
            semantic_sim = self.build_ii_sim_mat(threshold, use_half, sim_type='semantic')

            # 计算swing相似度
            swing_sim = self.swing_sim.get_sparse_similarity_matrix(threshold=threshold)

            # 确保swing相似度在正确的设备上
            swing_sim = swing_sim.to(device)

            # 如果语义相似度是稀疏的，转换为稠密以便融合
            if semantic_sim.is_sparse:
                semantic_sim_dense = semantic_sim.to_dense()
            else:
                semantic_sim_dense = semantic_sim

            if swing_sim.is_sparse:
                swing_sim_dense = swing_sim.to_dense()
            else:
                swing_sim_dense = swing_sim

            # 确保两个矩阵形状相同
            assert semantic_sim_dense.shape == swing_sim_dense.shape, \
                f"形状不匹配: semantic {semantic_sim_dense.shape}, swing {swing_sim_dense.shape}"

            # 加权融合
            fused_sim = (1 - self.swing_weight) * semantic_sim_dense + \
                       self.swing_weight * swing_sim_dense

            # 应用阈值并转换为稀疏矩阵
            mask = fused_sim > threshold
            rows, cols = torch.where(mask)
            # 过滤上三角 (j >= i)
            upper_triangle_mask = cols >= rows
            rows = rows[upper_triangle_mask]
            cols = cols[upper_triangle_mask]

            if rows.numel() > 0:
                indices = torch.stack([rows, cols], dim=0)
                values = fused_sim[rows, cols]
                return torch.sparse_coo_tensor(indices, values, (n_items, n_items)).coalesce()
            else:
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.int64, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    size=(n_items, n_items)
                ).coalesce()

        else:
            raise ValueError(f"未知的相似度类型: {sim_type}，可选 'semantic', 'embedding', 'fusion'")

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
        # Check if adjacency matrix is already initialized
        if hasattr(self, 'adjacency') and self.adjacency is not None:
            self.tokenizer.log("Graph already initialized, skipping...")
            return

        # Try to load cached adjacency matrix
        cache_dir = self.config.get('cache_dir', self.config.get('log_dir', './cache'))
        os.makedirs(cache_dir, exist_ok=True)

        # Generate cache filename based on dataset and model parameters
        dataset_name = self.config.get('dataset', 'unknown')
        n_items = self.dataset.n_items
        sim_type = self.config.get('sim_type', 'semantic')
        use_swing = self.config.get('use_swing', False)
        swing_weight = self.config.get('swing_weight', 0.5)

        if use_swing:
            cache_suffix = f"{sim_type}_swing{swing_weight:.2f}"
        else:
            cache_suffix = sim_type

        cache_file = os.path.join(cache_dir, f"adjacency_{dataset_name}_items{n_items}_{cache_suffix}.pt")

        if os.path.exists(cache_file):
            self.tokenizer.log(f"Loading cached adjacency matrix from {cache_file}")
            try:
                self.adjacency = torch.load(cache_file, map_location=self.config['device'])
                self.tokenizer.log("Graph initialized from cache.")
                return
            except Exception as e:
                self.tokenizer.log(f"Failed to load cached adjacency matrix: {e}, rebuilding...")

        # Build adjacency matrix
        self.tokenizer.log("Building item-item similarity matrix... (this may take a while)")

        # 根据配置选择相似度计算方式
        sim_type = self.config.get('sim_type', 'semantic')
        use_swing = self.config.get('use_swing', False)

        if use_swing or sim_type in ['fusion', 'embedding']:
            # 使用build_ii_sim_mat构建相似度矩阵（支持swing融合）
            sim_threshold = 0.5
            use_half = False
            self.tokenizer.log(f"Using similarity type: {sim_type}, swing enabled: {use_swing}")

            # 构建相似度矩阵
            item_item_sim = self.build_ii_sim_mat(
                threshold=sim_threshold,
                use_half=use_half,
                sim_type=sim_type
            )

            # 从相似度矩阵构建邻接列表
            self.adjacency = self.build_adjacency_list(item_item_sim)
        else:
            # 使用原有的优化方法（语义相似度）
            adj_idx, _ = self.build_ii_topk_adjacency(
                use_threshold=False,
                threshold=0.5,
                use_half=False
            )
            self.adjacency = adj_idx

        # Save to cache for future use
        try:
            torch.save(self.adjacency, cache_file)
            self.tokenizer.log(f"Saved adjacency matrix to cache: {cache_file}")
        except Exception as e:
            self.tokenizer.log(f"Failed to save adjacency matrix to cache: {e}")
        self.tokenizer.log("Graph initialized.")
    def graph_propagation(self, token_logits, n_return_sequences, return_scores=False):
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
            batch_scores = []
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
                batch_scores.append(scores[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)
            if return_scores:
                # Keep track of scores for the final selection
                topk_scores = torch.stack(batch_scores, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        # Get final predictions
        preds = topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1)

        if return_scores:
            # Get scores for the final predictions
            final_scores = topk_scores[:, :n_return_sequences].unsqueeze(-1)
            return preds, final_scores, visited_counts
        else:
            return preds, visited_counts

    def generate(self, batch, n_return_sequences=1, return_scores=False):
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
            if return_scores:
                preds, scores, visited_counts = self.graph_propagation(
                    token_logits=token_logits,
                    n_return_sequences=n_return_sequences,
                    return_scores=True
                )
                return preds, scores, visited_counts
            else:
                preds, visited_counts = self.graph_propagation(
                    token_logits=token_logits,
                    n_return_sequences=n_return_sequences,
                    return_scores=False
                )
                return preds, visited_counts
        else:
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),              # (batch_size, n_items, n_tokens)
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, code_dim)
            ).mean(dim=-1)
            if return_scores:
                scores, indices = item_logits.topk(n_return_sequences, dim=-1)
                preds = indices + 1
                return preds.unsqueeze(-1), scores.unsqueeze(-1)
            else:
                preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
                return preds.unsqueeze(-1)
