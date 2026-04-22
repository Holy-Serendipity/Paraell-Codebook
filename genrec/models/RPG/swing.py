import torch
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import os
import time
import functools
from logging import getLogger


class SwingSimilarity:
    """计算物品间的swing相似度

    Swing算法是一种基于物品的协同过滤算法，考虑共同点击的用户对物品相似度的贡献：
    sim(i,j) = Σ_{u∈U_i∩U_j} Σ_{v∈U_i∩U_j} 1/(α + |I_u ∩ I_v|)

    其中：
    - U_i: 点击物品i的用户集合
    - I_u: 用户u点击的物品集合
    - α: 平滑参数（通常设为1）
    """

    def __init__(self, dataset, alpha=1.0, min_cooccurrence=2, device='cpu'):
        """
        初始化Swing相似度计算器

        Args:
            dataset: AbstractDataset实例，包含all_item_seqs
            alpha: 平滑参数，防止分母为0
            min_cooccurrence: 最小共同出现次数阈值
            device: 计算设备（'cpu'或'cuda'）
        """
        self.dataset = dataset
        self.alpha = alpha
        self.min_cooccurrence = min_cooccurrence
        self.device = device
        self.n_items = dataset.n_items
        self.n_users = dataset.n_users

        # 从dataset获取用户-物品交互数据
        self.all_item_seqs = dataset.all_item_seqs

        # 缓存相似度矩阵
        self.sim_matrix = None
        self.sparse_sim_matrix = None

        self.logger = getLogger()

    def _build_user_item_matrix(self):
        """构建用户-物品交互矩阵（稀疏格式）"""
        self.logger.info(f"[SWING] Building user-item matrix for {self.n_users} users, {self.n_items} items")

        # 收集所有交互
        rows, cols = [], []
        for user_id, item_seq in self.all_item_seqs.items():
            # user_id可能是字符串，需要转换为整数索引
            user_idx = self.dataset.user2id.get(user_id, None)
            if user_idx is None:
                continue

            for item_id in item_seq:
                item_idx = self.dataset.item2id.get(item_id, None)
                if item_idx is None:
                    continue
                rows.append(user_idx)
                cols.append(item_idx)

        # 创建稀疏矩阵（CSR格式）
        data = np.ones(len(rows), dtype=np.float32)
        user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items)
        )

        self.logger.info(f"[SWING] User-item matrix shape: {user_item_matrix.shape}, nnz: {user_item_matrix.nnz}")
        return user_item_matrix

    def _compute_user_cosine(self, user_item_matrix):
        """计算用户共同点击矩阵（稀疏格式）

        计算用户间的共同点击物品数量 |I_u ∩ I_v|。
        返回稀疏矩阵以节省内存。
        """
        self.logger.info("[SWING] Computing user-user co-occurrence matrix（sparse)")

        # 计算用户-用户共同点击物品数量
        # (U x I) @ (I x U) = (U x U) 中的每个元素表示共同点击物品数
        user_cooccurrence = user_item_matrix @ user_item_matrix.T

        # 转换为稠密矩阵（对于大规模数据可能内存过大，后续需要优化）
        user_cooccurrence = user_cooccurrence.toarray()

        self.logger.info(f"[SWING] User co-occurrence matrix shape: {user_cooccurrence.shape}, nnz: {user_cooccurrence.nnz}")
        return user_cooccurrence

    def compute_similarity_matrix(self, threshold=0.0, use_sparse=True, sim_type='jaccard'):
        """计算物品相似度矩阵

        Args:
            threshold: 相似度阈值，低于此值的相似度设为0
            use_sparse: 是否返回稀疏矩阵
            sim_type: 相似度类型，'exact'（精确Swing）或'jaccard'（Jaccard近似，默认）
        Returns:
            相似度矩阵（稠密或稀疏格式）
        """
        self.logger.info("[SWING] Starting swing similarity computation")
        start_time = time.time()

        # 1. 构建用户-物品矩阵
        user_item_matrix = self._build_user_item_matrix()

        # 2. 计算物品相似度（跳过用户共同点击矩阵以节省内存）
        self.logger.info(f"[SWING] Computing item-item similarity matrix (type: {sim_type})")

        # 根据相似度类型选择算法
        if sim_type == 'jaccard':
            # 使用Jaccard相似度（计算更快，内存更友好）
            result = self._compute_jaccard_similarity(
                user_item_matrix, threshold, use_sparse
            )
        else:
            # 使用精确Swing算法（计算较慢）
            result = self._compute_exact_similarity(
                user_item_matrix, threshold, use_sparse
            )
        # 记录总计算时间
        elapsed = time.time() - start_time
        self.logger.info(f"[SWING] Total similarity computation completed in {elapsed:.2f}s")

        return result


    def get_similarity_matrix(self, normalized=True):
        """获取相似度矩阵

        Args:
            normalized: 是否归一化到[0,1]范围

        Returns:
            相似度矩阵（torch.Tensor）
        """
        if self.sparse_sim_matrix is not None:
            sim_matrix = self.sparse_sim_matrix
            # 转换为稠密矩阵以便归一化（对于大规模数据需要优化）
            sim_matrix = sim_matrix.toarray()
        elif self.sim_matrix is not None:
            sim_matrix = self.sim_matrix
        else:
            raise RuntimeError("Similarity matrix not computed. Call compute_similarity_matrix() first.")

        # 归一化到[0,1]范围
        if normalized and sim_matrix.max() > 0:
            sim_matrix = sim_matrix / sim_matrix.max()

        # 转换为torch.Tensor
        sim_tensor = torch.FloatTensor(sim_matrix).to(self.device)

        return sim_tensor

    def get_sparse_similarity_matrix(self, threshold=0.0, normalized=True):
        """获取稀疏相似度矩阵

        Args:
            threshold: 相似度阈值，低于此值的边被移除
            normalized: 是否归一化到[0,1]范围
        Returns:
            稀疏相似度矩阵（torch.sparse.Tensor）
        """
        if self.sparse_sim_matrix is None:
            self.compute_similarity_matrix(threshold=threshold, use_sparse=True)

        # 确保是COO格式以便访问.row和.col属性
        if not hasattr(self.sparse_sim_matrix, 'row') or not hasattr(self.sparse_sim_matrix, 'col'):
            # 如果不是COO格式，转换为COO
            sim_matrix = self.sparse_sim_matrix.tocoo()
            # 更新缓存以便后续使用
            self.sparse_sim_matrix = sim_matrix
        else:
            sim_matrix = self.sparse_sim_matrix

        # 应用阈值
        if threshold > 0:
            sim_matrix.data[sim_matrix.data < threshold] = 0
            sim_matrix.eliminate_zeros()

        # 转换为torch稀疏张量
        indices = torch.LongTensor(np.vstack([sim_matrix.row, sim_matrix.col]))
        values = torch.FloatTensor(sim_matrix.data)
        shape = torch.Size(sim_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        # 归一化
        if normalized and sparse_tensor._values().numel() > 0:
            max_val = sparse_tensor._values().max().item()
            if max_val > 0:
                sparse_tensor = sparse_tensor / max_val

        return sparse_tensor.to(self.device)

    def save_similarity_matrix(self, filepath):
        """保存相似度矩阵到文件"""
        if self.sparse_sim_matrix is not None:
            sp.save_npz(filepath, self.sparse_sim_matrix)
            self.logger.info(f"[SWING] Saved sparse similarity matrix to {filepath}")
        elif self.sim_matrix is not None:
            np.save(filepath, self.sim_matrix)
            self.logger.info(f"[SWING] Saved dense similarity matrix to {filepath}")
        else:
            raise RuntimeError("No similarity matrix to save")

    def load_similarity_matrix(self, filepath):
        """从文件加载相似度矩阵"""
        if filepath.endswith('.npz'):
            self.sparse_sim_matrix = sp.load_npz(filepath)
            self.logger.info(f"[SWING] Loaded sparse similarity matrix from {filepath}")
        elif filepath.endswith('.npy'):
            self.sim_matrix = np.load(filepath)
            self.logger.info(f"[SWING] Loaded dense similarity matrix from {filepath}")
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def _compute_jaccard_similarity(self, user_item_matrix, threshold=0.0, use_sparse=True):
        """计算物品间的Jaccard相似度（内存友好且高效）"""
        self.logger.info("[SWING] Computing Jaccard similarity matrix")
        start_time = time.time()

        # 计算物品-物品共同点击矩阵（交集大小）
        # (I x U) @ (U x I) = (I x I) 中的每个元素表示共同点击用户数
        cooccurrence = user_item_matrix.T @ user_item_matrix  # 稀疏矩阵

        # 转换为COO格式以便高效访问
        cooccurrence = cooccurrence.tocoo()

        # 获取每个物品的点击用户数
        item_degrees = np.array(user_item_matrix.sum(axis=0)).flatten()  # [n_items]

        # 初始化相似度矩阵
        if use_sparse:
            sim_rows, sim_cols, sim_data = [], [], []
        else:
            sim_matrix = np.zeros((self.n_items, self.n_items), dtype=np.float32)

        # 遍历所有非零共同点击对
        for i, j, intersect in zip(cooccurrence.row, cooccurrence.col, cooccurrence.data):
            if i > j:  # 只处理上三角，对称矩阵
                continue

            if intersect < self.min_cooccurrence:
                continue

            # 计算Jaccard相似度
            union = item_degrees[i] + item_degrees[j] - intersect
            if union > 0:
                similarity = intersect / union
            else:
                similarity = 0.0

            if similarity > threshold:
                if use_sparse:
                    sim_rows.append(i)
                    sim_cols.append(j)
                    sim_data.append(similarity)

                    if i != j:
                        sim_rows.append(j)
                        sim_cols.append(i)
                        sim_data.append(similarity)
                else:
                    sim_matrix[i, j] = similarity
                    sim_matrix[j, i] = similarity

        if use_sparse:
            # 创建稀疏矩阵（COO格式，便于转换为torch稀疏张量）
            sim_matrix = sp.coo_matrix(
                (sim_data, (sim_rows, sim_cols)),
                shape=(self.n_items, self.n_items)
            )
            self.sparse_sim_matrix = sim_matrix
        else:
            self.sim_matrix = sim_matrix

        elapsed = time.time() - start_time
        self.logger.info(f"[SWING] Jaccard similarity computation completed in {elapsed:.2f}s")

        if use_sparse:
            return self.sparse_sim_matrix
        else:
            return self.sim_matrix

    def _compute_exact_similarity(self, user_item_matrix, threshold=0.0, use_sparse=True):
        """计算精确的Swing相似度（计算密集，内存消耗大）"""
        self.logger.info("[SWING] Computing exact Swing similarity matrix")
        start_time = time.time()

        # 获取每件物品的用户集合和每个用户的物品集合
        item_users = defaultdict(set)
        user_items = defaultdict(set)
        for user_idx in range(self.n_users):
            # 获取该用户点击的物品
            items = user_item_matrix[user_idx].indices
            user_items[user_idx] = set(items)
            for item_idx in items:
                item_users[item_idx].add(user_idx)

        # 初始化相似度矩阵
        if use_sparse:
            sim_rows, sim_cols, sim_data = [], [], []
        else:
            sim_matrix = np.zeros((self.n_items, self.n_items), dtype=np.float32)

        # 计算每对物品的相似度
        items_list = list(item_users.keys())
        n_items = len(items_list)

        self.logger.info(f"[SWING] Computing similarity for {n_items} items with user interactions")

        # 使用缓存加速用户对共同点击物品数计算
        @functools.lru_cache(maxsize=100000)
        def get_cooccurrence(u, v):
            return len(user_items[u] & user_items[v])

        for i_idx in range(n_items):
            item_i = items_list[i_idx]
            users_i = item_users[item_i]

            if i_idx % 100 == 0:
                self.logger.info(f"[SWING] Processing item {i_idx}/{n_items}")

            for j_idx in range(i_idx, n_items):
                item_j = items_list[j_idx]
                users_j = item_users[item_j]

                # 共同点击用户
                common_users = users_i.intersection(users_j)
                if len(common_users) < self.min_cooccurrence:
                    continue

                # 计算swing相似度
                similarity = 0.0
                for u in common_users:
                    for v in common_users:
                        if u == v:
                            continue
                        # 计算用户u和v的共同点击物品数
                        cooccur = get_cooccurrence(u, v)
                        if cooccur > 0:
                            similarity += 1.0 / (self.alpha + cooccur)

                if similarity > threshold:
                    if use_sparse:
                        sim_rows.append(item_i)
                        sim_cols.append(item_j)
                        sim_data.append(similarity)

                        if item_i != item_j:
                            sim_rows.append(item_j)
                            sim_cols.append(item_i)
                            sim_data.append(similarity)
                    else:
                        sim_matrix[item_i, item_j] = similarity
                        sim_matrix[item_j, item_i] = similarity

        if use_sparse:
            # 创建稀疏矩阵（COO格式，便于转换为torch稀疏张量）
            sim_matrix = sp.coo_matrix(
                (sim_data, (sim_rows, sim_cols)),
                shape=(self.n_items, self.n_items)
            )
            self.sparse_sim_matrix = sim_matrix
        else:
            self.sim_matrix = sim_matrix

        elapsed = time.time() - start_time
        self.logger.info(f"[SWING] Exact Swing similarity computation completed in {elapsed:.2f}s")

        if use_sparse:
            return self.sparse_sim_matrix
        else:
            return self.sim_matrix

    def clear_cache(self):
        """清空缓存"""
        self.sim_matrix = None
        self.sparse_sim_matrix = None