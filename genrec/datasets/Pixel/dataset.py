import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Optional, Dict, List
from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class Pixel(AbstractDataset):
    def __init__(self, config: dict):
        super().__init__(config)

        self.log(f'[DATASET] {config.get("name", "NewDataset")}')

        # 配置参数
        self.min_interactions = config.get('min_interactions', 3)
        self.max_interactions = config.get('max_interactions', None)
        self.sort_method = config.get('sort_method', 'time')

        # 缓存目录
        self.cache_dir = config['cache_dir']

        self._process_raw_data()

    def _load_interactions(self, filepath: str) -> Dict[str, List[tuple]]:
        """加载交互数据，按用户分组"""
        self.log(f'[DATASET] Loading interactions from {filepath}')

        # 使用pandas分块读取大文件
        chunksize = 50000
        item_seqs = defaultdict(list)

        try:
            # 获取总行数用于进度条
            with open(filepath, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # 减去表头
        except:
            total_rows = 0

        chunk_iterator = pd.read_csv(
            filepath,
            chunksize=chunksize,
            iterator=True,
            encoding='utf-8'
        )

        with tqdm(total=total_rows, desc="Reading interactions", unit='row') as pbar:
            for chunk_df in chunk_iterator:
                for _, row in chunk_df.iterrows():
                    user_id = str(row['user_id'])
                    item_id = str(row['item_id'])
                    timestamp = int(row['timestamp'])

                    item_seqs[user_id].append((item_id, timestamp))
                pbar.update(len(chunk_df))

        # 按时间排序
        self._sort_interactions(item_seqs)

        return item_seqs

    def _sort_interactions(self, item_seqs: Dict[str, List[tuple]]):
        """对交互记录进行排序"""
        if self.sort_method == 'time':
            self.log('[DATASET] Ascending time sorting')
            for user_id in tqdm(item_seqs, desc="Sorting"):
                item_seqs[user_id].sort(key=lambda x: x[1])
        elif self.sort_method == 'time_desc':
            self.log('[DATASET] Descending time sorting')
            for user_id in tqdm(item_seqs, desc="Sorting"):
                item_seqs[user_id].sort(key=lambda x: x[1], reverse=True)

    def _filter_interactions(self, item_seqs: Dict[str, List[tuple]]) -> Dict[str, List[tuple]]:
        """过滤交互数据"""
        self.log('[DATASET] Filtering interactions...')

        filtered_seqs = defaultdict(list)
        total_users = len(item_seqs)

        with tqdm(total=total_users, desc="Filtering", unit='user') as pbar:
            for user_id, interactions in item_seqs.items():
                # 1. 最小交互次数过滤
                if len(interactions) < self.min_interactions:
                    continue

                # 2. 最大交互次数限制
                if self.max_interactions and len(interactions) > self.max_interactions:
                    interactions = interactions[:self.max_interactions]

                # 3. 可选：时间跨度过滤（如果需要）
                # if self.min_time_range:
                #     timestamps = [ts for _, ts in interactions]
                #     time_range = max(timestamps) - min(timestamps)
                #     if time_range < self.min_time_range:
                #         continue

                filtered_seqs[user_id] = interactions
                pbar.update(1)

        self.log(f'[DATASET] Filtered {len(filtered_seqs)} users '
                 f'(from {total_users})')

        return filtered_seqs

    def _remap_ids(self, item_seqs: Dict[str, List[tuple]]) -> tuple[dict, dict]:
        """重映射用户和物品ID"""
        self.log('[DATASET] Remapping user and item IDs...')

        for user_id, interactions in item_seqs.items():
            # 映射用户ID
            if user_id not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user_id] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user_id)

            # 映射物品ID并存储序列
            mapped_items = []
            for item_id, timestamp in interactions:
                if item_id not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item_id] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item_id)
                mapped_items.append(item_id)

            self.all_item_seqs[user_id] = mapped_items

        return self.all_item_seqs, self.id_mapping

    def _process_interactions(self, input_path: str, output_path: str) -> tuple[dict, dict]:
        """处理交互数据的主要流程"""
        seq_file = os.path.join(output_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')

        # 检查缓存
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[DATASET] Loading cached interactions...')
            with open(seq_file, 'r', encoding='utf-8') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r', encoding='utf-8') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping

        self.log('[DATASET] Processing interactions...')

        # 加载、过滤、重映射
        interactions_file = os.path.join(input_path, 'interaction.csv')
        item_seqs = self._load_interactions(interactions_file)
        filtered_seqs = self._filter_interactions(item_seqs)
        all_item_seqs, id_mapping = self._remap_ids(filtered_seqs)

        # 保存结果
        with open(seq_file, 'w', encoding='utf-8') as f:
            json.dump(all_item_seqs, f, ensure_ascii=False)
        with open(id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(id_mapping, f, ensure_ascii=False)

        total_interactions = sum(len(items) for items in all_item_seqs.values())
        self.log(f'[DATASET] Processed {len(all_item_seqs)} users, '
                 f'{total_interactions} interactions')

        return all_item_seqs, id_mapping

    def _load_metadata(self, filepath: str) -> Dict[str, dict]:
        """加载物品元数据"""
        self.log(f'[DATASET] Loading metadata from {filepath}')

        metadata = {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1
        except:
            total_rows = 0

        # 使用pandas读取，处理引号包裹的字段
        chunksize = 50000
        chunk_iterator = pd.read_csv(
            filepath,
            chunksize=chunksize,
            iterator=True,
            encoding='utf-8',
            quotechar='"'
        )

        with tqdm(total=total_rows, desc="Reading metadata", unit='row') as pbar:
            for chunk_df in chunk_iterator:
                for _, row in chunk_df.iterrows():
                    item_id = str(row['item_id'])

                    # 提取所有字段
                    meta_info = {
                        'view_number': float(row['view_number']),
                        'comment_number': float(row['comment_number']),
                        'thumbup_number': float(row['thumbup_number']),
                        'share_number': float(row['share_number']),
                        'coin_number': float(row['coin_number']),
                        'favorite_number': float(row['favorite_number']),
                        'barrage_number': float(row['barrage_number']),
                        'title': str(row['title']),
                        'tag': str(row['tag']),
                        'description': str(row['description'])
                    }

                    metadata[item_id] = meta_info
                pbar.update(len(chunk_df))

        return metadata

    def _extract_meta_sentence(self, meta: dict) -> str:
        """从元数据中提取文本句子"""
        # 数值特征转为描述性文本
        stats_parts = []
        stats_fields = [
            ('view_number', 'views'),
            ('comment_number', 'comments'),
            ('thumbup_number', 'thumb-ups'),
            ('share_number', 'shares'),
            ('coin_number', 'coins'),
            ('favorite_number', 'favorites'),
            ('barrage_number', 'barrages')
        ]

        for field, label in stats_fields:
            value = meta.get(field, 0)
            if value > 0:
                # 简化大数字表示
                if value >= 1000000:
                    stats_parts.append(f"{value / 1000000:.1f}M {label}")
                elif value >= 1000:
                    stats_parts.append(f"{value / 1000:.1f}K {label}")
                else:
                    stats_parts.append(f"{int(value)} {label}")

        # 构建句子
        sentence_parts = []

        # 标题
        if meta.get('title'):
            sentence_parts.append(f"Title: {clean_text(meta['title'])}.")

        # 标签
        if meta.get('tag'):
            sentence_parts.append(f"Tag: {clean_text(meta['tag'])}.")

        # 统计信息
        if stats_parts:
            stats_text = "It has " + ", ".join(stats_parts[:-1])
            if len(stats_parts) > 1:
                stats_text += f" and {stats_parts[-1]}"
            else:
                stats_text = stats_parts[0]
            sentence_parts.append(stats_text + ".")

        # 描述
        if meta.get('description'):
            sentence_parts.append(f"Description: {clean_text(meta['description'])}")

        return " ".join(sentence_parts)

    def _process_metadata(self, input_path: str, output_path: str) -> Optional[dict]:
        """处理元数据"""
        process_mode = self.config.get('metadata', 'sentence')
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')

        # 检查缓存
        if os.path.exists(meta_file):
            self.log('[DATASET] Loading cached metadata...')
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        self.log(f'[DATASET] Processing metadata, mode: {process_mode}')

        if process_mode == 'none':
            return None

        # 加载原始元数据
        metadata_file = os.path.join(input_path, 'item_info.csv')
        raw_metadata = self._load_metadata(metadata_file)

        # 只保留在item2id中的物品
        item2id = self.id_mapping.get('item2id', {})
        filtered_metadata = {
            item_id: meta
            for item_id, meta in raw_metadata.items()
            if item_id in item2id
        }

        # 根据模式处理
        if process_mode == 'raw':
            item2meta = filtered_metadata
        elif process_mode == 'sentence':
            item2meta = {}
            for item_id, meta in tqdm(filtered_metadata.items(),
                                      desc="Extracting meta sentences"):
                item2meta[item_id] = self._extract_meta_sentence(meta)
        else:
            raise NotImplementedError(
                f'Metadata processing type "{process_mode}" not implemented.'
            )

        # 保存结果
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(item2meta, f, ensure_ascii=False)

        return item2meta

    def _process_raw_data(self):
        """主处理流程"""
        # 创建目录
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        # 处理交互数据
        self.all_item_seqs, self.id_mapping = self._process_interactions(
            input_path=raw_data_path,
            output_path=processed_data_path
        )

        # 处理元数据
        self.item2meta = self._process_metadata(
            input_path=raw_data_path,
            output_path=processed_data_path
        )

    # 可选：添加数据统计方法
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'num_users': len(self.all_item_seqs),
            'num_items': len(self.id_mapping.get('id2item', [])),
            'total_interactions': sum(len(seq) for seq in self.all_item_seqs.values()),
            'avg_sequence_length': np.mean([len(seq) for seq in self.all_item_seqs.values()]),
            'sparsity': 1 - len(self.all_item_seqs) / (
                    len(self.id_mapping.get('id2user', [])) *
                    len(self.id_mapping.get('id2item', []))
            ) if self.id_mapping.get('id2user') and self.id_mapping.get('id2item') else None
        }
        return stats