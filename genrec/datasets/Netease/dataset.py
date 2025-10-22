import os
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Optional,Union
import pandas as pd
from genrec.dataset import AbstractDataset
from genrec.utils import download_file, clean_text
from datetime import datetime
import re
import csv
from html import unescape
class Netease(AbstractDataset):
    def __init__(self, config:dict) -> None:
        super(Netease, self).__init__(config)

        self.log(
            f'[Dataset] Netease'
        )
        self.sort_method=config['sort_method']
        self.cache_dir = os.path.join(
            config['cache_dir'], 'Netease'
        )
        self.process_raw()

    def load_likes(self, csv_file_path,sort_method='time')->dict:
        sort_method=self.sort_method
        item_seqs = defaultdict(list)
        # 先获取总行数用于进度条
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # 减去表头
        except:
            total_rows = 0
        self.log(f'[Dataset] Total rows: {total_rows}')
        chunksize=50000
        chunk_iterator=pd.read_csv(csv_file_path, chunksize=chunksize, iterator=True, encoding='utf-8')
        with tqdm(total=total_rows, desc="read data", unit='row') as pbar:
            for chunk_df in chunk_iterator:
                for _, row in chunk_df.iterrows():
                    row_json = json.loads(row.to_json())
                    timestamp = self.trans_unix_time(row_json['ts'])
                    if timestamp is not None:
                        item_seqs[row_json["role_id"]].append((row_json["work_id"], timestamp))
                pbar.update(len(chunk_df))
        if sort_method =='time':
            self.log(f'[Dataset] Ascending sorting')
            for user_id in tqdm(item_seqs, desc="sorting"):
                item_seqs[user_id] = sorted(item_seqs[user_id], key=lambda x: x[1])
        if sort_method =='time_desc':
            self.log(f'[Dataset] Descending sorting')
            for user_id in tqdm(item_seqs,desc="sorting"):
                item_seqs[user_id] = sorted(item_seqs[user_id], key=lambda x: x[1], reverse=True)
        return item_seqs

    def trans_unix_time(self, timestamp) -> int:
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d'
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(str(timestamp), fmt)
                return int(dt.timestamp())
            except ValueError:
                continue
        return 0
    def advanced_filter(self, item_seqs:dict,
                        min_interactions:int=3,
                        max_interactions:int=None,
                        min_time_range:int=None,
                        required_works:list=None)->dict:
        filtered_item_seqs = defaultdict(list)
        total_users = len(item_seqs)
        self.log(f'Total users: {total_users}')
        with tqdm(total=total_users, desc="filtering", unit='user') as pbar:
            for role_id, items in item_seqs.items():
                if len(items) < min_interactions:
                    continue
                if min_time_range:
                    timestamps=[ts for _, ts in items]
                    time_range=max(timestamps)-min(timestamps)
                    if time_range < min_time_range:
                        continue
                if required_works:
                    works={work_id for work_id, _ in items}
                    if not all(work in works for work in required_works):
                        continue
                filtered_item_seqs[role_id]=items
                pbar.update(1)
        self.log(f'Filtered {len(filtered_item_seqs)} users')
        return filtered_item_seqs
    def save_to_json(self, data:dict, output_path:str):
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=None,ensure_ascii=False,default=str)
    def remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        """
        重新映射用户和物品ID

        Args:
            item_seqs (dict): 包含用户-物品序列的字典

        Returns:
            all_item_seqs (dict): 重新映射后的用户-物品序列
            id_mapping (dict): 包含原始ID和重新映射ID之间映射的字典
                - user2id (dict): A dictionary mapping raw user IDs to remapped user IDs.
                - item2id (dict): A dictionary mapping raw item IDs to remapped item IDs.
                - id2user (list): A list mapping remapped user IDs to raw user IDs.
                - id2item (list): A list mapping remapped item IDs to raw item IDs.
        """
        self.log('Remapping user and item IDs...')
        for user, items in item_seqs.items():
            # 映射用户ID
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)

            # 映射物品ID
            mapped_items = []
            for item_id, _ in items:
                if item_id not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item_id] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item_id)
                mapped_items.append(item_id)

            # 存储映射后的序列
            self.all_item_seqs[user] = mapped_items

        return self.all_item_seqs, self.id_mapping
    def process_likes(self, input_path:str, output_path:str)->tuple[dict,dict]:
        seq_file=os.path.join(output_path, "all_item_seqs.json")
        id_mapping_file=os.path.join(output_path, "id_mapping.json")
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[Dataset] Loading all item seqs...')
            with open(seq_file, 'r', encoding='utf-8') as f:
                mapped_item_seqs = json.load(f)
            with open(id_mapping_file, 'r', encoding='utf-8') as f:
                id_mapping = json.load(f)
            return mapped_item_seqs, id_mapping
        self.log('[Dataset] Processing likes seqs...')
        input_file=os.path.join(input_path, "data_likes.csv")
        item_seqs = self.load_likes(input_file, sort_method='time')
        item_seqs_filtered = self.advanced_filter(
            item_seqs,
            min_interactions=3,
            max_interactions=None,
            min_time_range=None,
        )
        mapped_item_seqs, id_mapping = self.remap_ids(item_seqs_filtered)
        self.save_to_json(mapped_item_seqs, os.path.join(output_path,"all_item_seqs.json"))
        self.save_to_json(id_mapping, os.path.join(output_path,"id_mapping.json"))
        total_interactions = sum(len(items) for items in item_seqs_filtered.values())
        self.log(f"处理完成！最终数据: {len(item_seqs_filtered):,} 用户, {total_interactions:,} 交互")
        return mapped_item_seqs, id_mapping

    def list_to_str2(self, l: Union[list, str], remove_blank=False) -> str:
        """
        Converts a list or a string to a string representation.

        Args:
            l (Union[list, str]): The input list or string.

        Returns:
            str: The string representation of the input.
        """
        ret = ''
        if isinstance(l, list):
            ret = ', '.join(map(str, l))
        else:
            ret = l
        if remove_blank:
            ret = ret.replace(' ', '')
        return ret

    def clean_text(self,rawtext):
        """
        清理文本：移除HTML标签、特殊字符和多余空格
        """
        if not rawtext:
            return rawtext

        # 1. 移除HTML标签
        text = self.list_to_str2(rawtext)
        text = re.sub(r'<[^>]+>', '', text)

        # 2. 转换HTML实体（如 &quot; -> "）
        text = unescape(text)

        # 3. 移除所有引号（单引号和双引号）
        text = re.sub(r'[\'"]', '', text)
        # 4. 移除多余的空格和换行符
        text = re.sub(r'\s+', ' ', text).strip()

        # 5. 移除JSON字符串中的转义引号（可选，根据需求）
        text = text.replace('\\"', '').replace("\\'", '')

        # 6. 移除其他特殊字符（保留基本的标点符号）
        # 保留中文、英文、数字、基本标点符号
        text = re.sub(r'[^\w\u4e00-\u9fff\s\.\,\!\?\;:\-\#\(\)\"\']', '', text)

        return text

    def process_metadata(self, input_csv, output_json, item2id:dict)->Optional[dict]:
        """
        处理整个CSV文件，将所有数据存储到字典并写入JSON文件
        """
        result_dict = {}
        try:
            with open(input_csv, 'r', encoding='utf-8') as file:
                total_lines = sum(1 for _ in file) - 1
            # 读取CSV文件
            with open(input_csv, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                # 遍历每一行数据
                item_ids = set(item2id.keys())
                for row in tqdm(reader, total=total_lines, desc="speed", unit="row"):
                    if len(row) >= 3:
                        # 获取第一个字段作为key
                        key = row[0].strip()
                        if key not in item_ids:
                            continue
                        # 获取并清理第三个字段作为value
                        value = clean_text(row[1] + '.' + row[2])

                        # 添加到结果字典
                        result_dict[key] = value

            self.log(f"处理完成！共处理 {len(result_dict)} 条数据")
            self.log(f"结果已保存到: {output_json}")

            return result_dict
        except FileNotFoundError:
            self.log(f"错误: 找不到文件 {input_csv}")
            return {}
        except Exception as e:
            self.log(f"处理文件时发生错误： {e}")
            return {}

    def process_meta(self, input_path:str, output_path:str)->Optional[dict]:
        process_mode = self.config['metadata']
        input_file = os.path.join(input_path, 'data_items.csv')
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.log('[Dataset] Meta data has been processed...')
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        self.log('[Dataset] Processing meta data...')
        if process_mode=='none':
            return None
        if process_mode=='sentence':
            item2meta = self.process_metadata(
                input_csv=input_file,
                output_json=meta_file,
                item2id=self.item2id
            )
            self.save_to_json(item2meta, meta_file)
        else:
            raise NotImplementedError('Metadata processing type not implemented.')
        return item2meta
    def process_raw(self):
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        self.all_item_seqs, self.id_mapping = self.process_likes(
            input_path= raw_data_path,
            output_path= processed_data_path
        )

        self.item2meta = self.process_meta(
            input_path=raw_data_path,
            output_path=processed_data_path
        )