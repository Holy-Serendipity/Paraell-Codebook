#!/usr/bin/env python3
"""检查语义ID与实际样本数据的对应关系。

从cached文件中读取并展示：item_id → sem_ids → metadata 的映射示例。

用法:
    python scripts/inspect_sem_ids.py
    python scripts/inspect_sem_ids.py --cache_dir /data/cache/ --dataset Netease --n_samples 10
"""

import os
import json
import random
import argparse


def load_json(path, desc):
    print(f"[加载] {desc}: {path}")
    if not os.path.exists(path):
        print(f"  ❌ 文件不存在: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✅ 加载成功，共 {len(data)} 条")
    return data


def main():
    parser = argparse.ArgumentParser(description="检查语义ID与样本对应关系")
    parser.add_argument('--cache_dir', default='/data/cache/', help='缓存根目录')
    parser.add_argument('--dataset', default='Netease', help='数据集名称')
    parser.add_argument('--n_samples', type=int, default=5, help='展示样本数量')
    parser.add_argument('--sem_ids_path', default=None, help='指定sem_ids文件路径（覆盖自动推导）')
    parser.add_argument('--codebook_size', type=int, default=256, help='codebook_size，用于解码token')
    parser.add_argument('--n_codebook', type=int, default=32, help='语义ID的codebook数量')
    args = parser.parse_args()

    base = os.path.join(args.cache_dir, args.dataset, 'processed')

    # ── 1. 加载 id_mapping ──
    id_mapping = load_json(os.path.join(base, 'id_mapping.json'), "ID映射表")

    # ── 2. 加载 metadata ──
    meta = load_json(os.path.join(base, 'metadata.sentence.json'), "物品元数据(文本描述)")

    # ── 3. 加载 sem_ids ──
    if args.sem_ids_path:
        sem_ids_path = args.sem_ids_path
    else:
        # 自动推导路径: 找第一个 .sem_ids 文件
        import glob
        candidates = glob.glob(os.path.join(base, '**', '*.sem_ids'), recursive=True)
        if not candidates:
            print(f"\n❌ 未找到 .sem_ids 文件，请通过 --sem_ids_path 指定")
            return
        sem_ids_path = candidates[0]
        if len(candidates) > 1:
            print(f"\n⚠️  找到多个 .sem_ids 文件，使用第一个:")
            for c in candidates:
                print(f"   {c}")
    item2sem_ids = load_json(sem_ids_path, "语义ID表")
    if item2sem_ids is None:
        return

    # ── 4. 还原原始语义ID（去掉token偏移） ──
    # token = raw_sem_id + codebook_size * digit + 1
    # => raw_sem_id = token - (codebook_size * digit + 1)
    first_key = next(iter(item2sem_ids))
    first_val = item2sem_ids[first_key]
    is_tokenized = isinstance(first_val, list) and len(first_val) > 0 and first_val[0] > 256

    print(f"\n{'='*80}")
    print(f"语义ID查看器")
    print(f"数据集: {args.dataset}")
    print(f"语义ID文件: {sem_ids_path}")
    print(f"物品总数: {len(item2sem_ids)}")
    print(f"元数据数: {len(meta) if meta else 0}")
    print(f"每个物品语义ID维度: {len(first_val)}")
    print(f"数值范围: [{min(first_val)}, {max(first_val)}] → {'已token化' if is_tokenized else '原始PQ索引'}")
    print(f"{'='*80}\n")

    # ── 5. 随机抽取样本展示 ──
    sample_keys = random.sample(list(item2sem_ids.keys()), min(args.n_samples, len(item2sem_ids)))

    for idx, item_id in enumerate(sample_keys, 1):
        tokens = item2sem_ids[item_id]
        n_digit = len(tokens)

        if is_tokenized:
            # 解码为原始语义ID
            raw_ids = []
            for d in range(n_digit):
                raw = tokens[d] - (args.codebook_size * d + 1)
                raw_ids.append(raw)
            display_ids = raw_ids
            label = "原始PQ索引"
        else:
            display_ids = list(tokens)
            label = "PQ索引（未token化）"

        # 获取文本描述
        description = meta.get(item_id, "❌ 无对应元数据") if meta else "N/A"
        # 截断过长描述
        if len(description) > 120:
            description = description[:120] + "..."

        # 获取原始item_id（如果有id_mapping）
        orig_id = item_id
        if id_mapping and 'id2item' in id_mapping:
            # 检查 item_id 在 id_mapping 中的位置
            id2item = id_mapping['id2item']
            if item_id in id_mapping.get('item2id', {}):
                orig_id = f"{item_id} (内部ID: {id_mapping['item2id'][item_id]})"

        print(f"─── 样本 #{idx} ───")
        print(f"  item_id:    {orig_id}")
        print(f"  语义ID ({label}, {n_digit}维):")
        print(f"    {display_ids}")
        # 按位置分组显示，便于观察
        groups = []
        for g in range(0, n_digit, 8):
            chunk = display_ids[g:g+8]
            groups.append(f"    digit[{g:2d}-{g+len(chunk)-1:2d}]: {chunk}")
        print("\n".join(groups))
        print(f"  文本描述:   {description}")
        print()

    # ── 6. 额外统计信息 ──
    if meta:
        has_meta = sum(1 for k in item2sem_ids if k in meta)
        no_meta = len(item2sem_ids) - has_meta
        print(f"--- 统计 ---")
        print(f"  有元数据的物品: {has_meta}/{len(item2sem_ids)}")
        print(f"  无元数据的物品: {no_meta}/{len(item2sem_ids)}")
        if no_meta > 0:
            print(f"  无元数据示例: {random.sample([k for k in item2sem_ids if k not in meta], min(3, no_meta))}")


if __name__ == '__main__':
    main()
