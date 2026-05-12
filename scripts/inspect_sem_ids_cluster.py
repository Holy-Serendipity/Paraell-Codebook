#!/usr/bin/env python3
"""语义ID聚类检查器：通过相同digit值查找相似物品（文本+图像）。

核心原理：PQ量化中每个digit对应一个子空间聚类。
  两个物品在digit[d]上值相同 → 它们在子空间d中属于同一簇 → 语义相似。

用法:
    # 文本语义ID，按digit 0分组
    python scripts/inspect_sem_ids_cluster.py \
        --text_sem_ids /data/cache14/Netease/processed/64-8/Qwen3-Embedding-4B_OPQ64,IVF1,PQ64x8.sem_ids \
        --csv /data/cache14/Netease/raw/data_items.csv \
        --digit 0

    # 文本+图像语义ID
    python scripts/inspect_sem_ids_cluster.py \
        --text_sem_ids /data/cache14/Netease/processed/64-8/Qwen3-Embedding-4B_OPQ64,IVF1,PQ64x8.sem_ids \
        --img_sem_ids /data/cache14/Netease/processed/openai_clip-vit-base-patch32_OPQ8x8.img_sem_ids \
        --csv /data/cache14/Netease/raw/data_items.csv \
        --digit 5

    # 图像部分的digit（文本64维后，digit 64起是图像）
    python scripts/inspect_sem_ids_cluster.py \
        --text_sem_ids ... --img_sem_ids ... --csv ... \
        --digit 64
"""

import os
import json
import csv
import random
import re
import argparse
from collections import defaultdict


def load_json(path, desc="file"):
    print(f"[加载] {desc}: {path}")
    if not os.path.exists(path):
        print(f"  ❌ 文件不存在: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✅ 加载成功，共 {len(data)} 条")
    return data


def load_cover_urls(csv_path, item_ids: set) -> dict:
    """从 data_items.csv 第4列提取封面图URL。"""
    cover_urls = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 4:
                    key = str(row[0].strip())
                    if key in item_ids:
                        cover_urls[key] = row[3].strip()
        print(f"  ✅ 加载封面URL: {len(cover_urls)} 条")
    except Exception as e:
        print(f"  ⚠️  封面URL加载失败: {e}")
    return cover_urls


def load_metadata(csv_path, item_ids: set) -> dict:
    """从CSV解析 metadata（文本描述），用于展示。"""
    meta = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    key = str(row[0].strip())
                    if key in item_ids:
                        # 解析topics
                        tags_str = ""
                        try:
                            tags = json.loads(row[1])
                            if tags:
                                tags_str = "、".join(t.strip() for t in tags if t.strip())
                        except (json.JSONDecodeError, TypeError):
                            pass
                        # 解析name
                        name = ""
                        try:
                            meta_json = json.loads(row[2])
                            name = meta_json.get('name', '')
                        except (json.JSONDecodeError, TypeError):
                            pass
                        desc = name if name else tags_str
                        meta[key] = desc[:60] if desc else "(无描述)"
    except Exception:
        pass
    print(f"  ✅ 加载文本描述: {len(meta)} 条")
    return meta


def infer_codebook_size(path: str) -> int:
    """从语义ID文件路径推断 codebook_size。

    路径格式: .../processed/{n_codebook}-{bits}/{model}_{index_factory}.sem_ids
    如 64-8 → bits=8 → 2^8=256
    """
    m = re.search(r'/(\d+)-(\d+)/', path.replace('\\', '/'))
    if m:
        bits = int(m.group(2))
        size = 2 ** bits
        print(f"  ℹ️  从路径推断: codebook_size=2^{bits}={size}")
        return size
    print(f"  ⚠️  无法从路径推断codebook_size，默认256")
    return 256


def group_by_digit(item2sem_ids, digit: int):
    """按指定digit的值分组。返回 {code: [item_ids]}"""
    groups = defaultdict(list)
    for item_id, tokens in item2sem_ids.items():
        if digit < len(tokens):
            code = int(tokens[digit])
            groups[code].append(item_id)
    return groups


def main():
    parser = argparse.ArgumentParser(description="语义ID聚类检查器")
    parser.add_argument('--text_sem_ids', required=True, help='文本语义ID文件路径')
    parser.add_argument('--img_sem_ids', default=None, help='图像语义ID文件路径（可选）')
    parser.add_argument('--csv', required=True, help='data_items.csv路径')
    parser.add_argument('--digit', type=int, default=0, help='要检查的digit位置')
    parser.add_argument('--n', type=int, default=5, help='每个code组展示的样本数')
    parser.add_argument('--top_codes', type=int, default=5, help='展示最热门的多少个code值')
    args = parser.parse_args()

    # ── 自动推断codebook_size ──
    codebook_size = infer_codebook_size(args.text_sem_ids)

    # ── 加载文本语义ID ──
    item2text = load_json(args.text_sem_ids, "文本语义ID")
    if item2text is None:
        return
    all_item_ids = set(item2text.keys())

    # 值范围检查
    first_val = next(iter(item2text.values()))
    n_digit_text = len(first_val)
    all_vals = [v for ids in item2text.values() for v in ids[:5]]
    vmin, vmax = min(all_vals), max(all_vals)
    print(f"  文本: {n_digit_text}维, 值范围 [{vmin}, {vmax}], "
          f"超出codebook范围({vmax >= codebook_size and '⚠️' or '✓'})")

    # ── 加载图像语义ID ──
    item2img = None
    n_digit_img = 0
    if args.img_sem_ids and os.path.exists(args.img_sem_ids):
        item2img = load_json(args.img_sem_ids, "图像语义ID")
        if item2img:
            first_img = next(iter(item2img.values()))
            n_digit_img = len(first_img)
            img_vals = [v for ids in item2img.values() for v in ids[:3]]
            ivmin, ivmax = min(img_vals), max(img_vals)
            print(f"  图像: {n_digit_img}维, 值范围 [{ivmin}, {ivmax}]")

    # ── 加载封面URL和描述 ──
    cover_urls = load_cover_urls(args.csv, all_item_ids)
    descriptions = load_metadata(args.csv, all_item_ids)

    # ── 确定digit所属模态 ──
    if item2img and args.digit >= n_digit_text:
        modality = "图像"
        local_digit = args.digit - n_digit_text
        source = item2img
        print(f"\n📷 digit[{args.digit}] = 图像模态的第{local_digit}维")
    else:
        modality = "文本"
        local_digit = args.digit
        source = item2text
        print(f"\n📝 digit[{args.digit}] = 文本模态的第{local_digit}维")

    if local_digit >= len(next(iter(source.values()))):
        print(f"❌ 超出范围，该模态只有 {len(next(iter(source.values())))} 维")
        return

    # ── 按digit值分组，展示同code的物品 ──
    groups = group_by_digit(source, local_digit)
    total_items = sum(len(v) for v in groups.values())
    print(f"\ndigit[{args.digit}] 共 {len(groups)} 种code值，覆盖 {total_items} 个物品")
    print(f"(如果该位置编码有效，同code值的物品应语义相似)\n")

    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

    for code, items in sorted_groups[:args.top_codes]:
        sample = random.sample(items, min(args.n, len(items)))
        print(f"{'='*70}")
        print(f"  code={code:3d}  │ 共 {len(items)} 个物品  │ 模态: {modality}")
        print(f"{'='*70}")
        for i, item_id in enumerate(sample, 1):
            # 文本描述
            desc = descriptions.get(item_id, "N/A")

            # 文本语义ID前几维
            text_tokens = item2text.get(item_id, [])
            text_head = str(text_tokens[:6]) if text_tokens else "N/A"

            # 图像语义ID
            img_info = ""
            if item2img and item_id in item2img:
                img_tokens = item2img[item_id]
                img_info = f"  img={img_tokens}"

            url = cover_urls.get(item_id, "❌ 无封面")
            print(f"  [{i}] item={item_id}")
            print(f"       text={text_head}{img_info}")
            print(f"       desc: {desc}")
            print(f"       cover: {url}")
        print()

    # ── 图像模态聚类展示 ──
    if item2img:
        img_digit = 0  # 图像部分第0维
        img_groups = group_by_digit(item2img, img_digit)
        img_sorted = sorted(img_groups.items(), key=lambda x: -len(x[1]))

        print(f"{'='*70}")
        print(f"  📷 图像模态 digit[{args.digit + 1}]（图像第{img_digit}维）聚类（自动附加）")
        print(f"  digit[{args.digit + 1}] 共 {len(img_groups)} 种code值")
        print(f"{'='*70}")

        for code, items in img_sorted[:args.top_codes]:
            sample = random.sample(items, min(args.n, len(items)))
            print(f"\n  code={code:3d}  │ 共 {len(items)} 个物品  │ 模态: 图像")
            for i, item_id in enumerate(sample, 1):
                desc = descriptions.get(item_id, "N/A")
                img_tokens = item2img.get(item_id, [])
                url = cover_urls.get(item_id, "❌ 无封面")
                print(f"  [{i}] item={item_id}  img={img_tokens}")
                print(f"       desc: {desc}")
                print(f"       cover: {url}")

if __name__ == '__main__':
    main()
