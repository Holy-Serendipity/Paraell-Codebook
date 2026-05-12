#!/usr/bin/env python3
"""对比 metadata 优化前后的文本效果。

用法:
    python scripts/check_metadata.py --csv /data/cache14/Netease/raw/data_items.csv --n 10
"""

import os
import json
import csv
import random
import argparse


def clean_text_netease(text):
    """from genrec.utils"""
    import re
    from html import unescape
    if not text:
        return text
    text = re.sub(r'<[^>]+>', '', text)
    text = unescape(text)
    text = re.sub(r'[\'"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\\"', '').replace("\\'", '')
    return text


def old_way(row):
    """当前实现: row[1] + '.' + row[2]"""
    return clean_text_netease(row[1] + '.' + row[2])


def new_way(row):
    """优化: 解析 topics + fx_game_other_json 中的关键字段（与 dataset.py 逻辑一致）"""
    parts = []

    # 1. topics: JSON array of tags
    try:
        tags = json.loads(row[1])
        if tags and len(tags) > 0:
            tags_str = "、".join(t.strip() for t in tags if t.strip())
            if tags_str:
                parts.append(tags_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. fx_game_other_json: extract semantic fields
    try:
        meta = json.loads(row[2])

        # name
        name = meta.get('name', '').strip()
        if name:
            parts.append(name)

        # summary — clean # and dedup against topics
        summary = meta.get('summary', '').replace('#', '').strip()
        if summary and summary not in parts:
            parts.append(summary)

        # gender: 0=男性, 1=女性（JSON中可能为字符串"0"/"1"或整数0/1）
        gender_raw = meta.get('gender')
        if gender_raw is not None:
            try:
                gender = int(gender_raw)
                if gender == 0:
                    parts.append('男性')
                elif gender == 1:
                    parts.append('女性')
            except (ValueError, TypeError):
                pass

        # is_original_camera
        if meta.get('is_original_camera') is True:
            parts.append('原创拍摄')

        # allow_remake
        if meta.get('allow_remake') is True:
            parts.append('允许二创')

    except (json.JSONDecodeError, TypeError):
        pass

    return clean_text_netease(". ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='/data/cache14/Netease/raw/data_items.csv')
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ 文件不存在: {args.csv}")
        return

    rows = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"CSV列: {header}")
        for row in reader:
            if len(row) >= 3:
                rows.append(row)

    print(f"\n总行数: {len(rows)}")
    samples = random.sample(rows, min(args.n, len(rows)))

    for i, row in enumerate(samples, 1):
        work_id = row[0]
        old_text = old_way(row)
        new_text = new_way(row)

        print(f"\n{'='*70}")
        print(f"样本 #{i}  work_id={work_id}")
        print(f"{'-'*70}")
        # 解析显示一下原始字段
        try:
            tags = json.loads(row[1])
            print(f"  topics: {tags}")
        except:
            print(f"  topics: {row[1][:80]}...")
        try:
            meta = json.loads(row[2])
            print(f"  name: {meta.get('name', 'N/A')}")
            print(f"  summary: {meta.get('summary', 'N/A')}")
            gender_raw = meta.get('gender', 'N/A')
            print(f"  gender (raw): {gender_raw!r}")
            print(f"  is_original_camera: {meta.get('is_original_camera', 'N/A')}")
            print(f"  allow_remake: {meta.get('allow_remake', 'N/A')}")
        except:
            print(f"  fx_game_other_json: {row[2][:80]}...")
        print(f"{'-'*70}")
        print(f"  [旧] -> {old_text[:150]}")
        print(f"  [新] -> {new_text[:150]}")


if __name__ == '__main__':
    main()
