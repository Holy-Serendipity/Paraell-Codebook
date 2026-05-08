#!/usr/bin/env python3
"""预下载封面图到本地缓存，支持断点续传。

用法:
    python scripts/prefetch_images.py --dataset Netease --cache_dir /data/cache/

可选参数:
    --csv PATH       指定data_items.csv路径（默认 {cache_dir}/{dataset}/raw/data_items.csv）
    --img_dir PATH   指定图像缓存目录（默认 {cache_dir}/{dataset}/images/）
    --workers N      并行下载线程数（默认 8）
"""

import os
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import requests


def download_image(item_id, url, save_path, timeout=15):
    """下载单张图片，成功返回True，失败返回False。"""
    if os.path.exists(save_path):
        return True  # 已存在，跳过
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='预下载封面图到本地缓存')
    parser.add_argument('--dataset', default='Netease', help='数据集名称')
    parser.add_argument('--cache_dir', default='/data/cache/', help='缓存根目录')
    parser.add_argument('--csv', default=None, help='data_items.csv路径（覆盖自动推导）')
    parser.add_argument('--img_dir', default=None, help='图像缓存目录（覆盖自动推导）')
    parser.add_argument('--workers', type=int, default=8, help='并行下载线程数')
    args = parser.parse_args()

    # 路径推导
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(args.cache_dir, args.dataset, 'raw', 'data_items.csv')

    if args.img_dir:
        img_dir = args.img_dir
    else:
        img_dir = os.path.join(args.cache_dir, args.dataset, 'images')

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        exit(1)

    os.makedirs(img_dir, exist_ok=True)

    # 读取CSV，提取封面图URL
    items = []  # [(item_id, cover_url), ...]
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"CSV columns: {header}")
        for row in reader:
            if len(row) >= 4:
                item_id = row[0].strip()
                cover_url = row[3].strip()
                if cover_url:
                    items.append((item_id, cover_url))

    print(f"Total items with cover URLs: {len(items)}")

    # 统计已存在的
    already = sum(1 for item_id, _ in items
                  if os.path.exists(os.path.join(img_dir, f'{item_id}.jpg')))
    print(f"Already cached: {already}")
    print(f"To download: {len(items) - already}")

    # 并行下载
    failed = 0
    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_to_item = {}
        for item_id, url in items:
            save_path = os.path.join(img_dir, f'{item_id}.jpg')
            if os.path.exists(save_path):
                success += 1
                continue
            fut = pool.submit(download_image, item_id, url, save_path)
            fut_to_item[fut] = item_id

        with tqdm(total=len(fut_to_item), desc='Downloading', unit='img') as pbar:
            for fut in as_completed(fut_to_item):
                if fut.result():
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"\nDone! Success: {success}, Failed: {failed}")


if __name__ == '__main__':
    main()
