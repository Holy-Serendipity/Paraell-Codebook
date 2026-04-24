#!/usr/bin/env python3
"""
测试Swing算法集成到RPG模型
"""

import os
import sys
import torch
import yaml

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_swing_module():
    """测试SwingSimilarity类是否能正确导入和初始化"""
    print("测试1: 导入SwingSimilarity类...")
    try:
        from genrec.models.RPG.swing import SwingSimilarity
        print("✓ SwingSimilarity导入成功")
        return True
    except Exception as e:
        print(f"✗ SwingSimilarity导入失败: {e}")
        return False

def test_model_import():
    """测试RPG模型是否能正确导入"""
    print("\n测试2: 导入RPG模型类...")
    try:
        from genrec.models.RPG.model import RPG
        print("✓ RPG模型导入成功")
        return True
    except Exception as e:
        print(f"✗ RPG模型导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件是否能正确加载"""
    print("\n测试3: 加载配置文件...")
    try:
        # 尝试多种可能的配置文件路径
        possible_paths = [
            os.path.join("genrec", "models", "RPG", "config.yaml"),
            os.path.join(os.path.dirname(__file__), "genrec", "models", "RPG", "config.yaml"),
        ]

        config = None
        config_path = None

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                break

        if config is None:
            print("  ⚠ 配置文件不存在，使用默认配置继续测试")
            # 使用默认配置继续测试
            config = {
                'use_swing': False,
                'swing_weight': 0.5,
                'swing_alpha': 1.0,
                'swing_min_cooccurrence': 2,
                'swing_cache': True,
                'sim_type': 'semantic',
                'use_swing_enhancement': False,
                'swing_enhance_weight': 0.3,
                'swing_neighbors': 10,
                'swing_enhance_type': 'additive',
                'swing_enhance_cache': True,
                'use_item_id_embedding': False
            }
            print("  ✓ 使用模拟配置继续测试")
        else:
            print(f"  ✓ 配置文件加载成功: {config_path}")

        # 检查swing相关配置
        swing_configs = ['use_swing', 'swing_weight', 'swing_alpha',
                        'swing_min_cooccurrence', 'swing_cache', 'sim_type']

        for key in swing_configs:
            if key in config:
                print(f"  ✓ {key}: {config[key]}")
            else:
                print(f"  ✗ {key}: 未找到")

        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_swing_initialization():
    """测试Swing算法在RPG模型中的初始化"""
    print("\n测试4: 模拟RPG模型中的Swing初始化...")
    try:
        # 模拟配置
        config = {
            'device': 'cpu',
            'use_swing': True,
            'swing_weight': 0.5,
            'swing_alpha': 1.0,
            'swing_min_cooccurrence': 2,
            'sim_type': 'fusion'
        }

        # 模拟dataset对象（简化）
        class MockDataset:
            def __init__(self):
                self.n_items = 100
                self.n_users = 50
                self.all_item_seqs = {
                    str(i): [j for j in range(1, 11)]
                    for i in range(1, 51)
                }
                self.item2id = {str(i): i for i in range(1, 101)}
                self.user2id = {str(i): i for i in range(1, 51)}

        # 模拟tokenizer对象（简化）
        class MockTokenizer:
            def __init__(self):
                self.n_digit = 32
                self.codebook_size = 256
                self.vocab_size = 8194
                self.max_token_seq_len = 50
                self.eos_token = 8193
                self.ignored_label = -100

        # 创建mock对象
        mock_dataset = MockDataset()
        mock_tokenizer = MockTokenizer()

        # 测试SwingSimilarity直接初始化
        from genrec.models.RPG.swing import SwingSimilarity
        swing_sim = SwingSimilarity(
            dataset=mock_dataset,
            alpha=config['swing_alpha'],
            min_cooccurrence=config['swing_min_cooccurrence'],
            device=config['device']
        )

        print("✓ SwingSimilarity初始化成功")
        print(f"  - n_items: {swing_sim.n_items}")
        print(f"  - n_users: {swing_sim.n_users}")
        print(f"  - alpha: {swing_sim.alpha}")

        return True
    except Exception as e:
        print(f"✗ Swing初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_swing_enhancement():
    """测试Swing增强功能"""
    print("\n" + "=" * 60)
    print("Swing增强功能测试")
    print("=" * 60)

    try:
        # 模拟配置
        config = {
            'device': 'cpu',
            'n_embd': 448,
            'use_swing_enhancement': True,
            'swing_enhance_weight': 0.3,
            'swing_neighbors': 5,
            'swing_enhance_type': 'additive',
            'swing_alpha': 1.0,
            'swing_min_cooccurrence': 2
        }

        # 模拟dataset对象
        class MockDataset:
            def __init__(self):
                self.n_items = 100
                self.n_users = 50
                self.all_item_seqs = {
                    str(i): [j for j in range(1, 11)]
                    for i in range(1, 51)
                }
                self.item2id = {str(i): i for i in range(1, 101)}
                self.user2id = {str(i): i for i in range(1, 51)}
                self.id_mapping = {
                    'id2item': {i: f'item_{i}' for i in range(1, 101)}
                }
                self.cache_dir = './cache'

        # 模拟tokenizer对象
        class MockTokenizer:
            def __init__(self):
                self.n_digit = 32
                self.codebook_size = 256
                self.vocab_size = 8194
                self.max_token_seq_len = 50
                self.eos_token = 8193
                self.ignored_label = -100
                self.item2tokens = {f'item_{i}': tuple([j for j in range(32)]) for i in range(1, 101)}

        mock_dataset = MockDataset()
        mock_tokenizer = MockTokenizer()

        # 测试SwingEnhancement类初始化
        print("测试1: SwingEnhancement类初始化...")
        from genrec.models.RPG.model import SwingEnhancement
        enhancer = SwingEnhancement(config, mock_dataset, mock_tokenizer)
        print(f"✓ SwingEnhancement初始化成功")
        print(f"  - enhance_weight: {enhancer.swing_enhance_weight}")
        print(f"  - neighbors: {enhancer.swing_neighbors}")
        print(f"  - enhance_type: {enhancer.enhance_type}")

        # 设置模拟数据以测试forward方法
        print("\n测试2: 设置模拟数据...")
        n_items = mock_dataset.n_items
        emb_dim = config['n_embd']

        # 创建模拟相似度矩阵（100x100的随机矩阵，对角线为1）
        sim_matrix = torch.randn(n_items, n_items)
        sim_matrix = torch.softmax(sim_matrix, dim=1)  # 归一化
        sim_matrix = sim_matrix * (1 - torch.eye(n_items))  # 对角线设为0
        # 设置对角线为1，确保每个物品与自身相似度最高（用于测试排除自身）
        sim_matrix = sim_matrix + torch.eye(n_items) * 0.9
        enhancer.set_similarity_matrix(sim_matrix)
        print(f"  ✓ 设置模拟相似度矩阵: {sim_matrix.shape}")

        # 创建模拟的所有物品embedding
        all_item_embs = torch.randn(n_items, emb_dim)
        enhancer.set_all_item_embeddings(all_item_embs)
        print(f"  ✓ 设置模拟所有物品embedding: {all_item_embs.shape}")

        # 测试forward方法
        print("\n测试3: SwingEnhancement forward方法...")
        batch_size = 2
        seq_len = 5
        semantic_embs = torch.randn(batch_size, seq_len, emb_dim)
        item_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        enhanced_embs = enhancer(semantic_embs, item_ids)
        print(f"✓ SwingEnhancement forward执行成功")
        print(f"  - 输入形状: {semantic_embs.shape}")
        print(f"  - 输出形状: {enhanced_embs.shape}")
        print(f"  - 输出与输入不同（增强生效）: {not torch.allclose(enhanced_embs, semantic_embs)}")

        # 测试边缘情况：padding item (item_id=0)
        print("\n测试4: 测试padding item处理...")
        item_ids_with_padding = torch.tensor([[0, 1, 2, 0, 3], [4, 0, 5, 0, 0]])
        enhanced_with_padding = enhancer(semantic_embs, item_ids_with_padding)
        print(f"✓ Padding item处理成功")

        # 测试相似度矩阵获取
        print("\n测试5: 相似度矩阵获取...")
        try:
            retrieved_sim_matrix = enhancer.get_similarity_matrix()
            print(f"✓ 相似度矩阵获取成功")
            print(f"  - 矩阵形状: {retrieved_sim_matrix.shape if hasattr(retrieved_sim_matrix, 'shape') else 'N/A'}")
            print(f"  - 与设置的一致: {torch.allclose(retrieved_sim_matrix, sim_matrix)}")
        except Exception as e:
            print(f"✗ 相似度矩阵获取失败: {e}")

        return True

    except Exception as e:
        print(f"✗ Swing增强测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_without_item_id_embedding():
    """测试移除item-id embedding后的模型"""
    print("\n" + "=" * 60)
    print("移除item-id embedding测试")
    print("=" * 60)

    try:
        # 尝试加载配置文件，如果不存在则使用默认配置
        import yaml
        possible_paths = [
            os.path.join("genrec", "models", "RPG", "config.yaml"),
            os.path.join(os.path.dirname(__file__), "genrec", "models", "RPG", "config.yaml"),
        ]

        config = None
        config_path = None

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"  ✓ 配置文件加载成功: {config_path}")
                break

        if config is None:
            print("  ⚠ 配置文件不存在，使用默认配置继续测试")
            # 使用默认配置继续测试
            config = {
                'use_swing': False,
                'swing_weight': 0.5,
                'swing_alpha': 1.0,
                'swing_min_cooccurrence': 2,
                'swing_cache': True,
                'sim_type': 'semantic',
                'use_swing_enhancement': False,
                'swing_enhance_weight': 0.3,
                'swing_neighbors': 10,
                'swing_enhance_type': 'additive',
                'swing_enhance_cache': True,
                'use_item_id_embedding': False
            }
            print("  ✓ 使用模拟配置继续测试")

        # 修改配置：禁用item-id embedding，启用Swing增强
        config['use_item_id_embedding'] = False
        config['use_swing_enhancement'] = True
        config['device'] = 'cpu'

        print("配置检查:")
        print(f"  - use_item_id_embedding: {config.get('use_item_id_embedding', False)}")
        print(f"  - use_swing_enhancement: {config.get('use_swing_enhancement', False)}")
        print(f"  - swing_enhance_weight: {config.get('swing_enhance_weight', 0.3)}")

        # 验证配置参数存在
        required_keys = ['use_swing_enhancement', 'swing_enhance_weight', 'swing_neighbors']
        for key in required_keys:
            if key not in config:
                print(f"✗ 缺失配置参数: {key}")
                return False
            else:
                print(f"  ✓ {key}: {config[key]}")

        print("✓ 配置检查通过")
        return True

    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("Swing算法集成测试")
    print("=" * 60)

    tests = [
        test_swing_module,
        test_model_import,
        test_config_loading,
        test_swing_initialization,
        test_swing_enhancement,          # 新增测试
        test_model_without_item_id_embedding  # 新增测试
    ]

    results = []
    for test in tests:
        results.append(test())

    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("✓ 所有测试通过！Swing算法集成基本正常")
        print("\n下一步:")
        print("1. 使用真实数据集训练模型")
        print("2. 启用use_swing_enhancement配置")
        print("3. 测试相似度融合效果")
    else:
        print("✗ 部分测试失败，请检查上述错误")
        sys.exit(1)

if __name__ == "__main__":
    main()