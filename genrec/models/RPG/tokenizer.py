import os
import math
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import faiss.contrib.torch
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from FlagEmbedding import BGEM3FlagModel
try:
    import vllm
    from vllm import LLM
    HAS_VLLM = True
except Exception as e:
    vllm = None
    LLM = None
    HAS_VLLM = False
    # Log a warning but don't raise - vllm may not be needed for all embedding models
    import warnings
    warnings.warn(f"vllm import failed: {e}. This may not affect functionality if not using vllm-based embedding models.")

try:
    from PIL import Image
    import requests
    from io import BytesIO
    import torch
    HAS_IMAGE_IO = True
except Exception as e:
    Image = None
    requests = None
    BytesIO = None
    HAS_IMAGE_IO = False
    warnings.warn(f"image I/O imports failed: {e}. Image embedding will not be available.")

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_HF_CLIP = True
except Exception as e:
    CLIPModel = None
    CLIPProcessor = None
    HAS_HF_CLIP = False
    warnings.warn(f"HuggingFace CLIP import failed: {e}. Image embedding will not be available.")
class RPGTokenizer(AbstractTokenizer):
    """
    An example when "codebook_size == 256, n_codebooks == 32":
        0: padding
        1-256: digit 1
        257-512: digit 2
        ...
        7937-8192: digit 32
        8193: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        n_codebook_bits (int): The number of bits for the codebook.
        index_factory (str): The index factory name for the OPQ algorithm.
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.index_factory = f'OPQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{self.n_codebook_bits}'

        # 多模态图像配置
        self.use_img_embedding = config.get('use_img_embedding', False)
        self.img_codebook = config.get('img_codebook', 0) if self.use_img_embedding else 0
        self.img_emb_model = config.get('img_emb_model', '')
        self.img_emb_dim = config.get('img_emb_dim', 512)
        self.img_emb_batch_size = config.get('img_emb_batch_size', 64)

        super(RPGTokenizer, self).__init__(config, dataset)
        # Debug dataset structure
        # self.log(f'[TOKENIZER] Dataset structure check:')
        # self.log(
        #     f'[TOKENIZER]  - has item2id: {hasattr(dataset, "item2id")}, type: {type(getattr(dataset, "item2id", None))}')
        # self.log(
        #     f'[TOKENIZER]  - has user2id: {hasattr(dataset, "user2id")}, type: {type(getattr(dataset, "user2id", None))}')
        # self.log(
        #     f'[TOKENIZER]  - has id_mapping: {hasattr(dataset, "id_mapping")}, value: {getattr(dataset, "id_mapping", None)}')

        if hasattr(dataset, 'id_mapping') and dataset.id_mapping is not None:
            self.log(f'[TOKENIZER]  - id_mapping keys: {list(dataset.id_mapping.keys())}')
            if 'id2item' in dataset.id_mapping:
                self.log(f'[TOKENIZER]  - id2item type: {type(dataset.id_mapping["id2item"])}')
                if hasattr(dataset.id_mapping['id2item'], '__len__'):
                    self.log(f'[TOKENIZER]  - id2item length: {len(dataset.id_mapping["id2item"])}')
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        # Check id_mapping before accessing
        if not hasattr(dataset, 'id_mapping') or dataset.id_mapping is None:
            raise ValueError("Dataset.id_mapping is None or not present. Cannot initialize tokenizer.")

        if 'id2item' not in dataset.id_mapping:
            raise ValueError("Dataset.id_mapping does not contain 'id2item' key. Available keys: " + str(
                list(dataset.id_mapping.keys())))
        self.id2item = dataset.id_mapping['id2item']
        self.log(
            f'[TOKENIZER] Successfully set id2item (type: {type(self.id2item)}, length: {len(self.id2item) if hasattr(self.id2item, "__len__") else "N/A"})')

        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        Text codebooks + image codebooks (if enabled).
        """
        text_n = self.config['n_codebook']
        img_n = self.config.get('img_codebook', 0) if self.config.get('use_img_embedding', False) else 0
        return text_n + img_n

    @property
    def codebook_size(self):
        """
        Returns an integer representing the number of codebooks for the tokenizer.
        """
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length, including the EOS token.

        Returns:
            int: The maximum token sequence length.
        """
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        if 'MiniLM' in self.config['sent_emb_model']:
            sent_emb_model = SentenceTransformer(
                self.config['sent_emb_model']
            ).to(self.config['device'])

            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
        elif 'bge' in self.config['sent_emb_model']:
            sent_emb_model = BGEM3FlagModel(
                model_name_or_path=self.config['sent_emb_model'],
                use_fp16=True,
                devices=self.config['device']
            )
            encoding_config={
                'batch_size': self.config['sent_emb_batch_size'],
                'max_length': 8192,
                'return_dense':True,
                'return_sparse':False,
                'return_colbert_vecs':False
            }

            encode_results = sent_emb_model.encode(
                meta_sentences, **encoding_config
            )

            sent_embs = encode_results['dense_vecs']
        elif 'Qwen' in self.config['sent_emb_model']:
            self.log(f"开始编码 {len(meta_sentences)} 个句子")
            self.log(f"batch_size: {self.config['sent_emb_batch_size']}")
            sent_emb_model = SentenceTransformer(
                self.config['sent_emb_model']
            ).to(self.config['device'])
            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device'],
            )
            self.log(f"编码完成，输出形状: {sent_embs.shape}")
            self.log(f"输入句子数: {len(meta_sentences)}")
            self.log(f"输出嵌入数: {sent_embs.shape[0]}")

            # 确保一致性
            if len(meta_sentences) != sent_embs.shape[0]:
                self.log("警告: 输入输出数量不匹配!")
        elif 'text-embedding-3' in self.config['sent_emb_model']:
            from openai import OpenAI
            client = OpenAI(api_key=self.config['openai_api_key'])

            sent_embs = []
            for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                try:
                    responses = client.embeddings.create(
                        input=meta_sentences[i: i + self.config['sent_emb_batch_size']],
                        model=self.config['sent_emb_model']
                    )
                except:
                    self.log(f'[TOKENIZER] Failed to encode sentence embeddings for {i} - {i + self.config["sent_emb_batch_size"]}')
                    batch = meta_sentences[i: i + self.config['sent_emb_batch_size']]

                    from genrec.utils import num_tokens_from_string
                    new_batch = []
                    for sent in batch:
                        n_tokens = num_tokens_from_string(sent, 'cl100k_base')
                        if n_tokens < 8192:
                            new_batch.append(sent)
                        else:
                            n_chars = 8192 / n_tokens * len(sent) - 100
                            new_batch.append(sent[:int(n_chars)])

                    self.log(f'[TOKENIZER] Retrying with {len(new_batch)} sentences')
                    responses = client.embeddings.create(
                        input=new_batch,
                        model=self.config['sent_emb_model']
                    )

                for response in responses.data:
                    sent_embs.append(response.embedding)
            sent_embs = np.array(sent_embs, dtype=np.float32)

        # 释放模型显存
        if 'sent_emb_model' in dir():
            del sent_emb_model
            if 'torch' in globals() or 'torch' in locals():
                torch.cuda.empty_cache()
        sent_embs.tofile(output_path)
        return sent_embs

    # ── 图像Embedding ──────────────────────────────────────────

    def _load_image(self, item_id: str, cover_url: str, cache_dir: str) -> Image.Image:
        """加载封面图：优先读本地缓存，失败则从URL下载。

        Returns:
            PIL.Image or None: 加载成功的图像，失败返回None
        """
        # 尝试本地缓存
        local_path = os.path.join(cache_dir, 'images', f'{item_id}.jpg')
        if os.path.exists(local_path):
            try:
                return Image.open(local_path).convert('RGB')
            except Exception:
                pass

        # 从URL下载
        if not HAS_IMAGE_IO or requests is None:
            return None
        try:
            resp = requests.get(cover_url, timeout=10)
            resp.raise_for_status()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(resp.content)
            return Image.open(BytesIO(resp.content)).convert('RGB')
        except Exception:
            return None

    def _encode_img_emb(self, dataset: AbstractDataset, output_path: str) -> np.ndarray:
        """用HuggingFace CLIP编码封面图embeddings并缓存（加载到GPU）。

        Returns:
            np.ndarray: shape=(n_items-1, img_emb_dim)，第i行对应item_id=i+1。
                        无封面图的item对应行全零。
        """
        n_items = dataset.n_items - 1
        img_embs = np.zeros((n_items, self.img_emb_dim), dtype=np.float32)

        if not HAS_HF_CLIP or CLIPModel is None:
            self.log('[TOKENIZER] HuggingFace CLIP not available, using zero image embeddings')
            img_embs.tofile(output_path)
            return img_embs

        cover_urls = getattr(dataset, 'cover_urls', None) or {}
        if not cover_urls:
            self.log('[TOKENIZER] No cover URLs found, using zero image embeddings')
            img_embs.tofile(output_path)
            return img_embs

        # 从本地路径加载CLIP，FP16减少内存
        self.log(f'[TOKENIZER] Loading CLIP model from {self.img_emb_model}')
        clip_model = CLIPModel.from_pretrained(self.img_emb_model, torch_dtype=torch.float16)
        clip_processor = CLIPProcessor.from_pretrained(self.img_emb_model, use_fast=False)
        clip_model.eval()
        device = self.config.get('device', 'cuda')
        clip_model.to(device)

        cache_dir = dataset.cache_dir
        # 只收集元信息（idx, item_id, url），不加载图片
        valid_items = []  # [(img_idx, item_id, cover_url), ...]
        for i in range(1, dataset.n_items):
            item_id = dataset.id_mapping['id2item'][i]
            cover_url = cover_urls.get(str(item_id), '')
            if cover_url:
                valid_items.append((i - 1, str(item_id), cover_url))

        self.log(f'[TOKENIZER] Encoding {len(valid_items)} cover images ('
                 f'{n_items - len(valid_items)} items have no cover)')

        # 分批加载+编码，避免所有图片同时驻留内存
        batch_size = self.img_emb_batch_size
        for start in tqdm(range(0, len(valid_items), batch_size),
                          desc='Encoding images', unit='batch'):
            batch_items = valid_items[start:start + batch_size]
            # 按需加载当前批次的图片
            batch_imgs = []
            batch_indices = []
            for img_idx, item_id, cover_url in batch_items:
                img = self._load_image(item_id, cover_url, cache_dir)
                if img is not None:
                    batch_imgs.append(img)
                    batch_indices.append(img_idx)
            if not batch_imgs:
                continue
            inputs = clip_processor(images=batch_imgs, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                embs = clip_model.get_image_features(**inputs).cpu().numpy()
            img_embs[batch_indices] = embs
            # 当前批次的图片即刻释放
            del batch_imgs, batch_items

        # 释放CLIP模型和元数据内存
        del clip_model, clip_processor, valid_items
        torch.cuda.empty_cache()
        import gc; gc.collect()
        self.log(f'[TOKENIZER] Image embeddings shape: {img_embs.shape}')
        img_embs.tofile(output_path)
        return img_embs

    def _generate_semantic_id_img(self, img_embs: np.ndarray, sem_ids_path: str):
        """对图像embedding进行OPQ量化，生成图像语义ID。

        Args:
            img_embs: shape=(n_items-1, img_emb_dim) 的图像embedding
            sem_ids_path: 输出缓存路径
        """
        if img_embs.shape[0] == 0:
            self.log('[TOKENIZER] No image embeddings, skipping OPQ')
            return

        img_codebook = self.config['img_codebook']
        img_bits = self.n_codebook_bits  # same bits as text (8 for codebook_size=256)
        img_factory = f'OPQ{img_codebook},IVF1,PQ{img_codebook}x{img_bits}'

        self.log(f'[TOKENIZER] Training image OPQ index: {img_factory}')
        faiss.omp_set_num_threads(self.config.get('faiss_omp_num_threads', 4))

        index = faiss.index_factory(
            img_embs.shape[1],
            img_factory,
            faiss.METRIC_INNER_PRODUCT
        )

        use_gpu = self.config.get('opq_use_gpu', False) and self.img_emb_dim <= 2048
        if use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, self.config.get('opq_gpu_id', 0), index)
                self.log('[TOKENIZER] Image OPQ on GPU')
            except Exception:
                use_gpu = False

        # Train on ALL items (image content is independent of user interactions)
        index.train(img_embs)
        index.add(img_embs)

        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        # Extract PQ codes
        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        # Decode PQ codes to per-codebook indices
        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(img_codebook):
                code.append(bs.read(img_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        # Build {item_id: tuple(img_sem_ids)} mapping
        item2img_ids = {}
        for i in range(pq_codes.shape[0]):
            item_id = i + 1
            if item_id >= len(self.id2item):
                continue
            item = self.id2item[item_id]
            item2img_ids[item] = tuple(pq_codes[i].tolist())

        self.log(f'[TOKENIZER] Saving image semantic IDs to {sem_ids_path}')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2img_ids, f)
        self.log(f'[TOKENIZER] Image semantic IDs saved, {len(item2img_ids)} items')

    # ── 训练Mask ────────────────────────────────────────────────

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        # Check if split_data exists
        if dataset.split_data is None or 'train' not in dataset.split_data:
            # For inference/generation mode, use all items for FAISS training
            self.log(f'[TOKENIZER] No training data found, using all items for FAISS training')
            return np.ones(dataset.n_items - 1, dtype=bool)
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """
        Generates semantic IDs using the OPQ algorithm.

        Args:
            sent_embs (numpy.ndarray): Array of sentence embeddings.
            sem_ids_path (str): Path to save the generated semantic IDs.
            train_mask (numpy.ndarray): Boolean mask indicating the training samples.
        """
        # if self.config['opq_use_gpu']:
        use_gpu = self.config.get('opq_use_gpu', False)
        gpu_initialized = False
        res = None
        co = None

        # Check if GPU FAISS is available
        if use_gpu:
            if hasattr(faiss, 'StandardGpuResources'):
                try:
                    self.log(f'[TOKENIZER] GPU FAISS available, initializing...')
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(1024 * 1024 * 512)
                    if hasattr(faiss, 'GpuClonerOptions'):
                        co = faiss.GpuClonerOptions()
                        co.useFloat16 = self.config['n_codebook'] >= 56
                    gpu_initialized = True
                    self.log(f'[TOKENIZER] GPU FAISS initialized successfully')
                except Exception as e:
                    self.log(f'[TOKENIZER] GPU FAISS initialization failed: {e}, falling back to CPU mode')
                    use_gpu = False
                    gpu_initialized = False
                    # Update config to avoid future GPU attempts
                    self.config['opq_use_gpu'] = False
                    self.log(f'[TOKENIZER] Updated config: opq_use_gpu = False')
            else:
                self.log(f'[TOKENIZER] GPU FAISS not available (no StandardGpuResources), falling back to CPU mode')
                use_gpu = False
                # Update config to avoid future GPU attempts
                self.config['opq_use_gpu'] = False
                self.log(f'[TOKENIZER] Updated config: opq_use_gpu = False')

        faiss.omp_set_num_threads(self.config.get('faiss_omp_num_threads', 4))
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index... (GPU: {use_gpu})')
        if use_gpu and gpu_initialized and res is not None:
            try:
                index = faiss.index_cpu_to_gpu(res, self.config.get('opq_gpu_id', 0), index, co)
                self.log(f'[TOKENIZER] Index moved to GPU successfully')
            except Exception as e:
                self.log(f'[TOKENIZER] Failed to move index to GPU: {e}, falling back to CPU')
                use_gpu = False
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if use_gpu and gpu_initialized and res is not None:
            try:
                index = faiss.index_gpu_to_cpu(index)
                self.log(f'[TOKENIZER] Index moved back to CPU successfully')
            except Exception as e:
                self.log(f'[TOKENIZER] Failed to move index back to CPU: {e}, continuing with GPU index')

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.config['n_codebook']):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        self.log(f'[TOKENIZER] Generating semantic IDs for {pq_codes.shape[0]} items')
        self.log(
            f'[TOKENIZER] id2item type: {type(self.id2item)}, length: {len(self.id2item) if hasattr(self.id2item, "__len__") else "N/A"}')

        if self.id2item is None:
            raise ValueError("self.id2item is None. Cannot map FAISS codes to items.")

        if not isinstance(self.id2item, (dict, list, tuple)):
            raise ValueError(f"self.id2item is not subscriptable. Type: {type(self.id2item)}")
        for i in range(pq_codes.shape[0]):
            # item = self.id2item[i + 1]
            item_id = i + 1
            if item_id >= len(self.id2item):
                self.log(
                    f'[TOKENIZER] Warning: item_id {item_id} out of range for id2item (len={len(self.id2item)}), skipping')
                continue
            try:
                item = self.id2item[item_id]
                item2sem_ids[item] = tuple(pq_codes[i].tolist())
            except (KeyError, IndexError) as e:
                self.log(f'[TOKENIZER] Error accessing id2item[{item_id}]: {e}', level='warning')
                # Try to create a placeholder item ID
                placeholder_item = f"item_{item_id}"
                item2sem_ids[placeholder_item] = tuple(pq_codes[i].tolist())
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        try:
            # Log relevant config keys for debugging
            relevant_keys = ['sent_emb_model', 'sent_emb_dim', 'sent_emb_pca', 'sent_emb_batch_size',
                             'n_codebook', 'codebook_size', 'device', 'metadata']
            config_info = {k: self.config.get(k, 'NOT SET') for k in relevant_keys}
            self.log(f'[TOKENIZER] Config for tokenizer init: {config_info}')

            # Debug dataset structure
            self.log(
                f'[TOKENIZER] Dataset info: n_items={dataset.n_items}, has split_data={dataset.split_data is not None}')
            if hasattr(dataset, 'id_mapping'):
                self.log(f'[TOKENIZER] Dataset has id_mapping: {dataset.id_mapping is not None}')
                if dataset.id_mapping is not None:
                    self.log(f'[TOKENIZER] id_mapping keys: {list(dataset.id_mapping.keys())}')

            # Load semantic IDs
            sem_ids_path = os.path.join(
                dataset.cache_dir, 'processed', f'{self.config["n_codebook"]}-{self.n_codebook_bits}',
                f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
            )
            self.log(f'[TOKENIZER] Semantic IDs path: {sem_ids_path}')
            self.log(f'[TOKENIZER] Semantic IDs file exists: {os.path.exists(sem_ids_path)}')

            if not os.path.exists(sem_ids_path):
                # Load or encode sentence embeddings
                sent_emb_path = os.path.join(
                    dataset.cache_dir, 'processed',
                    f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
                )
                self.log(f'[TOKENIZER] Sentence embeddings path: {sent_emb_path}')
                self.log(f'[TOKENIZER] Sentence embeddings file exists: {os.path.exists(sent_emb_path)}')

                if os.path.exists(sent_emb_path):
                    self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                    # Check if sent_emb_dim is in config
                    if 'sent_emb_dim' not in self.config or self.config['sent_emb_dim'] is None:
                        # Try to infer dimension from file size and number of items
                        file_size = os.path.getsize(sent_emb_path)
                        n_items = dataset.n_items - 1  # item IDs start from 1
                        if n_items > 0:
                            inferred_dim = file_size // (n_items * 4)  # 4 bytes per float32
                            self.config['sent_emb_dim'] = inferred_dim
                            self.log(
                                f'[TOKENIZER] Inferred sent_emb_dim: {inferred_dim} from file size {file_size} and {n_items} items')
                        else:
                            raise ValueError("Cannot infer sent_emb_dim: n_items is 0")
                    sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
                else:
                    self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                    sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
                # PCA
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    sent_embs = pca.fit_transform(sent_embs)
                self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

                # Generate semantic IDs
                training_item_mask = self._get_items_for_training(dataset)
                self.log(
                    f'[TOKENIZER] Training item mask shape: {training_item_mask.shape}, sum: {training_item_mask.sum()}')
                self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)

                # 释放文本embedding和OPQ训练的大内存，为CLIP腾空间
                del sent_embs, training_item_mask
                import gc; gc.collect()
                self.log(f'[TOKENIZER] Freed text embedding memory before loading image models')

            self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
            item2sem_ids = json.load(open(sem_ids_path, 'r'))
            self.log(f'[TOKENIZER] Loaded {len(item2sem_ids)} text semantic IDs')

            # ── 图像语义ID路径 ────────────────────────────────
            if self.use_img_embedding:
                img_model_name = os.path.basename(self.img_emb_model)
                img_sem_ids_path = os.path.join(
                    dataset.cache_dir, 'processed',
                    f'{img_model_name}_OPQ{self.img_codebook}x{self.n_codebook_bits}.img_sem_ids'
                )

                if not os.path.exists(img_sem_ids_path):
                    # Load or encode image embeddings
                    img_emb_path = os.path.join(
                        dataset.cache_dir, 'processed',
                        f'{img_model_name}.img_emb'
                    )

                    if os.path.exists(img_emb_path):
                        self.log(f'[TOKENIZER] Loading image embeddings from {img_emb_path}...')
                        img_embs = np.fromfile(img_emb_path, dtype=np.float32).reshape(-1, self.img_emb_dim)
                    else:
                        self.log(f'[TOKENIZER] Encoding image embeddings...')
                        img_embs = self._encode_img_emb(dataset, img_emb_path)

                    self._generate_semantic_id_img(img_embs, img_sem_ids_path)

                self.log(f'[TOKENIZER] Loading image semantic IDs from {img_sem_ids_path}...')
                img_item2sem_ids = json.load(open(img_sem_ids_path, 'r'))
                self.log(f'[TOKENIZER] Loaded {len(img_item2sem_ids)} image semantic IDs')

                # 拼接文本+图像语义ID: text_ids(32,) + img_ids(8,) = (40,)
                for item in list(item2sem_ids.keys()):
                    text_ids = item2sem_ids[item]
                    img_ids = img_item2sem_ids.get(item, tuple([0] * self.img_codebook))
                    item2sem_ids[item] = text_ids + img_ids

                self.log(f'[TOKENIZER] Concatenated text+image semantic IDs (total n_digit={self.n_digit})')

            item2tokens = self._sem_ids_to_tokens(item2sem_ids)

            return item2tokens

        except Exception as e:
            self.log(f'[TOKENIZER] Error in _init_tokenizer: {str(e)}', level='error')
            import traceback
            self.log(f'[TOKENIZER] Traceback: {traceback.format_exc()}', level='error')
            raise

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing the item sequence.
            split (str): The split type ('train' or 'val' or 'test').

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                # Add 1 as the target item is not included in the input sequence
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            # Tokenize the later items if len(item_seq) > max_item_seq_len + 1
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):],
                pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
