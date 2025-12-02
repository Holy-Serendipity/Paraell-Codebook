# RPPG

Generate semantic IDs using parallel codebooks. Replace the OpenAI model:text-embedding-3-large(paid API call) with other free embedding_models
- datasets中包含两个数据处理文件，均需要交互历史数据和item元信息，生成三个文件：all_items_seq(交互历史文件)、id_remaps(id双向重映射文件)、metadata_sentences（item描述文本文件）
- main为包装所有程序文件，运行此即可
- pipeline为所有程序入口文件，详细训练流程在此
- default.yaml包含训练基本参数如epoch等
- RPG/config.yaml包含码本基本参数
- datasets中的config为数据集路径等参数