#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗RAG系统 - 向量化与存储脚本
适配修改后的中文配置
"""

# ========== 路径修复 ==========
import os
import sys

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上一级就是项目根目录

# 添加项目根目录到Python路径
sys.path.insert(0, project_root)

print(f"📁 脚本目录: {current_dir}")
print(f"📁 项目根目录: {project_root}")
print()
# ========== 结束路径修复 ==========

import json
import time

# 现在应该可以正常导入了
from models_副本 import load_embedding_model
from milvus_utils import get_milvus_client, setup_milvus_collection
from config import (
    COLLECTION_NAME, EMBEDDING_DIM, EMBEDDING_MODEL_NAME, 
    DATA_FILE, id_to_doc_map
)

def load_and_prepare_data():
    """加载并准备数据"""
    print(f"📂 加载数据文件: {DATA_FILE}")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ 错误: 数据文件不存在: {DATA_FILE}")
        print("请确保 config.py 中的 DATA_FILE 路径正确")
        return None, None
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 加载 {len(data)} 条记录")
    
    # 准备文本和元数据
    texts = []
    metadata_list = []
    
    for i, item in enumerate(data):
        # 提取文本（尝试不同字段名）
        text = item.get('abstract', item.get('text', item.get('content', '')))
        if not text or len(text.strip()) < 10:  # 跳过太短的文本
            continue
            
        texts.append(text)
        
        # 构建元数据
        metadata = {
            'id': item.get('id', f"doc_{i}"),
            'title': item.get('title', f"Chunk {i}"),
            'content': text,  # 用于rag_core.py检索
            'abstract': text,
            'chunk_index': item.get('chunk_index', i),
            'source_file': item.get('source_file', 'medical.json')
        }
        metadata_list.append(metadata)
        
        # 填充全局映射（用于rag_core.py）
        id_to_doc_map[i] = metadata
    
    print(f"✅ 准备 {len(texts)} 个有效文本")
    return texts, metadata_list

def batch_vectorize(texts, model):
    """分批向量化文本"""
    print(f"🔢 开始向量化 {len(texts)} 个文本...")
    
    batch_size = 64
    all_embeddings = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # 向量化
        batch_embeddings = model.encode(
            batch_texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.extend(batch_embeddings)
        
        # 显示进度
        progress = (batch_idx + 1) / total_batches * 100
        print(f"  进度: {end_idx}/{len(texts)} ({progress:.1f}%)")
        
        # 每10批或最后一批保存中间结果
        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
            # 可选：保存检查点
            pass
    
    print(f"✅ 向量化完成，生成 {len(all_embeddings)} 个向量")
    if all_embeddings:
        print(f"📏 向量维度: {len(all_embeddings[0])} (应与EMBEDDING_DIM={EMBEDDING_DIM}匹配)")
    
    return all_embeddings

def store_in_milvus(embeddings, metadata_list):
    """存储到Milvus"""
    print(f"🗄️  连接到Milvus...")
    
    # 尝试直接创建客户端（避免streamlit缓存问题）
    try:
        from pymilvus import MilvusClient
        client = MilvusClient("./milvus_lite_data.db")
        print("✅ Milvus客户端创建成功")
    except Exception as e:
        print(f"❌ 创建Milvus客户端失败: {e}")
        print("尝试使用get_milvus_client()...")
        client = get_milvus_client()
        if not client:
            return False
    
    # 检查或创建集合
    print(f"📋 检查集合: {COLLECTION_NAME}")
    
    # 获取所有集合
    collections = client.list_collections()
    print(f"  现有集合: {collections}")
    
    if COLLECTION_NAME in collections:
        print(f"⚠️  集合已存在，正在删除...")
        try:
            client.drop_collection(COLLECTION_NAME)
            print("✅ 旧集合已删除")
        except Exception as e:
            print(f"❌ 删除集合失败: {e}")
            # 继续尝试
    
    # 设置新集合
    print(f"🔨 创建新集合...")
    setup_success = setup_milvus_collection(client)
    if not setup_success:
        print("尝试手动创建集合...")
        # 手动创建集合
        try:
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="chunk_idx", datatype=DataType.INT64)
            
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema
            )
            print("✅ 手动创建集合成功")
        except Exception as e:
            print(f"❌ 手动创建集合失败: {e}")
            return False
    
    # 准备插入数据
    print(f"📥 准备插入数据...")
    
    insert_data = []
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
        insert_data.append({
            "id": i,  # Milvus需要整数ID
            "vector": embedding.tolist(),
            "text": metadata['content'],
            "title": metadata['title'],
            "doc_id": metadata['id'],
            "chunk_idx": metadata['chunk_index']
        })
    
    # 分批插入
    batch_size = 100
    inserted_count = 0
    
    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i:i+batch_size]
        
        try:
            res = client.insert(collection_name=COLLECTION_NAME, data=batch)
            inserted_count += len(batch)
            print(f"  插入批次 {i//batch_size + 1}: "
                  f"{inserted_count}/{len(insert_data)} 条")
        except Exception as e:
            print(f"❌ 批次插入失败，尝试单条插入: {e}")
            # 单条插入
            for item in batch:
                try:
                    client.insert(collection_name=COLLECTION_NAME, data=[item])
                    inserted_count += 1
                except Exception as e2:
                    print(f"   跳过记录: {e2}")
    
    # 创建索引
    print(f"🔍 创建向量索引...")
    try:
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 256}
        }
        client.create_index(
            collection_name=COLLECTION_NAME,
            field_name="vector",
            index_params=index_params
        )
        print("✅ 索引创建成功")
    except Exception as e:
        print(f"⚠️  索引创建失败（可能已存在）: {e}")
    
    # 获取统计信息
    try:
        stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
        print(f"📊 集合统计:")
        print(f"  记录数: {stats['row_count']}")
        # 打印前几个分区信息
        for i, partition in enumerate(stats['partitions'][:3]):
            print(f"  分区{i}: {partition['segment_count']} segments")
    except Exception as e:
        print(f"⚠️  无法获取统计: {e}")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("医疗RAG系统 - 向量化与存储")
    print("=" * 60)
    
    print(f"📝 配置信息:")
    print(f"  嵌入模型: {EMBEDDING_MODEL_NAME}")
    print(f"  向量维度: {EMBEDDING_DIM}")
    print(f"  集合名称: {COLLECTION_NAME}")
    print(f"  数据文件: {DATA_FILE}")
    print()
    
    # 1. 加载数据
    texts, metadata_list = load_and_prepare_data()
    if not texts:
        print("❌ 数据加载失败，请检查DATA_FILE配置")
        return
    
    # 2. 加载模型
    print(f"🧠 加载嵌入模型...")
    start_time = time.time()
    
    # 尝试使用缓存加载
    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if not model:
        # 直接加载
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("✅ 直接加载模型成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("提示: BAAI/bge-small-zh-v1.5 模型约400MB，首次下载需要时间")
            print("可以尝试: pip install sentence-transformers")
            return
    
    model_load_time = time.time() - start_time
    print(f"✅ 模型加载完成 ({model_load_time:.1f}秒)")
    
    # 3. 向量化
    embeddings = batch_vectorize(texts, model)
    if not embeddings:
        print("❌ 向量化失败")
        return
    
    # 4. 存储到Milvus
    success = store_in_milvus(embeddings, metadata_list)
    
    if success:
        print("\n" + "🎉" * 20)
        print("向量化与存储完成！")
        print("🎉" * 20)
        print(f"\n📊 总结:")
        print(f"  文档数量: {len(texts)}")
        print(f"  向量维度: {len(embeddings[0])}")
        print(f"  存储文件: ./milvus_lite_data.db")
        print(f"  集合名称: {COLLECTION_NAME}")
        print(f"\n🚀 下一步:")
        print("  运行: streamlit run app.py")
        print("  访问: http://localhost:8501")
    else:
        print("\n❌ 向量化与存储失败")

if __name__ == "__main__":
    main()