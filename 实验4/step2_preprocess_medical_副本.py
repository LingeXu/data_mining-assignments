#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理 medical.json 数据
将长文本分割成chunks并保存为processed_data.json
"""

import os
import json
import re

def load_medical_data(file_path):
    """加载medical.json数据"""
    print(f"📂 加载数据文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取文本内容
    context_text = data.get('context', '')
    corpus_name = data.get('corpus_name', 'medical')
    
    print(f"📊 数据集名称: {corpus_name}")
    print(f"📏 文本长度: {len(context_text)} 字符")
    
    if not context_text:
        print("❌ 错误: context字段为空")
        return None
    
    return context_text, corpus_name

def split_text_by_paragraphs(text, max_chunk_size=1000, min_chunk_size=200):
    """
    智能分块策略：先按段落分，再对长段落进行二次分割
    
    Args:
        text: 长文本
        max_chunk_size: 最大块大小
        min_chunk_size: 最小块大小
    """
    print("🔪 开始文本分块...")
    
    # 1. 先按换行符分割成段落
    raw_paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    print(f"  原始段落数: {len(raw_paragraphs)}")
    
    chunks = []
    
    # 2. 处理每个段落
    for i, paragraph in enumerate(raw_paragraphs):
        if len(paragraph) <= max_chunk_size:
            # 段落长度合适，直接作为一个chunk
            if len(paragraph) >= min_chunk_size:
                chunks.append(paragraph)
            else:
                # 太短的段落，尝试与下一个段落合并
                if i + 1 < len(raw_paragraphs):
                    combined = paragraph + " " + raw_paragraphs[i + 1]
                    if len(combined) <= max_chunk_size:
                        chunks.append(combined)
                    else:
                        # 合并后还是太长，各自处理
                        if len(paragraph) >= min_chunk_size:
                            chunks.append(paragraph)
                else:
                    # 最后一个段落，如果太短但有一定长度，还是保留
                    if len(paragraph) >= 50:  # 至少50字符
                        chunks.append(paragraph)
        else:
            # 段落太长，需要进一步分割
            # 按句子分割（简单按句号、问号、感叹号分割）
            sentences = re.split(r'(?<=[。！？])', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            current_chunk = ""
            for sentence in sentences:
                # 如果当前chunk加上新句子不超过max_size，就合并
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    current_chunk = current_chunk + sentence if not current_chunk else current_chunk + " " + sentence
                else:
                    # 保存当前chunk，开始新的chunk
                    if current_chunk and len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            # 添加最后一个chunk
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
    
    print(f"  生成chunk数量: {len(chunks)}")
    
    # 3. 统计信息
    if chunks:
        avg_len = sum(len(c) for c in chunks) / len(chunks)
        min_len = min(len(c) for c in chunks)
        max_len = max(len(c) for c in chunks)
        
        print(f"  Chunk长度统计:")
        print(f"    平均: {avg_len:.0f} 字符")
        print(f"    最小: {min_len} 字符")
        print(f"    最大: {max_len} 字符")
    
    return chunks

def save_chunks_to_json(chunks, corpus_name, output_path):
    """保存chunks为Milvus可用的JSON格式"""
    
    milvus_data = []
    
    for i, chunk in enumerate(chunks):
        entry = {
            "id": f"{corpus_name}_{i:06d}",
            "title": f"{corpus_name}_chunk_{i}",
            "abstract": chunk,
            "source_file": "medical.json",
            "chunk_index": i,
            "corpus_name": corpus_name
        }
        milvus_data.append(entry)
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(milvus_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 数据保存到: {output_path}")
    print(f"📋 总记录数: {len(milvus_data)}")
    
    return milvus_data

def main():
    # 配置参数
    input_file = "./data/medical.json"
    output_file = "./data/processed_medical.json"
    
    # 分块参数
    MAX_CHUNK_SIZE = 800  # 最大块大小（字符）
    MIN_CHUNK_SIZE = 100   # 最小块大小（字符）
    
    print("=" * 60)
    print("医疗数据预处理脚本")
    print("=" * 60)
    
    # 1. 加载数据
    result = load_medical_data(input_file)
    if not result:
        return
    
    text, corpus_name = result
    
    # 2. 分块处理
    chunks = split_text_by_paragraphs(
        text, 
        max_chunk_size=MAX_CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE
    )
    
    if not chunks:
        print("❌ 错误: 未生成任何chunk")
        return
    
    # 3. 保存处理结果
    save_chunks_to_json(chunks, corpus_name, output_file)
    
    # 4. 显示样例
    print(f"\n🔍 处理结果样例 (前3个chunk):")
    for i in range(min(3, len(chunks))):
        print(f"\nChunk {i}:")
        print("-" * 40)
        print(chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i])
        print("-" * 40)

if __name__ == "__main__":
    main()