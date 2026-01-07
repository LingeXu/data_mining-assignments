#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版预处理脚本 - 更细粒度的分块
"""

import os
import json
import re

def load_medical_data(file_path):
    """加载medical.json数据"""
    print(f"📂 加载数据文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    context_text = data.get('context', '')
    corpus_name = data.get('corpus_name', 'medical')
    
    print(f"📊 数据集名称: {corpus_name}")
    print(f"📏 文本长度: {len(context_text)} 字符")
    
    return context_text, corpus_name

def split_text_intelligently(text, max_chunk_size=1000, overlap=100):
    """
    智能分块：识别自然标题并分割
    
    观察到的模式：
    1. 数字+空格+大写标题（如"7 Adrenal glands"）
    2. 可能还有其他标题模式
    """
    print("🔪 开始智能分块...")
    
    # 模式1：数字开头+空格+大写单词（可能的章节标题）
    # 例如: "7 Adrenal glands", "8 Adrenal tumors"
    title_pattern1 = r'\n(\d+\s+[A-Z][a-z]+(?:\s+[A-Za-z]+)*)'
    
    # 模式2：纯大写单词（可能的标题）
    # 例如: "KEY POINTS", "SIGNS AND SYMPTOMS"
    title_pattern2 = r'\n([A-Z][A-Z\s]+[A-Z])'
    
    # 模式3：常见医疗章节标题
    medical_sections = [
        r'\n(About\s+.+)',
        r'\n(What is\s+.+\?)',
        r'\n(How is\s+.+\?)',
        r'\n(Signs and symptoms)',
        r'\n(Risk factors)',
        r'\n(Diagnosis)',
        r'\n(Treatment)',
        r'\n(Key points)',
    ]
    
    # 找到所有可能的标题位置
    titles = []
    
    # 查找模式1
    for match in re.finditer(title_pattern1, text):
        titles.append((match.start(), match.group(1), "数字标题"))
    
    # 查找模式2
    for match in re.finditer(title_pattern2, text):
        # 过滤掉太短的（可能不是标题）
        if len(match.group(1).strip()) > 5:
            titles.append((match.start(), match.group(1).strip(), "大写标题"))
    
    # 查找医疗章节
    for pattern in medical_sections:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            titles.append((match.start(), match.group(1), "医疗章节"))
    
    # 按位置排序并去重（相近位置只保留一个）
    titles.sort(key=lambda x: x[0])
    
    # 合并相近的标题（50字符内）
    unique_titles = []
    for title in titles:
        if not unique_titles or title[0] - unique_titles[-1][0] > 50:
            unique_titles.append(title)
    
    print(f"  发现 {len(unique_titles)} 个潜在标题分割点")
    
    # 显示前10个标题
    print("  前10个标题:")
    for i, (pos, title_text, title_type) in enumerate(unique_titles[:10]):
        print(f"    {i+1}. 位置{pos}: [{title_type}] {title_text}")
    
    # 如果没有找到标题，回退到固定长度分块
    if len(unique_titles) < 5:
        print("  警告：标题点太少，使用固定长度分块")
        return split_text_fixed_length(text, max_chunk_size, overlap)
    
    # 根据标题分割文本
    chunks = []
    last_pos = 0
    
    for i in range(len(unique_titles)):
        pos, title_text, title_type = unique_titles[i]
        
        # 当前标题到下一个标题（或结尾）之间的文本
        if i + 1 < len(unique_titles):
            next_pos = unique_titles[i + 1][0]
            chunk_text = text[last_pos:next_pos].strip()
        else:
            chunk_text = text[last_pos:].strip()
        
        # 如果chunk太长，进一步分割
        if chunk_text and len(chunk_text) > 0:
            if len(chunk_text) > max_chunk_size * 1.5:
                # 对长chunk进行二次分割
                sub_chunks = split_text_fixed_length(chunk_text, max_chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        last_pos = pos
    
    # 添加最后一部分（如果有）
    if last_pos < len(text):
        final_chunk = text[last_pos:].strip()
        if final_chunk:
            if len(final_chunk) > max_chunk_size * 1.5:
                sub_chunks = split_text_fixed_length(final_chunk, max_chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(final_chunk)
    
    return chunks

def split_text_fixed_length(text, chunk_size=1000, overlap=100):
    """固定长度分块（备用方案）"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界处结束
        if end < len(text):
            # 寻找最近的句子结束符
            for punct in ['。', '.', '!', '?', '\n']:
                punct_pos = text.rfind(punct, end - 50, end + 50)
                if punct_pos != -1 and punct_pos > start:
                    end = punct_pos + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap  # 设置重叠
        if start <= 0:
            break
    
    return chunks

def save_chunks_to_json(chunks, corpus_name, output_path):
    """保存chunks为Milvus可用的JSON格式"""
    
    milvus_data = []
    
    for i, chunk in enumerate(chunks):
        # 提取chunk的前50字符作为标题
        preview = chunk[:50].replace('\n', ' ')
        
        entry = {
            "id": f"{corpus_name}_{i:06d}",
            "title": f"{preview}...",
            "abstract": chunk,
            "source_file": "medical.json",
            "chunk_index": i,
            "corpus_name": corpus_name,
            "chunk_length": len(chunk)
        }
        milvus_data.append(entry)
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(milvus_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 数据保存到: {output_path}")
    print(f"📋 总记录数: {len(milvus_data)}")
    
    # 统计信息
    if milvus_data:
        lengths = [item["chunk_length"] for item in milvus_data]
        print(f"📊 Chunk长度统计:")
        print(f"    平均: {sum(lengths)/len(lengths):.0f} 字符")
        print(f"    最小: {min(lengths)} 字符")
        print(f"    最大: {max(lengths)} 字符")
        print(f"    <500字符: {sum(1 for l in lengths if l < 500)} 个")
        print(f"    500-1500字符: {sum(1 for l in lengths if 500 <= l < 1500)} 个")
        print(f"    >1500字符: {sum(1 for l in lengths if l >= 1500)} 个")
    
    return milvus_data

def main():
    # 配置参数
    input_file = "./data/medical.json"
    output_file = "./data/processed_medical_v2.json"
    
    # 分块参数
    MAX_CHUNK_SIZE = 1200  # 最大块大小（字符）
    OVERLAP = 150          # 重叠大小
    
    print("=" * 60)
    print("医疗数据预处理脚本 V2（智能分块）")
    print("=" * 60)
    
    # 1. 加载数据
    text, corpus_name = load_medical_data(input_file)
    
    # 2. 智能分块
    chunks = split_text_intelligently(
        text, 
        max_chunk_size=MAX_CHUNK_SIZE,
        overlap=OVERLAP
    )
    
    print(f"\n✅ 生成 {len(chunks)} 个chunks")
    
    # 3. 保存处理结果
    save_chunks_to_json(chunks, corpus_name, output_file)
    
    # 4. 显示样例
    print(f"\n🔍 处理结果样例 (前3个chunk):")
    for i in range(min(3, len(chunks))):
        print(f"\nChunk {i} (长度: {len(chunks[i])} 字符):")
        print("-" * 50)
        print(chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i])
        print("-" * 50)

if __name__ == "__main__":
    main()