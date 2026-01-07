#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探索 medical.json 中的 context 数据
"""

import json
import os

def explore_context_data(file_path):
    """探索context字段中的数据"""
    print(f"🔍 读取文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "=" * 60)
    print("数据诊断报告")
    print("=" * 60)
    
    # 显示字典的所有键
    print(f"📋 JSON顶层键: {list(data.keys())}")
    
    # 检查每个键的类型和内容
    for key, value in data.items():
        print(f"\n🔑 键: '{key}'")
        print(f"  类型: {type(value)}")
        
        if isinstance(value, list):
            print(f"  列表长度: {len(value)}")
            if len(value) > 0:
                print(f"  第一个元素类型: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"  第一个元素的键: {list(value[0].keys())}")
                    # 显示第一个文档的预览
                    first_doc = value[0]
                    print(f"  文档预览:")
                    for k, v in list(first_doc.items())[:3]:  # 只显示前3个字段
                        if isinstance(v, str) and len(v) > 100:
                            print(f"    {k}: {v[:100]}...")
                        else:
                            print(f"    {k}: {v}")
        
        elif isinstance(value, str):
            print(f"  内容: {value}")
        
        elif isinstance(value, dict):
            print(f"  字典键: {list(value.keys())}")
    
    # 重点分析context字段
    if 'context' in data and isinstance(data['context'], list):
        context_list = data['context']
        print(f"\n📊 CONTEXT字段详细分析:")
        print(f"  文档总数: {len(context_list)}")
        
        if context_list:
            # 统计文档结构
            first_doc = context_list[0]
            print(f"  文档字段: {list(first_doc.keys())}")
            
            # 检查关键字段
            text_fields = ['text', 'content', 'document_text', 'article']
            found_text_field = None
            for field in text_fields:
                if field in first_doc:
                    found_text_field = field
                    break
            
            if found_text_field:
                print(f"  文本字段名: '{found_text_field}'")
                
                # 分析文本长度
                sample_size = min(20, len(context_list))
                texts = [doc.get(found_text_field, '') for doc in context_list[:sample_size]]
                lengths = [len(t) for t in texts]
                
                print(f"\n📏 文本长度分析 (前{sample_size}个文档):")
                print(f"  平均长度: {sum(lengths)/len(lengths):.0f} 字符")
                print(f"  最短长度: {min(lengths)} 字符")
                print(f"  最长长度: {max(lengths)} 字符")
                
                # 显示第一个文档的文本片段
                if texts[0]:
                    print(f"\n🔍 第一个文档文本片段:")
                    print("-" * 50)
                    print(texts[0][:300] + "..." if len(texts[0]) > 300 else texts[0])
                    print("-" * 50)
            else:
                print(f"⚠️  警告: 未找到标准文本字段")
                print(f"  第一个文档的所有字段:")
                for k, v in first_doc.items():
                    print(f"    {k}: {type(v).__name__}")
    else:
        print(f"⚠️  警告: 未找到context字段或context不是列表")
    
    return True

if __name__ == "__main__":
    data_file = os.path.join(os.getcwd(), "data", "medical.json")
    
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        exit(1)
    
    print(f"📁 当前目录: {os.getcwd()}")
    print(f"📄 数据文件: {data_file}")
    print()
    
    explore_context_data(data_file)
    
    print("\n✅ 探索完成")
    print("\n💡 下一步建议:")
    print("1. 如果context字段包含文档列表，预处理时需要提取data['context']")
    print("2. 确认文本字段名，可能是'text'或'content'")
    print("3. 检查文档数量是否足够（应该不止2条）")