#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析medical.json中的文本结构
"""

import json
import os
import re

def analyze_medical_text(file_path):
    """分析医疗文本的结构"""
    print(f"📂 分析文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 JSON结构:")
    for key, value in data.items():
        print(f"  {key}: {type(value).__name__}")
    
    # 获取context文本
    context_text = data.get('context', '')
    if not context_text:
        print("❌ context字段为空")
        return
    
    print(f"\n📏 文本长度: {len(context_text)} 字符")
    print(f"📄 文本大小: {len(context_text.encode('utf-8')) / 1024:.1f} KB")
    
    # 分析文本结构
    print(f"\n🔍 文本结构分析:")
    
    # 1. 查看开头
    print("开头100字符:")
    print("-" * 50)
    print(context_text[:100])
    print("-" * 50)
    
    # 2. 查看中间部分
    if len(context_text) > 500:
        mid_start = len(context_text) // 2
        print(f"\n中间部分 (位置{mid_start}-{mid_start+100}):")
        print("-" * 50)
        print(context_text[mid_start:mid_start+100])
        print("-" * 50)
    
    # 3. 查看结尾
    if len(context_text) > 200:
        print(f"\n结尾100字符:")
        print("-" * 50)
        print(context_text[-100:])
        print("-" * 50)
    
    # 4. 查找常见的分隔符
    print(f"\n🔧 查找文本分隔模式:")
    
    # 查找可能的标题或分隔符
    patterns = [
        r'\n#+ ',  # Markdown标题
        r'\n\d+\.\s',  # 数字列表
        r'\n•\s',  # 项目符号
        r'\n-{3,}',  # 分隔线
        r'\n[A-Z][a-z]+: ',  # 标题样式
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, context_text[:5000])  # 只检查前5000字符
        if matches:
            print(f"  找到模式 '{pattern}': {len(matches)} 次")
            if matches:
                print(f"    示例: {matches[0]}")
    
    # 5. 按段落分割查看
    paragraphs = [p for p in context_text.split('\n') if p.strip()]
    print(f"\n📑 段落数量 (按换行符): {len(paragraphs)}")
    if paragraphs:
        print(f"第一段: {paragraphs[0][:150]}...")
        print(f"平均段落长度: {sum(len(p) for p in paragraphs[:20])/len(paragraphs[:20]):.0f} 字符")
    
    # 6. 查找关键词
    medical_keywords = ['癌症', '治疗', '症状', '诊断', '药物', '医院', '医生']
    print(f"\n🏥 医疗关键词出现次数:")
    for keyword in medical_keywords:
        count = context_text.count(keyword)
        if count > 0:
            print(f"  {keyword}: {count} 次")
    
    return context_text

if __name__ == "__main__":
    data_file = os.path.join(os.getcwd(), "data", "medical.json")
    
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        exit(1)
    
    print(f"📁 工作目录: {os.getcwd()}")
    print()
    
    text = analyze_medical_text(data_file)
    
    print(f"\n✅ 分析完成")
    print(f"\n💡 对RAG系统的启示:")
    print("1. 需要将长文本分割成chunks（分块）")
    print("2. 分块策略很重要：按段落、固定长度或语义分割")
    print("3. 预处理脚本需要处理字符串格式而非列表格式")