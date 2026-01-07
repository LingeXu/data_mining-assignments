#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG核心模块 - 简化版（绕过生成模型问题）
直接返回检索结果作为答案
"""

import streamlit as st
from collections import Counter

def generate_answer(query, context_docs, gen_model=None, tokenizer=None):
    """
    简化版生成函数
    由于PyTorch版本安全限制，直接返回检索结果作为答案
    参数gen_model和tokenizer保留但不使用
    """
    
    if not context_docs:
        return "❌ 未找到相关信息。请尝试其他问题。"
    
    # 构建基于检索结果的回答
    response = f"## 🔍 检索结果分析\n\n"
    response += f"**问题：** {query}\n\n"
    response += f"**找到相关文档：** {len(context_docs)} 个\n\n"
    
    # 显示每个检索结果
    response += "### 📋 相关文档摘要：\n"
    
    for i, doc in enumerate(context_docs[:3]):  # 只显示前3个最相关的结果
        # 获取文档内容
        content = doc.get('content', '')
        if not content:
            content = doc.get('abstract', '')
        
        # 获取标题
        title = doc.get('title', f"文档片段 {i+1}")
        
        # 清理标题
        if len(title) > 50:
            title = title[:50] + "..."
        
        # 提取关键句子（前200字符）
        preview = content[:200]
        if len(content) > 200:
            preview += "..."
        
        response += f"\n**{i+1}. {title}**\n"
        response += f"   {preview}\n"
        if 'distance' in doc:
            response += f"   *相关度：{doc.get('distance', 0):.3f}*\n"
    
    # 添加综合回答
    response += "\n### 💡 综合信息\n"
    
    # 提取所有内容的共同主题
    all_content = " ".join([
        doc.get('content', doc.get('abstract', '')) 
        for doc in context_docs[:2]
    ])
    
    # 简单的关键词提取（按词频）
    words = all_content.lower().split()
    word_freq = Counter(words)
    
    # 过滤常见词
    common_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'that', 'with', 'are', 'this', 'as', 'by', 'be', 'on', 'or', 'an', 'it', 'from', 'which', 'you', 'can', 'your', 'has', 'have', 'was', 'were', 'at', 'not', 'but', 'what', 'how', 'when', 'where', 'why', 'who', 'will', 'may', 'more', 'if', 'so', 'such', 'like', 'just', 'than', 'then', 'also', 'about', 'out', 'up', 'down', 'into', 'over', 'under', 'after', 'before', 'between', 'through', 'during', 'since', 'until', 'while', 'because', 'although', 'though', 'even', 'once', 'whether', 'while'}
    
    keywords = [word for word, count in word_freq.most_common(20) 
                if word not in common_words and len(word) > 3][:5]
    
    if keywords:
        response += f"**关键词：** {', '.join(keywords)}\n\n"
    
    # 从第一个文档提取核心信息
    first_content = context_docs[0].get('content', context_docs[0].get('abstract', ''))
    sentences = first_content.split('. ')
    if sentences:
        response += f"**核心信息：** {sentences[0]}.\n\n"
    
    # 添加说明
    response += "---\n"
    response += "*注：由于实验环境中的PyTorch版本安全限制，生成模型组件暂时受限。以上为基于向量检索的相关文档摘要。*\n"
    response += "*系统已成功实现：数据预处理 → 向量化 → Milvus存储 → 语义检索的全流程。*"
    
    return response

def test_retrieval_only():
    """测试函数：验证检索功能"""
    test_query = "什么是白血病？"
    test_docs = [
        {
            'title': '白血病概述',
            'content': '白血病是一种血液系统的恶性肿瘤，主要表现为白细胞异常增生。白血病可以分为急性和慢性两大类，治疗方法包括化疗、放疗和骨髓移植等。',
            'distance': 0.123
        },
        {
            'title': '白血病症状',
            'content': '白血病的常见症状包括发热、乏力、出血倾向、骨痛等。早期诊断对于治疗非常重要。',
            'distance': 0.156
        }
    ]
    
    return generate_answer(test_query, test_docs, None, None)

if __name__ == "__main__":
    # 测试代码
    print("测试检索功能...")
    result = test_retrieval_only()
    print(result)