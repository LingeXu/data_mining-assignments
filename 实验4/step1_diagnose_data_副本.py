#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据诊断脚本
专门查看medical.json的实际结构
"""

import json
import os
import sys

def diagnose_data(file_path):
    """诊断数据结构"""
    print(f"📂 诊断文件: {file_path}")
    print(f"📏 文件大小: {os.path.getsize(file_path)} 字节")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # 先读前1000字符看看
            print(f"\n🔍 文件前1000字符:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            
            # 回到文件开头，完整读取
            f.seek(0)
            data = json.load(f)
        
        print(f"\n✅ JSON解析成功")
        print(f"📊 数据Python类型: {type(data)}")
        
        if isinstance(data, list):
            print(f"📋 数据类型: 列表")
            print(f"📈 列表长度: {len(data)}")
            if len(data) > 0:
                print(f"\n📝 第一个元素类型: {type(data[0])}")
                print(f"🔑 第一个元素的键: {list(data[0].keys()) if isinstance(data[0], dict) else '不是字典'}")
                
        elif isinstance(data, dict):
            print(f"📋 数据类型: 字典")
            print(f"🔑 字典的键: {list(data.keys())}")
            # 查看字典的第一个值
            first_key = list(data.keys())[0] if data else None
            if first_key:
                first_value = data[first_key]
                print(f"\n🔍 第一个键值对:")
                print(f"  键: '{first_key}'")
                print(f"  值类型: {type(first_value)}")
                if isinstance(first_value, dict):
                    print(f"  值的键: {list(first_value.keys())}")
        
        else:
            print(f"📋 数据类型: {type(data)}")
            print(f"🔍 数据内容预览: {str(data)[:200]}...")
        
        # 尝试不同方式访问
        print(f"\n🧪 尝试访问数据:")
        
        # 方法1：如果是列表的列表
        if isinstance(data, list) and len(data) > 0:
            print(f"1. data[0]: 成功 - {type(data[0])}")
        else:
            print(f"1. data[0]: 失败 - 不是列表或列表为空")
        
        # 方法2：如果是字典
        if isinstance(data, dict):
            first_key = list(data.keys())[0] if data else None
            print(f"2. data['{first_key}']: 成功 - {type(data.get(first_key))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 使用绝对路径
    data_file = os.path.join(os.getcwd(), "data", "medical.json")
    
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        print(f"当前目录: {os.getcwd()}")
        print(f"data目录内容: {os.listdir('data') if os.path.exists('data') else 'data目录不存在'}")
        sys.exit(1)
    
    success = diagnose_data(data_file)
    
    if success:
        print("\n✅ 数据诊断完成")
        print(f"\n💡 建议:")
        print("1. 如果是字典格式，可能需要用 data.values() 获取文档列表")
        print("2. 如果只有2条数据，可能需要检查是否下载了正确的文件")
    else:
        print("\n❌ 数据诊断失败")