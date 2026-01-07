import streamlit as st
import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache' 

from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map
)
from data_utils import load_data
from models_副本 import load_embedding_model
from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents

# ========== 简单回答函数（完全独立，不依赖rag_core.py） ==========
def generate_simple_answer(query, context_docs):
    """最简单版本：只返回检索结果，不依赖任何生成模型"""
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
    
    # 添加综合信息
    response += "\n### 💡 关键信息提取\n"
    
    # 从第一个文档提取核心信息
    if context_docs:
        first_content = context_docs[0].get('content', context_docs[0].get('abstract', ''))
        if first_content:
            # 找第一个完整的句子
            sentences = first_content.split('. ')
            if sentences and len(sentences[0]) > 10:
                response += f"**核心信息：** {sentences[0]}.\n\n"
    
    # 添加说明
    response += "---\n"
    response += "*注：由于实验环境中的PyTorch版本安全限制，生成模型组件暂时受限。以上为基于向量检索的相关文档摘要。*\n"
    response += "*系统已成功实现：数据预处理 → 向量化 → Milvus存储 → 语义检索的全流程。*"
    
    return response

# ========== Streamlit 应用主界面 ==========
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (Milvus Lite)")
st.markdown(f"使用 Milvus Lite 和 `{EMBEDDING_MODEL_NAME}` 构建的医疗问答系统")

# --- 初始化系统组件 ---
milvus_client = get_milvus_client()

if milvus_client:
    # 设置集合
    collection_is_ready = setup_milvus_collection(milvus_client)
    
    # 加载嵌入模型
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    
    # 显示状态
    st.success("✅ 系统初始化成功")
    st.info("⚠️ 注意：由于PyTorch版本安全限制，生成模型暂时禁用，仅展示检索功能")
    
    if collection_is_ready and embedding_model:
        # 加载数据
        pubmed_data = load_data(DATA_FILE)
        
        # 索引数据（如果需要）
        if pubmed_data:
            indexing_successful = index_data_if_needed(milvus_client, pubmed_data, embedding_model)
            if indexing_successful:
                st.success(f"✅ 数据索引完成，已加载 {len(id_to_doc_map) if id_to_doc_map else 0} 个文档")
            else:
                st.warning("⚠️ 数据索引可能不完整")
        else:
            st.warning(f"⚠️ 无法从 {DATA_FILE} 加载数据文件")
            indexing_successful = False
        
        st.divider()
        
        # --- RAG 问答交互部分 ---
        st.header("🧪 医疗问答测试")
        
        # 输入问题
        query = st.text_input("请输入一个医疗相关问题：", 
                            placeholder="例如：什么是白血病？皮肤癌有哪些症状？",
                            key="query_input")
        
        if st.button("🔍 搜索答案", type="primary", key="submit_button") and query:
            start_time = time.time()
            
            # 1. 搜索相似文档
            with st.spinner("正在搜索相关医疗文档..."):
                retrieved_ids, distances = search_similar_documents(milvus_client, query, embedding_model)
            
            if not retrieved_ids:
                st.warning("⚠️ 未找到相关医疗文档，请尝试其他问题")
            else:
                # 2. 从映射中获取文档内容
                retrieved_docs = []
                for idx, doc_id in enumerate(retrieved_ids):
                    if doc_id in id_to_doc_map:
                        doc = id_to_doc_map[doc_id].copy()  # 复制一份避免修改原数据
                        # 添加距离信息
                        if distances and idx < len(distances):
                            doc['distance'] = distances[idx]
                        retrieved_docs.append(doc)
                
                if not retrieved_docs:
                    st.error("❌ 文档映射错误，无法获取文档内容")
                else:
                    # 3. 显示检索到的文档
                    st.subheader("📄 检索到的相关文档")
                    
                    for i, doc in enumerate(retrieved_docs[:3]):  # 只显示前3个
                        with st.expander(f"文档 {i+1}: {doc.get('title', '无标题')[:60]}...", 
                                       expanded=(i == 0)):
                            st.write(f"**标题：** {doc.get('title', '无标题')}")
                            st.write(f"**内容：** {doc.get('abstract', '无内容')}")
                            if 'distance' in doc:
                                st.write(f"**相关度：** {doc['distance']:.4f} (值越小越相关)")
                    
                    st.divider()
                    
                    # 4. 生成并显示答案
                    st.subheader("💡 答案摘要")
                    with st.spinner("正在生成答案摘要..."):
                        answer = generate_simple_answer(query, retrieved_docs)
                        st.markdown(answer)
                    
                    # 显示性能信息
                    end_time = time.time()
                    st.info(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒 | 检索文档数: {len(retrieved_docs)}")
    else:
        st.error("❌ 系统组件初始化失败，请检查日志")
else:
    st.error("❌ Milvus 数据库连接失败")

# ========== 侧边栏：系统配置信息 ==========
st.sidebar.header("⚙️ 系统配置")
st.sidebar.markdown(f"**数据库文件：** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**集合名称：** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件：** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型：** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**检索数量：** Top-{TOP_K}")
st.sidebar.markdown(f"**最大索引数：** {MAX_ARTICLES_TO_INDEX}")

# 显示当前文档数量
doc_count = len(id_to_doc_map) if id_to_doc_map else 0
st.sidebar.markdown(f"**已加载文档：** {doc_count} 条")

# 显示示例问题
st.sidebar.header("💡 示例问题")
st.sidebar.markdown("""
1. 什么是白血病？
2. 皮肤癌有哪些症状？
3. 如何诊断乳腺癌？
4. 癌症的治疗方法有哪些？
5. 什么是化疗？
""")

# 技术说明
st.sidebar.header("📋 技术说明")
st.sidebar.markdown("""
- **RAG系统架构**：检索增强生成
- **向量数据库**：Milvus Lite
- **嵌入模型**：BAAI/bge-small-zh-v1.5
- **检索方式**：余弦相似度
- **数据来源**：GraphRAG医疗数据集
""")