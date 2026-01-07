from gensim.models import Word2Vec
import pandas as pd
import os

# 1. 加载模型
model = Word2Vec.load('/Users/xulingexu/Downloads/amazon_w2v_128.model')

# 2. 准备查询词（可随意增删）
queries = ["music", "excellent", "game", "book", "price"]

# 3. 收集结果
rows = []
for q in queries:
    if q not in model.wv:
        continue
    for w, s in model.wv.most_similar(q, topn=10):
        rows.append({"query": q, "similar_word": w, "similarity": round(s, 4)})

# 4. 写 CSV
os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/similarity.csv", index=False, encoding="utf-8")
print("已生成 results/similarity.csv，共", len(rows), "行")