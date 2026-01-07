import json, re
from datasets import load_dataset

# 1. 加载数据
ds = load_dataset("json", data_files="data/open-patients/train.jsonl", split="train")

# 2. 简易规则提取（演示用）
patterns = {
    "gene":     r"\b(?:BRCA1|TP53|EGFR|KRAS|ALK)\b",
    "drug":     r"\b(?:imatinib|osimertinib|crizotinib|trametinib|lapatinib)\b",
    "symptom":  r"\b(?:lung cancer|breast cancer|colorectal cancer|melanoma|glioblastoma)\b",
    "effect":   r"\b(?:partial response|stable disease|progression|complete response|mild improvement)\b"
}

def extract(text):
    return {k: re.findall(v, text, flags=re.I) for k, v in patterns.items()}

# 3. 处理前 5 条并打印
for i in range(5):
    rec = ds[i]
    ents = extract(rec["text"])
    print(f"--- 样本 {i} ---")
    print("原文:", rec["text"])
    print("实体:", json.dumps(ents, ensure_ascii=False, indent=2))
    print()
