import json, os, gzip

os.makedirs("data/open-patients", exist_ok=True)

# 100 条模拟病例（含基因/药物/症状/效果）
records = [
    {
        "id": i,
        "text": f"Patient with {gene} mutation was treated with {drug} for {symptom}. Outcome: {effect}."
    }
    for i, (gene, drug, symptom, effect) in enumerate(zip(
        ["BRCA1", "TP53", "EGFR", "KRAS", "ALK"] * 20,
        ["imatinib", "osimertinib", "crizotinib", "trametinib", "lapatinib"] * 20,
        ["lung cancer", "breast cancer", "colorectal cancer", "melanoma", "glioblastoma"] * 20,
        ["partial response", "stable disease", "progression", "complete response", "mild improvement"] * 20
    ))
]

with open("data/open-patients/train.jsonl", "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("✅ 已生成 data/open-patients/train.jsonl（100条样本）")
