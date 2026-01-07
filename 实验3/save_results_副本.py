import re,json, csv, os
from datasets import load_dataset

ds = load_dataset("json", data_files="data/open-patients/train.jsonl", split="train")

patterns = {
    "gene":     r"\b(?:BRCA1|TP53|EGFR|KRAS|ALK)\b",
    "drug":     r"\b(?:imatinib|osimertinib|crizotinib|trametinib|lapatinib)\b",
    "symptom":  r"\b(?:lung cancer|breast cancer|colorectal cancer|melanoma|glioblastoma)\b",
    "effect":   r"\b(?:partial response|stable disease|progression|complete response|mild improvement)\b"
}

def extract(text):
    return {k: "|".join(re.findall(v, text, flags=re.I)) for k, v in patterns.items()}

os.makedirs("results", exist_ok=True)
with open("results/extracted_entities.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "gene", "drug", "symptom", "effect"])
    for i, rec in enumerate(ds):
        ents = extract(rec["text"])
        writer.writerow([i, rec["text"][:80]+"...", ents["gene"], ents["drug"], ents["symptom"], ents["effect"]])
        if i == 9: break   # 只写前10条做演示

print("✅ 结果已保存到 results/extracted_entities.csv")
