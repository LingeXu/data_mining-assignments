from datasets import load_dataset

data_path = "data/open-patients/train.jsonl"
dataset = load_dataset("json", data_files=data_path, split="train")

print(f"✅ 成功加载 {len(dataset)} 条样本")
print("第一条样本：")
print(dataset[0])
