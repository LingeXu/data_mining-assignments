import os

# 输入/输出路径
in_file  = "data/train.txt"
out_train = "data/train.txt"   # 覆盖原文件为 80%
out_dev   = "data/dev.txt"     # 20% 作为验证集

# 读取全部行
with open(in_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 80/20 切分
split_idx = int(0.8 * len(lines))

# 写入
with open(out_train, "w", encoding="utf-8") as f:
    f.writelines(lines[:split_idx])

with open(out_dev, "w", encoding="utf-8") as f:
    f.writelines(lines[split_idx:])

print("划分完成：")
print("  train.txt 行数:", len(lines[:split_idx]))
print("  dev.txt   行数:", len(lines[split_idx:]))