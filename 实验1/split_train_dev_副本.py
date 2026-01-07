# split_train_dev.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DOWNLOAD_DIR = "/Users/xulingexu/Downloads"
files = ["train_part_1.csv", "train_part_2.csv"]
out_train = os.path.join(DOWNLOAD_DIR, "train_sampled.csv")
out_dev   = os.path.join(DOWNLOAD_DIR, "dev_sampled.csv")

# 流式读取 + 随机划分
train_chunks, dev_chunks = [], []
for f in files:
    reader = pd.read_csv(os.path.join(DOWNLOAD_DIR, f),
                         header=None, names=['score', 'title', 'body'],
                         chunksize=500_000)
    for chunk in reader:
        tr, de = train_test_split(chunk, test_size=0.2,
                                  random_state=42, shuffle=True)
        train_chunks.append(tr)
        dev_chunks.append(de)

pd.concat(train_chunks, ignore_index=True).to_csv(out_train, index=False, header=False)
pd.concat(dev_chunks,   ignore_index=True).to_csv(out_dev,   index=False, header=False)
print("划分完成：")
print("  train_sampled.csv 行数:", sum(len(c) for c in train_chunks))
print("  dev_sampled.csv   行数:", sum(len(c) for c in dev_chunks))