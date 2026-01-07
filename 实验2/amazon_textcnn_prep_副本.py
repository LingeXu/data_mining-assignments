import pandas as pd

# 读取数据
train = pd.read_csv('/Users/xulingexu/Desktop/train.csv', header=None)
dev   = pd.read_csv('/Users/xulingexu/Desktop/dev.csv',   header=None)
test  = pd.read_csv('/Users/xulingexu/Desktop/test.csv',  header=None)

# 命名列
columns = ['popularity', 'title', 'content']
train.columns = columns
dev.columns   = columns
test.columns  = columns

# 查看前几行
print(train.head())

# 合并 title 和 content，用空格隔开
train['text'] = (train['title'].fillna('') + ' ' + train['content'].fillna('')).str.strip()
dev['text']   = (dev['title'].fillna('')   + ' ' + dev['content'].fillna('')).str.strip()
test['text']  = (test['title'].fillna('')  + ' ' + test['content'].fillna('')).str.strip()

# 把 popularity 从 1/2 映射到 0/1（负面=0，正面=1）
train['popularity'] = train['popularity'] - 1
dev['popularity']   = dev['popularity']   - 1
test['popularity']  = test['popularity']  - 1

# 只看我们需要的两列
train = train[['popularity', 'text']]
dev   = dev[['popularity', 'text']]
test  = test[['popularity', 'text']]

# 快速确认
print(train.head())

import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 文本清洗函数
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # 仅保留字母、数字、空格
    text = re.sub(r"\s+", " ", text).strip()  # 合并多余空格
    return text

train['text'] = train['text'].apply(clean)
dev['text']   = dev['text'].apply(clean)
test['text']  = test['text'].apply(clean)

# 2. 构建词汇表
MAX_VOCAB = 20000
MAX_LEN   = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train['text'])

# 3. 转为序列并填充
X_train = pad_sequences(tokenizer.texts_to_sequences(train['text']), maxlen=MAX_LEN)
X_dev   = pad_sequences(tokenizer.texts_to_sequences(dev['text']),   maxlen=MAX_LEN)
X_test  = pad_sequences(tokenizer.texts_to_sequences(test['text']),  maxlen=MAX_LEN)

y_train = train['popularity'].values
y_dev   = dev['popularity'].values
y_test  = test['popularity'].values

# 4. 快速确认形状
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_dev   shape:", X_dev.shape,   "y_dev   shape:", y_dev.shape)
print("X_test  shape:", X_test.shape,  "y_test  shape:", y_test.shape)

# 打印清洗后的样例
print("\nCleaned sample:")
print(train['text'].iloc[0])

# 打印序列化后的样例
print("\nSequence sample:")
print(X_train[0])

import numpy as np
import os

save_dir = '/Users/xulingexu/Desktop'
np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'X_dev.npy'),   X_dev)
np.save(os.path.join(save_dir, 'X_test.npy'),  X_test)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(save_dir, 'y_dev.npy'),   y_dev)
np.save(os.path.join(save_dir, 'y_test.npy'),  y_test)
print("Saved .npy files to Desktop.")
