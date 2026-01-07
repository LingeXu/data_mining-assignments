# w2v_vectors.py
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import re

DOWNLOAD_DIR = "/Users/xulingexu/Downloads"
train_path = os.path.join(DOWNLOAD_DIR, "train_sampled.csv")
dev_path   = os.path.join(DOWNLOAD_DIR, "dev_sampled.csv")
test_path  = os.path.join(DOWNLOAD_DIR, "test.csv")

# 读取 3 列无表头 CSV
cols = ['score', 'title', 'body']
train_df = pd.read_csv(train_path, header=None, names=cols, low_memory=False)
dev_df   = pd.read_csv(dev_path,   header=None, names=cols, low_memory=False)
test_df  = pd.read_csv(test_path,  header=None, names=cols, low_memory=False)

def extract_label(df):
    scores = df['score'].astype(str).str.replace('"', '').astype(int)
    # 1星=0，2星=1
    labels = scores.map({1: 0, 2: 1})
    return np.ones(len(df), dtype=bool), labels.values

mask_train, y_train = extract_label(train_df)
mask_dev,   y_dev   = extract_label(dev_df)
mask_test,  y_test  = extract_label(test_df)

# 停用词
STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

def clean_tokenize(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

for df in [train_df, dev_df, test_df]:
    df['tokens'] = df['body'].apply(clean_tokenize)

# 训练 Word2Vec（128 维）
w2v_model = Word2Vec(
    sentences=train_df['tokens'],
    vector_size=128,
    window=5,
    min_count=2,
    workers=4,
    seed=42
)

def tokens_to_vector(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train = np.vstack(train_df['tokens'][mask_train].apply(lambda x: tokens_to_vector(x, w2v_model)))
X_dev   = np.vstack(dev_df['tokens'][mask_dev].apply(lambda x: tokens_to_vector(x, w2v_model)))
X_test  = np.vstack(test_df['tokens'][mask_test].apply(lambda x: tokens_to_vector(x, w2v_model)))

# 保存
save_path = os.path.join(DOWNLOAD_DIR, "w2v_vectors_fixed.npz")
np.savez_compressed(save_path,
                    X_train=X_train, y_train=y_train,
                    X_dev=X_dev,     y_dev=y_dev,
                    X_test=X_test,   y_test=y_test)
w2v_model.save(os.path.join(DOWNLOAD_DIR, "amazon_w2v_128.model"))
print("词向量已保存为：", save_path)
print("训练集:", X_train.shape, y_train.shape)
print("开发集:", X_dev.shape,   y_dev.shape)
print("测试集:", X_test.shape,  y_test.shape)