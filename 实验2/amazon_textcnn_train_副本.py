# amazon_textcnn_train.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# 1. 加载预处理好的数据
data_dir = '/Users/xulingexu/Desktop'
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
X_dev   = np.load(os.path.join(data_dir, 'X_dev.npy'))
X_test  = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
y_dev   = np.load(os.path.join(data_dir, 'y_dev.npy'))
y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))

# 2. 超参数
VOCAB_SIZE = 20000
EMB_DIM    = 128
MAX_LEN    = 100
FILTERS    = 128
KERNEL_SZ  = 3
DROPOUT    = 0.5
BATCH      = 256
EPOCHS     = 5

# 3. 构建 TextCNN（Kim 2015 简化版）
model = Sequential([
    Embedding(VOCAB_SIZE, EMB_DIM, input_length=MAX_LEN),
    Conv1D(FILTERS, KERNEL_SZ, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# 4. 回调：早停 + 保存最佳
ckpt_path = os.path.join(data_dir, 'textcnn_best.h5')
callbacks = [
    EarlyStopping(patience=2, restore_best_weights=True),
    ModelCheckpoint(ckpt_path, save_best_only=True, verbose=1)
]

# 5. 训练
history = model.fit(
    X_train, y_train,
    validation_data=(X_dev, y_dev),
    batch_size=BATCH,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# 6. 在测试集评估
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# 7. 保存模型（结构 + 权重）
model.save('/Users/xulingexu/Desktop/textcnn_final.h5')

# 8. 保存训练历史（loss/acc 曲线数据）
np.save('/Users/xulingexu/Desktop/history.npy', history.history)

print("模型已保存：textcnn_final.h5")
print("训练曲线数据已保存：history.npy")