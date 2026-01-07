import matplotlib.pyplot as plt
import numpy as np
import os

print(">>> 脚本已启动")

hist_path = '/Users/xulingexu/Desktop/history.npy'
if not os.path.exists(hist_path):
    print("❌ history.npy 不存在")
    exit(1)

hist = np.load(hist_path, allow_pickle=True).item()
print("✅ 加载成功，keys:", list(hist.keys()))

# 画真实训练曲线
plt.figure(figsize=(6, 4))
plt.plot(hist['accuracy'], label='train acc')
plt.plot(hist['val_accuracy'], label='dev acc')
plt.title('TextCNN Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

# 保存图片
save_path = '/Users/xulingexu/Desktop/accuracy_curve.png'
plt.savefig(save_path, dpi=300)
print(f"✅ 图片已保存到 {save_path}")

# 弹出窗口（macOS）
plt.show()