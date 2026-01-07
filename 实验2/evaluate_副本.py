import numpy as np, tensorflow as tf, sklearn.metrics as m
X_test = np.load('/Users/xulingexu/Desktop/X_test.npy')
y_test = np.load('/Users/xulingexu/Desktop/y_test.npy')
model  = tf.keras.models.load_model('/Users/xulingexu/Desktop/textcnn_final.h5')
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(m.classification_report(y_test, y_pred, target_names=['Neg','Pos']))