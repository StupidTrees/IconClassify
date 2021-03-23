from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from classification_ds_helper import getImageClassificationDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset, index_to_label = getImageClassificationDataset('F:\\Python workplace\\IconClassify\\data\\', size=[100, 100],
                                                        batch=32, channels=1)

# testDs, _ = getImageClassificationDataset('F:\DeepLearning\datasets\sponge_test', size=[200, 200])
# 构建模型的卷积、池化部分

# 训练模型
total_eff = []
total_acc = []
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(100, 100, 1), filters=8, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))  # 最大值池化，取每4格内的最大值
model.add(keras.layers.Conv2D(filters=8, kernel_size=(4, 4), activation='relu'))  # 再卷积，共64组3*3卷积核，采用relu作为输出
model.add(keras.layers.MaxPooling2D((2, 2), strides=2))  # 再池化
model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dense(78, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(dataset, epochs=40)
print(history)
# eff, acc = model.evaluate(testDs)
# total_acc.append(acc)
# total_eff.append(eff)
model.save('base.h5')

# print(total_eff)
# print(total_acc)
