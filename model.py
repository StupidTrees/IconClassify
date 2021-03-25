from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from classification_ds_helper import getImageClassificationDataset
import os
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset, index_to_label = getImageClassificationDataset('data', size=input_size,
                                                        batch=batch_size, channels=1)

# testDs, _ = getImageClassificationDataset('F:\DeepLearning\datasets\sponge_test', size=[200, 200])
# 构建模型的卷积、池化部分

# 训练模型
total_eff = []
total_acc = []
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=(input_size[0], input_size[1], 1), filters=12, kernel_size=(4, 4),
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))  # 最大值池化，取每4格内的最大值
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))  # 再卷积，共64组3*3卷积核，采用relu作为输出
model.add(keras.layers.MaxPooling2D((2, 2), strides=2))  # 再池化
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dense(output_num, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(dataset, epochs=40)
print(history)
model.save('base.h5')
