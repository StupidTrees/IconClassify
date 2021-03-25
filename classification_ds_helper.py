import pathlib
import random
import tensorflow as tf
import cv2
import numpy as np


def alpha2white_opencv2(img):
    sp = img.shape
    width = sp[0]
    height = sp[1]
    if sp[2] == 3:
        return img
    for yh in range(height):
        for xw in range(width):
            color_d = img[xw, yh]
            if color_d[3] == 0:
                img[xw, yh] = [255, 255, 255, 255]
    r, g, b, a = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img


def getImageClassificationDataset(path, size, batch=10, repeat=False, channels=3):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # 使用pathlib创建Path对象，表示根目录
    data_root = pathlib.Path(path)
    # 该目录下所有文件（递归）的路径列表，glob为正则搜寻，*/*表示该目录包括子目录里所有文件
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]  # 转化为字符串类型
    random.shuffle(all_image_paths)  # 随机排序

    label_names = [item.name for item in data_root.glob('*/') if item.is_dir()]  # 在根目录中查找,筛选出目录对象，获取其名称成为看i额表
    label_to_index = dict((name, index) for index, name in enumerate(label_names))  # 获得各名称-下标的字典
    index_to_label = dict((index, name) for index, name in enumerate(label_names))
    # 根据所有图片目录所在的文件夹名称，找到其对应的分类下标，制成训练数据Label
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    # 写载入图片并规范化的函数
    def load_and_preprocess_image(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 读入图片文件
        image = cv2.resize(image, size)  # 把图片规范化为192*192*1x
        image = alpha2white_opencv2(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=2)
        ts = tf.convert_to_tensor(gray, dtype=float)
        ts /= 255.0  # normalize to [0,1] range  # 把数值范围规范到0~1
        return ts

    # 将上述资源封装为Tensorflow的Datasets

    # 创建路径的数据集
    # path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    # 使用map将路径集经函数load_and_prosess_image映射到图片数据集image_ds，第一个参数传入的是函数
    # image_ds = path_ds.map(load_and_preprocess_image)  # , num_parallel_calls=AUTOTUNE)

    tensors = [load_and_preprocess_image(p) for p in all_image_paths]
    image_ds = tf.data.Dataset.from_tensor_slices(tensors)

    # 同样用该函数可以创建标签数据集
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    # 二者可以打包，成为图片-标签数据集
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # 上面的操作便构建好数据集了，下面演示训练时数据集的使用
    ds = image_label_ds.shuffle(buffer_size=len(all_image_paths))  # 先打乱
    if repeat:
        ds = ds.repeat()  # 后重复
    ds = ds.batch(batch)  # 设置批次
    ds = ds.prefetch(buffer_size=AUTOTUNE)  # 设置模型在训练时后台加载batch
    return ds, index_to_label
