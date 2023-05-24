# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input


class VGG19Net:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        # self.input_shape = (224, 224, 3)
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.include_top = False
        # include_top：是否保留顶层的3个全连接网络
        # weights：None代表随机初始化，即不加载预训练权重。'imagenet' 代表加载预训练权重
        # input_tensor：可填入Keras tensor作为模型的图像输出tensor
        # input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于48，如(200,200,3)
        # pooling：当include_top = False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。‘avg’代表全局平均池化，‘max’代表全局最大值池化。
        # classes：可选，图片分类的类别数，仅当include_top = True并且不加载预训练权重时可用。
        self.model_vgg = VGG19(include_top=self.include_top, weights=self.weight,
                               input_shape=self.input_shape, pooling=self.pooling)
        self.model_vgg.predict(np.zeros((1, 224, 224, 3)))

    # 提取vgg19最后一层卷积特征
    def pic2vec(self, img_path):
        img = image.image_utils.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.image_utils.img_to_array(img)
        print(f"img shape:{img.shape}")
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model_vgg.predict(img)
        print(feat.shape)
        # norm_feat = feat[0] / LA.norm(feat[0])
        return feat[0]

    def img2vec(self, img):
        img = image.image_utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model_vgg.predict(img)
        # print(feat.shape)
        # norm_feat = feat[0] / LA.norm(feat[0])
        return feat[0]


def img2vec():
    img_path = "./data/chair01.jpg"
    model = VGG19Net()
    f = model.pic2vec(img_path)
    print(f)
    print("done")


def img_clip():
    img_path = "./chair01.jpg"
    o_img = image.image_utils.load_img(img_path)
    o_img_n = image.image_utils.img_to_array(o_img)
    s_img_n = image.image_utils.smart_resize(o_img_n, (500, 500))

    print(f"o_img_n shape: {o_img_n.shape}")
    print(f"s_img_n shape: {s_img_n.shape}")

    image.image_utils.save_img("./chair01s.jpg", s_img_n)


if __name__ == '__main__':
    img2vec()