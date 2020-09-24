import paddle
import paddle.fluid as fluid
import numpy as np
import os
import random
import shutil
from PIL import Image
import matplotlib.pyplot as plt

DATADIR = '/home/aistudio/work/训练集'

def Img_predeal(img):# 图片预处理
    im = np.array(img).astype(np.float32)
    # 矩阵转置
    im = im.transpose((2, 0, 1))      #0，1，2 即通常所说的第一维，第二维，第三维，详情见CSDN
    im = im / 255.0
    #im = np.expand_dims(im, axis=0)  #在第axis维添加数据,预测时需加入此句代码（保证预测的数据和训练的数据维度相同），训练时不能加此句
    return im

def data_loader(datadir, batch_size=10, mode = 'train'):
    filenames = []
    # 将datadir目录下的文件列出来，每条文件都要读入，剔除.xml文件，仅提取.jpg文件
    old_filenames = os.listdir(datadir)
    for of in old_filenames:# 剔除.xml文件，仅提取.jpg文件
        if of[-3:] == 'jpg':
            filenames.append(of)
        else:
            continue# 如果是.xml文件则跳过
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = Image.open(filepath)   #424x240数组
            img = img.resize((224, 224)) #压缩为224x224数组
            im = Img_predeal(img)
            if name[:3] == 'red':
                label = 1
            else:
                label = 0
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(im)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32').reshape(-1, 3, 224, 224)
                labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32').reshape(-1, 3, 224, 224)
            labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()