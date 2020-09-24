# 在训练集上测试结果
import paddle
import paddle.fluid as fluid
import numpy as np
import os
import random
import shutil
from PIL import Image

def Img_val_predeal(img):# 图片预测处理
    im = np.array(img).astype(np.float32)
    # 矩阵转置
    im = im.transpose((2, 0, 1))     
    im = im / 255.0
    im = np.expand_dims(im, axis=0)  
    return im

error = 0 #识别错误的张数
class_dict ={'gre':'绿灯','red':'红灯'}
label_list = ['gre','red']
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
f=open('result.txt','w')
with fluid.scope_guard(inference_scope):
    # 从指定目录中加载 推理model(inference model)
    [inference_program,  # 预测用的program
     feed_target_names,  
     fetch_targets] = fluid.io.load_inference_model("Red_Green.inference.model",infer_exe)  


    testfile = os.listdir('/home/aistudio/work/测试集')
    for path in testfile:
        ipath = os.path.join('/home/aistudio/work/测试集',path)
        img = Image.open(ipath)
        img = img.resize((224, 224))  #压缩为224x224数组
        im = Img_val_predeal(img)
        results = infer_exe.run(inference_program,  # 运行预测程序
                                feed={feed_target_names[0]: im},  # 喂入要预测的img
                                fetch_list=fetch_targets)  # 得到推测结果
        #print('results', results)
        index = np.argmax(results[0])
        mat = "{:^15}\t{:^4}\t{:^10}\t{:^4}\t{:^10}\n"
        #print(mat.format(path,class_dict[path[:3]],"预测结果:  %s" % class_dict[label_list[index]],'得分：',results[0][0][index]))
        f.write(mat.format(path,class_dict[path[:3]],"预测结果:  %s" % class_dict[label_list[index]],'得分：',results[0][0][index]))
        if path[:3] != label_list[np.argmax(results[0])]:
            error += 1
    f.close()
    print('在测试集上预测的准确率为:',(400-error)/400.)