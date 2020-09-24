all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

train_reader = data_loader(DATADIR,10,'train')

data_shape = [3, 224, 224]
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

#用SqueezeNet网络进行分类，类别数为2
net = SqueezeNet(images,2)
predict =  net.prediction

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label) # 交叉熵
avg_cost = fluid.layers.mean(cost)                            # 计算cost中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)       #使用输入和标签计算准确率

test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer =fluid.optimizer.Adam(learning_rate=0.0001)
optimizer.minimize(avg_cost)
print("完成")
place = fluid.CPUPlace()

# 创建执行器，初始化参数

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

EPOCH_NUM = 5 #训练5轮
model_save_dir = "/home/aistudio/Red_Green.inference.model"

for pass_id in range(EPOCH_NUM):
    # 开始训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed={'images':data[0], 'label':data[1]},  # 喂入一个batch的数据
                                        fetch_list=[avg_cost, acc])  # fetch均方误差和准确率

        # 每5个batch打印一次训练结果
        if batch_id % 5 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    all_train_iter=all_train_iter+10
    all_train_iters.append(all_train_iter)
    all_train_costs.append(train_cost[0])
    all_train_accs.append(train_acc[0])

# 保存模型
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['images'],
                              [predict],
                              exe)
print('训练模型保存完成！')
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")