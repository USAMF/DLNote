#--------------------------------- Tensorflow 框架 ----------------------------------

# 机器学习 -> 算法更复杂 -> 深度学习 -> 1.图像理解(卷积神经网络) 2.语音识别 3.自然语言(循环神经网络)
# 深度学习框架: caffe Tensorflow(Tensorboard) Torch Theano CNTK .....

#--------------------------------- Tensorflow 结构 ----------------------------------

# 计算密集型框架:tensflow..
# IO密集型框架:scrapy django..
#
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 没有GPU 警告开启AVX2
# tf.compat.v1.disable_eager_execution()  # 保证sess.run()能够正常运行
# # 创建一张图包含了一组OP和Tensor 上下文环境
# g = tf.Graph()
# print(g)
# with g.as_default(): # 表示在这个图下
#     c = tf.constant(11.0)
#     print(c.graph)
# print('*'*50)
# # 实现一个加法运算
# a = tf.constant(5.0)
# b = tf.constant(6.0)
# sum1 = tf.add(a,b)
# graph = tf.Graph # 默认的这张图 相当于是给程序分配一段内存
# print(graph)
# print(sum1)
# # 有重载的机制 默认会给运算符重载成op类型
# var1 = 2.0
# sum2 = a + var1
# print(sum2)
# plt = tf.compat.v1.placeholder(tf.float32,[None,3]) # placeholder是op占位符 None表示样本不固定
# # session只能运行一张图 可以在会话当中指定图去运行
# # 只要有会话的上下文环境 就可以使用方便的eval()
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
#     # print(sess.run([a,b,sum1,sum2]))
#     # print(sum1.eval())
#     print(sess.run(plt,feed_dict={plt:[[1,2,3],[4,5,6]]}))
#     print(a.graph)
#     print(sum1.graph)
#     print(sess.graph)
# 数据称为张量(tensor)
# operation(op):专门运算的操作节点 所有操作都是一个op
# 图graph:整个程序的结构 (tensor + flow)
# 会话session:运算程序的图

#--------------------------------- Tensorflow Graph ----------------------------------

# 图graph默认已经注册 一组表示tf.Operation计算单位的对象和tf.Tensor表示操作之间流动的数据单元的对象

# 获取调用:
#   1.tf.Graph
#   2.operation session tensor 都能获取所在的Graph属性

#--------------------------------- Tensorflow OP ----------------------------------

# 标量运算 向量运算 矩阵运算 带状态的运算 神经网络组件 存储/回复 队列及同步运算 控制流
# OP:只要使用tensorflow的API定义的函数接口都是OP
# Tensor:就指的是数据 OP是载体 运载着Tensor

#--------------------------------- Tensorflow 会话 ----------------------------------

# tensorflow分成前端系统和后端系统:
#   前端系统:定义程序的图的结构
#   后端系统:运算图结构

# 会话作用(使用默认注册的图):
#   1.运行图的结构
#   2.分配资源计算
#   3.掌握资源(tensorflow变量的资源 队列 线程)

# 会话开启与关闭:
#   tf.compat.v1.Session().run() 相当于启动图
#   tf.compat.v1.Session().close() 关闭图
#   with tf.compat.v1.Session() as sess:
#       sess.run(...)

# 会话参数
# graph=图
# config=tf.ConfigProto(log_device_placement=True) 显示cpu/gpu信息

# 会话的run()方法 run(fetches,feed_dict=None,graph=None)
#   fetches:要运行的[ops]和计算tensor 支持列表/dict/OrderedDict(重载的运算符也能运行)
#   feed_dict:允许调用者覆盖图中指定张量的值 与tf.compat.v1.placeholder搭配

# 返回值异常 RuntimeError(session状态无效) TypeError(fetches/feed_dict类型不合适)

#--------------------------------- Tensorflow tensor ----------------------------------

# 张量tensor 是Tensorflow的基本数据格式 一个类型化的N维数组 搭载在op
# tensor有三个因素: 1.op类型 2.形状 3.数据类型
# tensor的维度称为阶
# tensor的数据类型: DT_FLOAT DT_DOUBLE DT_INT64/32 DT_STRING DT_BOOL DT_QINT32(用于量化Ops)....

# tensor的属性 :
#   graph
#   op
#   name 张量的op名称
#   shape: 0维() 1维(1) 2维(?,3) 3维(1,2,3)

# tensor形状的概念 Tensorflow动态形状和静态形状在于有没有生成一个新的张量数据
#   静态形状:创建一个张量 初始状态的形状
#       tf.Tensor.get_shape:获取静态形状
#       tf.Tensor.set_shape():更新Tensor对象的静态形状
#   动态形状:一种描述原始张量在执行过程中的一种形状(动态变化)
#       tf.reshape:创建一个具有不同动态形状的新张量

# plt = tf.compat.v1.placeholder(tf.float32,[None,2]) # 形状不固定
# print(plt)
# plt.set_shape([3,2]) # 静态形状只能同维度修改 动态可以修改成不同维度
# print(plt)
# # plt.set_shape([4,2]) # 对于静态形状来说 一旦张量形状固定了 不能再次设置静态形状
# plt_reshape = tf.reshape(plt,[2,3])# 动态形状可以创建一个新的张量 改变时一定要注意元素数量要匹配2*3=6
# print(plt_reshape)

#--------------------------------- Tensorflow的API在官网有介绍-----------------------

# # 生成固定值张量
# a = tf.zeros([3,4],dtype=tf.float32,name=None)
# b = tf.ones([3,4],dtype=tf.float32,name=None)
# c = tf.constant([1,2,3,4],dtype=None,shape=None,name='Const')
#
# # 生成随机值张量
# d = tf.random.normal([2,3],mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
#
# # tensor改变类型
# # tf.string_to_number(string_tensor,out_type=None,name=None)
# # tf.to_float(x,name='toFloat')
# # tf.to_int32(x,name='toInt32')
# # tf.cst(x,dtype,name=None) # 万能转换
# e = tf.cast([[1,2,3],[4,5,6]],tf.float32)
#
# # tensor切片与扩展
# f = tf.concat([e,[[7,8,9],[10,11,12]]],axis=0,name='concat')
# with tf.compat.v1.Session() as sess:
#     print(sess.run([f]))

#--------------------------------- Tensorflow 变量OP -----------------------

# 变量也是一种OP tf.Variable() 是一种特殊的张量 能够持久化存储 它的值就是张量 默认被训练
#   1.变量能够持久化保存 普通张量OP是不行的
#   2.当定义一个变量OP的时候 一定要在会话当中去运行初始化
#   3.name参数 在tensorboard使用的时候显示名字 可以让相同OP名字进行区分

# a = tf.constant(3.0,name='a') # 张量OP
# b = tf.constant(4.0,name='b')
# c  = tf.add(a,b,name='add')
# var = tf.Variable(tf.random.normal([2,3],mean=0.0,stddev=1.0),name='variable') # 此时var没有初始化
# # 必须做一步显示的初始化所有全局变量的OP
# initOP = tf.compat.v1.global_variables_initializer()
# print(a,var)

# with tf.compat.v1.Session() as sess:
#     # 必须运行初始化所有全局变量的OP
#     sess.run(initOP)
#     # 把程序的图结构写入事件文件 graph:把指定的图写入事件文件中 返回filewirter实例
#     filewriter = tf.compat.v1.summary.FileWriter("./summary/",graph=sess.graph)

#     print(sess.run([c,var]))

#--------------------------------- 可视化Tensorboard -----------------------

# 程序图结构 -> 序列化events事件文件 -> tensorboard ->web界面
# tf.summary:摘要
#   tf.compat.v1.summary.FileWriter('文件路径',graph=图)

# 开启:
# tensorboard --logdir="./Documents/DLNote/summary/"
# 可视化:
# 127.0.0.1:6006

# tensorboard默认张量不使用的时候不显示

#--------------------------------- 线性回归案例 --------------------------

# 算法:线性回归 策略:均方误差 优化:梯度下降
# weight bias为何能进行梯度优化 ? tf.Variable(trainable=True) 反向逆推
# 学习率过大 -> 梯度爆炸 -> NaN
# 变量作用域 tf.Variable=scope()
# 增加变量(tesorboard观察参数 损失值变化)显示
# 模型保存和加载:默认保存变量 checkpoint文件 
# 自定义命令行参数:
#   1.首先定义有哪些参数需要在运行时候指定
#   2.程序当中获取定义的命令行参数
#   3.python xxx.py max_step=500 --model_dir="./ckpt/model"

# 参数：名字 默认值 说明
tf.app.flags.DEFINE_integer("max_stpe",100,"模型训练步数")
tf.app.flags.DEFINE_string("model_dir"," ","模型文件加载路径")

# 定义获取命令行参数名字
FLAGS = tf.app.flags.FLAGS


import tensorflow.compat.v1 as tf
import os

def mylinearregression():
    '''
    实现线性回归

    Returns
    -------
    None.

    '''
    
    tf.disable_eager_execution()
    
    #　变量作用域 让模型代码更加清晰
    with tf.variable_scope("data"):
        #　1.准备数据 x特征值[100,1] y目标值[100]
        x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name='xData')
        # 矩阵相乘必须是二维的
        yTrue = tf.matmul(x,[[0.7]]) + 0.8 # [100,1]
    
    with tf.variable_scope("model"):
        # 2.建立线性回归模型 前提要确定数据 因为只有1个特征 -> 那就只需要1个权重w 1个偏置b y=kw+b
        # 随机给一个权重w和偏置b的值 让他去计算损失 然后在当前状态下优化
        # 用变量定义模型参数才能优化
        weight = tf.Variable(tf.random_normal([1,1],mean=0.0,stddev=1.0),name='w') # 注意weight是二维 因为要与x(二维)进行矩阵相乘
        bias = tf.Variable(0.0,name='b')
        
        yPredict = tf.matmul(x,weight) + bias
    
    with tf.variable_scope("loss"):
        # 3.建立损失函数 均方误差 -> 误差平方 
        # tf.reduce_mean() 求和后再求平均值
        loss = tf.reduce_mean(tf.square(yTrue-yPredict))
    
    with tf.variable_scope("optimizer"):
        # 4.梯度下降优化损失函数 learning_rate:0~1
        trainOP = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    # 变量显示第一步:收集变量tensor 
    tf.summary.scalar("losses",loss)
    tf.summary.histogram("weights",weight)
        
    # 变量显示第二步:定义合并tensor的OP
    merged = tf.summary.merge_all()
    
    # 定义一个保存模型的实例
    saver = tf.train.Saver()
    
    # 通过会话运行程序
    initOP = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initOP)
        
        # 打印随机最先初始化的权重和偏置
        print('打印随机最先初始化的权重:%f,偏置:%f' % (sess.run(weight),sess.run(bias)))
        

        # 建立事件文件
        fileWriter = tf.summary.FileWriter("./summary/",graph=sess.graph)
        
        # 加载模型 覆盖模型当中随机定义的参数 从上次训练的参数开始
        if os.path.exists("./ckpt/model/checkpoint"):
            saver.restore(sess,"./ckpt/model")
            
        # 循环训练 运行优化
        for i in range(FLAGS.max_step):
            sess.run(trainOP)
            
            # 变量显示第三步:运行合并的OP
            summary = sess.run(merged)
            # 变量显示第四步:将summary添加到事件文件
            fileWriter.add_summary(summary,i)
            
            print('第%d次优化的权重:%f,偏置:%f' % (i,sess.run(weight),sess.run(bias)))
        
        # 保存模型
        saver.save(sess,"./ckpt/model")
    
    return None

if __name__ == '__main__':
    mylinearregression()