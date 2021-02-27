# -*- coding: utf-8 -*-

# Target:Deep Learning Processing

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_eager_execution()

# 1.prepare datasets
trainX = np.linspace(-1,1,100)
trainY = 2 * trainX + np.random.randn(*trainX.shape) * 0.3 # y=2x



# 2.set up model
# 正向搭建模型: x(输入)w(权重)+b -> 累加输入 ->  单个神经元 -> 多个神经元
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
z = tf.multiply(X, W) + b

# 反向搭建模型
cost = tf.reduce_mean(tf.square(Y-z))
learningRate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# 3.Iterative trainning model
init = tf.global_variables_initializer()

trainingEpochs = 20
displayStep = 2

with tf.Session() as sess:
    sess.run(init)
    plotData = {"batchsize":[],"loss":[]}
    
    # 向模型输入数据
    for epoch in range(trainingEpochs):
        for (x,y) in zip(trainX,trainY):
            sess.run(optimizer,feed_dict={X:x,Y:y})
                   
        if epoch % displayStep == 0:
            loss = sess.run(cost,feed_dict={X:trainX,Y:trainY})
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss=="NA"):
                plotData["batchsize"].append(epoch)
                plotData["loss"].append(loss)

    print("Finished!")
    print("cost=",sess.run(cost,feed_dict={X:trainX,Y:trainY}),"W=",sess.run(W),"b=",sess.run(b))

    plt.plot(trainX,trainY,'ro',label='Original data')
    plt.plot(trainX,sess.run(W)*trainX+sess.run(b),label='Fittedline')
    plt.legend()
    plt.show()

    plotdata['avgloss'] = moving_average(plotData['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotData["batchsize"],plotData["avgloss"],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.show()