import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import os

# print(sess.run(Z3, feed_dict={X:test}))
def dispose():
    # 使用PIL处理图片，并转为jpg
    im = Image.open(r"./jump_temp.png")
    im = im.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    im = im.crop(region)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, im)
    # plt.imshow(bg)
    # plt.show()
    bg = np.expand_dims(bg, 0)
    return np.array(bg) / 255

def forward_propagation(X, parameters):
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    Z1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, w2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.matmul(P2, w3, name="Z3")
    return Z3

bg = dispose()
train_x = np.load('train_x.zip_files/train_x.npy')
# train_x_2 = np.load('train_x_2.npy')
train_y_2 = np.load('train_y_2.npy')
# plt.imshow(train_x[3,:,:,:])
# plt.show()



sess = tf.Session()
## 这里是恢复graph
saver = tf.train.import_meta_graph('save_model/my_model.meta')
## 这里是恢复各个权重参数
saver.restore(sess, tf.train.latest_checkpoint('./save_model'))

## 获取输入的tensor
graph = tf.get_default_graph()

w1 = graph.get_tensor_by_name('w1:0')
w2 = graph.get_tensor_by_name('w2:0')
w3 = graph.get_tensor_by_name('w3:0')
parameters = {'w1': w1, 'w2': w2, 'w3': w3}

X = graph.get_tensor_by_name('X:0')

Z3 = forward_propagation(X, parameters)
print(Z3)
# sess.run(tf.global_variables_initializer())
y_mean = graph.get_tensor_by_name('y_mean:0')
y_std = graph.get_tensor_by_name('y_std:0')
## 获取到encoder_op
# forward = tf.get_collection("forward")
forward = tf.get_collection("forward")[0]
## 给定数据，运行encoder_op
a = sess.run(Z3, feed_dict={X:bg})
xx = sess.run(Z3, feed_dict={X:train_x[:100]})
# cc = sess.run(Z3, feed_dict={X:train_x_2})
# print(len(cc))
# print(cc-train_y_2[:])

y_mean, y_std = sess.run([y_mean, y_std])
print(a*y_std+y_mean)
print(xx*y_std+y_mean)

