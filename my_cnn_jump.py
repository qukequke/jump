import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

def pull_screen():
    os.system('adb shell screencap -p /sdcard/jump_temp.png')
    os.system('adb pull /sdcard/jump_temp.png .')
    # 使用PIL处理图片，并转为jpg
    im = Image.open(r"./jump_temp.png")
    w, h = im.size
    # 将图片压缩，并截取中间部分，截取后为100*100
    im = im.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    im = im.crop(region)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, im)
    return np.array(bg)/255


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
    Z3 = tf.matmul(P2, w3)
    return Z3


def get_cnn_predict(bg, y_mean, y_std):
    press_time = sess.run(Z3, feed_dict={X: bg})
    y_mean, y_std = sess.run([y_mean, y_std])
    return press_time*y_std + y_mean


def jump(press_time):
    cmd = 'adb shell input swipe 320 410 320 410 ' + str(int(press_time)+10)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph('save_model/my_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./save_model'))
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name('w1:0')
    w2 = graph.get_tensor_by_name('w2:0')
    w3 = graph.get_tensor_by_name('w3:0')
    parameters = {'w1': w1, 'w2': w2, 'w3': w3}
    X = graph.get_tensor_by_name('X:0')

    Z3 = forward_propagation(X, parameters)

    y_mean = graph.get_tensor_by_name('y_mean:0')
    y_std = graph.get_tensor_by_name('y_std:0')

    while True:
        bg = pull_screen()
        bg = np.expand_dims(bg, 0)
        press_time = get_cnn_predict(bg, y_mean, y_std)
        jump(press_time)
        time.sleep(random.uniform(0.9, 1.2))