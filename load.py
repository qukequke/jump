import tensorflow as tf
import numpy as np

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
    # dropout = tf.layers.dropout(inputs=Z3, rate=0.4, training=True)
    # Z3 = tf.layers.dense(inputs=dropout, units=1, name='Z3')
    print(Z3)
    return Z3


train_x = np.load('train_x.zip_files/train_x.npy')
# train_x = train_x.astype('float32')

save_file = "./save_model/my_model.meta"
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(save_file)
    saver.restore(sess, tf.train.latest_checkpoint('./save_model'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    print(X)
    Z3 = graph.get_tensor_by_name("Z2:0")
    w1 = graph.get_tensor_by_name('w1:0')
    w2 = graph.get_tensor_by_name('w2:0')
    w3 = graph.get_tensor_by_name('w3:0')
    parameters = {'w1': w1, 'w2': w2, 'w3': w3}
    Z3_copy = forward_propagation(X, parameters)
    print(Z3_copy)
    # print(sess.run('w1:0'))
    print(X)
    print(Z3)
    a = sess.run(Z3, feed_dict={X: train_x[:4, :, :, :]})
    a1 = sess.run(Z3_copy, feed_dict={X: train_x[:4, :, :, :]})
    print(a)
    print(a1)

