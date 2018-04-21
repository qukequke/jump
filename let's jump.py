import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

tf.reset_default_graph()
start_time = time.time()

def initial_parameters():
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    w1 = tf.get_variable('w1', [4, 4, 3, 8], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    print(w1)
    w2 = tf.get_variable('w2', [2, 2, 8, 16], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w3 = tf.get_variable('w3', [256, 1], dtype=tf.float32, initializer=tf.zeros_initializer)
    parameters = {'w1': w1, 'w2': w2, 'w3': w3}
    return parameters


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
    Z3 = tf.matmul(P2, w3, name='Z2')
    # dropout = tf.layers.dropout(inputs=Z3, rate=0.4, training=True)
    # Z3 = tf.layers.dense(inputs=dropout, units=1, name='Z3')
    print(Z3)
    # Z3 = tf.layers.Dense(inputs=P2, units=1, name='w3', activation=None)
    return Z3


def compute_cost(Z3, Y):
    '''
    利用tf.nn 计算损失函数
    与nn不同在与 数据的排列方式不同了，nn是样本再axis1，这个是在axis0
    '''
    return tf.reduce_mean(tf.square(Z3-Y))


def creat_place_holder(nH0, nW0, n_C0, n_y):
    '''
    创建x,y placeholder
    还是数据排列方式不同了
    '''
    X = tf.placeholder(dtype=tf.float32, shape=[None, nH0, nW0, n_C0], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name='Y')
    return X, Y


def load_data():
    '''
    载入数据并处理  包括归一化 和 转变独热码
    数据不同reshape了
    '''
    train_x = np.array(np.load('train_x.zip_files/train_x.npy'))
    train_y = np.array(np.load('train_y.npy'))
    print(train_y.shape)
    return train_x, train_y


def mini_batch(X, Y, mini_batch_size, seed=0):
    '''
    将数据划分为minib
    也根据数据发生改变了
    '''
    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    batch_num = m // mini_batch_size
    batches = []
    for i in range(batch_num):
        X_batch = shuffled_X[i*mini_batch_size:i*mini_batch_size+mini_batch_size, :, :, :]
        Y_batch = shuffled_Y[i*mini_batch_size:i*mini_batch_size+mini_batch_size, :]
        batch = (X_batch, Y_batch)
        batches.append(batch)
    if m % mini_batch_size != 0:
        X_batch = shuffled_X[batch_num*mini_batch_size:m, :, :, :]
        Y_batch = shuffled_Y[batch_num*mini_batch_size:m, :]
        batch = (X_batch, Y_batch)
        batches.append(batch)
    return batches


if __name__ == '__main__':
    #最后结果训练正确率97%，测试85
    #测试计算图也发生了一点变化，axis defalt0变为1
    train_set_x, train_set_y = load_data()
    y_mean = np.mean(train_set_y)
    y_std = np.mean(train_set_y)
    tf.constant(y_mean, name='y_mean')
    tf.constant(y_std, name='y_std')
    train_set_y = (train_set_y-y_mean) / y_std
    m, n_H0, n_W0, n_C0 = train_set_x.shape
    n_y = train_set_y.shape[1]
    costs=[] #画图用 存放损失函数
    batches = mini_batch(train_set_x, train_set_y, 16)

    X, Y = creat_place_holder(n_H0, n_W0, n_C0, n_y)

    parameter = initial_parameters()

    Z3 = forward_propagation(X, parameter)

    cost = compute_cost(Z3, Y)

    optimer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)

    init = tf.global_variables_initializer()
    num_minibatches = int(train_set_y.shape[0]/16)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1):
            epoch_loss = 0
            for mini_X, mini_Y in batches:
                _, loss = sess.run([optimer, cost], feed_dict={X: mini_X, Y: mini_Y})
                epoch_loss += loss / num_minibatches
            if i % 20 == 0:
                print('epoch', i, ":", epoch_loss)
                costs.append(epoch_loss)
        parameter = sess.run(parameter)
        print(parameter['w1'][1, 1, 1])
        tf.add_to_collection('forward', Z3)
        saver = tf.train.Saver()
        saver.save(sess, './save_model/my_model')
        Z = sess.run(Z3, feed_dict={X:train_set_x[:10]})
        result_z = Z*y_std+y_mean
        print(result_z)
#         correct_pred = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1)) #cnn样本是横着的，m*n所有要在轴1上比较
#         accurcy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
#         print("train accury", sess.run(accurcy, feed_dict={X: train_set_x, Y: train_set_y}))
#         print("test accury", sess.run(accurcy, feed_dict={X: test_set_x, Y: test_set_y}))
    end_time = time.time()
    print(end_time - start_time)
    plt.plot(costs)
    plt.show()
