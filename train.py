# -*- coding: utf-8 -*-
import tensorflow as tf
import MathData
import numpy as np
import symclass

CLASS_NUM = len(symclass.sym_classes)



class DataSet(object):
    def __init__(self):
        self._epochs_completed = 0
        images, labels = MathData.load_data('./data',28,CLASS_NUM)
        print("loading complete")
        self._images = np.array(images)
        self._labels = np.array(labels)
        self._num_samples = len(self._images)
        self._index_in_epochs = self._num_samples+1
    def next_batch(self, batch_size):
        start = self._index_in_epochs
        self._index_in_epochs+=batch_size
        if self._index_in_epochs > self._num_samples:
            perm = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epochs = batch_size
        end = self._index_in_epochs
        return self._images[start:end],self._labels[start:end]

data_sets = DataSet()
testimg,testlabel = data_sets.next_batch(50)
print(testimg[0])
print(testlabel[0])
print(type(testimg[0][0]))
print(type(testlabel[0][0]))
# x = tf.placeholder("float", [None, 784])
x = tf.placeholder(tf.float32, [None, 784])

# y_ = tf.placeholder("float", [None,10])
y_ = tf.placeholder(tf.float32, [None, CLASS_NUM])

# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

W_conv1 = weight_variable([5,5,1,32]);
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
W_fc2 = weight_variable([1024,CLASS_NUM])
b_fc2 = bias_variable([CLASS_NUM])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

for i in range(20000):
  images,labels = data_sets.next_batch(50)
  #print images.shape
  #print labels.shape
  #]print(i)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:images, y_: labels, keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
    saver.save(sess, 'test_model',global_step=i+1)
  train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5})

