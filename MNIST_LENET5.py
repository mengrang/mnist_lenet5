#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
#配置神经网络参数
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
BATCH_SIZE=100
IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10
#卷积层深宽
CONV1_DEEP=32
CONV1_SIZE=5
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层
FC_SIZE=512


#卷积神经网络的前向传播。新参数train用于区别训练过程和测试过程。将用dropout，只在训练时使用
def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
#池化层
    with tf.name_scope('layer2-pool1'):
         pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable(
            "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        #过滤器
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #将layer4pool输出化为layer5fc输入。lay4输出为（7，7，64）矩阵，而第五层全连接层需要输入格式为向量，所以在这里需要将这个矩阵拉直盛一个向量。
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable("weight",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:fc1=tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable(
            "weight",[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
    return   logit
####################################################################################################################################################################################
#########################################################################################################################

import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import MNIST_LENET5
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="MNIST"
MODEL_NAME="model.ckpt"
def train(mnist):
    # 调整输入数据placeholder的格式，输入为一个四维矩阵。
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    MNIST_LENET5. IMAGE_SIZE,
                                    MNIST_LENET5.IMAGE_SIZE,
                                    MNIST_LENET5.NUM_CHANNELS], name='x-input')
    reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                  MNIST_LENET5.IMAGE_SIZE,
                                  MNIST_LENET5.IMAGE_SIZE,
                                  MNIST_LENET5.NUM_CHANNELS))
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=MNIST_LENET5(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAvrage(
        MOVING_AVERAGE_DECAY,global_step)
    variables_average_op=variable_average.apply(tf.trainable_variables())
    cross_entropy=tf.nn.aparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
        LEARNING-RATE_BASE,
        global_step,
        mninst.train.num_xamples/BATCH_SIZE,
        LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
        #初始化TF持久类
        saver=tf.train.Saver()
        with tf.Session as sess:
            tf.initialize_all_variables().run()
            for i in range(TRAINING_STEPS):
               xs,ys=mnist.train.next_batch(BATCH_SIZE)
               [_, loss_value, step] =sess.run([train_op, loss, global_step], feed_dict={x:xs, y:ys})
        #每1000轮保存一次模型
               if i %1000==0:
                  print("After %d training step(s),loss on  training""batch is %g."%(step,loss_value))
                  saver.save(
                  sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step
            )
def main(argv=None):
            mnist = input_data.read_data_sets('MNIST', one_hot=True)
            train(mnist)

if __name__=='_main_':
   tf.app.run()
################################
##################################
###################################
config=tf.ConfigProto(allow_soft_placement=True)
from tensorflow.examples.tutorials.mnist import input_data
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
import MNIST_LENET5
import mnist_train
#每10s加载一次模型，在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS=10
def evaluate(mnist):
    with tf.Graph().as_default()as g:
        x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                        MNIST_LENET5.IMAGE_SIZE,
                                        MNIST_LENET5.IMAGE_SIZE,
                                        MNIST_LENET5.NUM_CHANNELS], name='x-input')
        #reshaped_xs = np.reshape(xs, (BATCH_SIZE,
        #                              MNIST_LENET5.IMAGE_SIZE,
        #                              MNIST_LENET5.IMAGE_SIZE,
         #                             MNIST_LENET5.NUM_CHANNELS))
        y_=tf.placeholder(
            tf.float32,[None,MNIST_LENET5.OUTPUT_NODE],name='y-input'
        )
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        y=MNIST_LENET5.inference(x,None,None)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path\
                    .split('/')[-1].split('/')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training step(s),validation""accuracy= %g "%(global_step,accyracy_score))
                else:
                    print('No checkpoint file found')
                    return
                    time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist=input_data.read_data_sets("MNIST",one_hot=True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()