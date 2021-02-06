import tensorflow as tf
import numpy as np

#定义网络一些层的接口函数
def weight_variable(shape, stddev,name="weights"):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(val,shape, name="biases"):
    initial = tf.constant(value=val, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(input, w,strides):
    return tf.nn.conv2d(input, w, strides = strides, padding='SAME')

def pool_max(input,ksize,strides,padding,name):
    return tf.nn.max_pool(input,
                               ksize=ksize,
                               strides=strides,
                               padding=padding,
                               name=name)

def bn_layer(x,is_training):
    return tf.layers.batch_normalization(x ,training=is_training)

def fc(input, w, b):
    return tf.matmul(input, w) + b

def count_handler_min(src,dst):
    tmpmin = tf.reduce_min(src)
    tmpmin = tf.cast(tmpmin,dtype=tf.float32)
    tmpmin = tf.reshape(tmpmin,shape=[-1])
    return tf.concat([dst,tmpmin],0)

def count_handler_max(src,dst):
    tmpmax = tf.reduce_max(src)
    tmpmax = tf.cast(tmpmax,dtype=tf.float32)
    tmpmax=tf.reshape(tmpmax,shape=[-1])
    return tf.concat([dst,tmpmax],0)

def build_network(input,IF_TRAIN=True,IF_COUNT_MAX_AND_MIN=False):
    datamin = tf.constant([0.0],dtype = tf.float32)
    datamax = tf.constant([0.0],dtype = tf.float32)
     # 卷积层1        64个1x1的卷积核（1通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('layer1_conv'):
        kernel = weight_variable([1,1,2,64],1.0)
        biases = bias_variable(0.1,[64])
        tmpconv =  conv2d(input, kernel,[1,1,1,1])+ biases
        bn1 = bn_layer(tmpconv,IF_TRAIN)
        conv1 = tf.nn.relu6(bn1, name='conv2d_out')

    # 池化层1       3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
    # 如果使用tensorlfow 的python API，即convert方式对post training model 进行量化，目前由于不支持LRN操作，建议先把norm1部分注释
    with tf.variable_scope('layer_pool_lrn'):
        pool1 = pool_max(conv1,[1,3,3,1],[1,2,2,1],'SAME','pooling')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_out')


    # 卷积层2        16个3x3的卷积核（64通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('layer2_conv'):
        kernel = weight_variable([3, 3, 64,16],0.1)
        biases = bias_variable(0.1,[16])
        tmpconv = conv2d(norm1, kernel,[1,1,1,1]) + biases
        bn2 = bn_layer(tmpconv,IF_TRAIN)
        conv2 = tf.nn.relu6(bn2, name='conv2d_out')

    # 池化层2       3x3最大池化，步长strides为2，池化后执行lrn()操作
    # 如果使用tensorlfow 的python API，即convert方式对post training model 进行量化，目前由于不支持LRN操作，建议先把norm2部分注释
    with tf.variable_scope('layer2_pool_lrn'):
        pool2 = pool_max(conv2,[1,3,3,1],[1,1,1,1],'SAME','pooling')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

    # 全连接层3
    with tf.variable_scope('layer3_fullyconnect'):
        shape = int(np.prod(norm2.get_shape()[1:]))
        kernel = weight_variable([shape,128],0.005)
        biases = bias_variable(0.1,[128])
        flat = tf.reshape(pool2,[-1,shape])
        fc3 = tf.nn.relu6(fc(flat,kernel,biases),name = 'fullyconnect_out')

    # 全连接层4
    with tf.variable_scope('layer4_fullyconnect'):
        kernel = weight_variable([128,128],0.005)
        biases = bias_variable(0.1,[128])
        fc4 = tf.nn.relu6(fc(fc3,kernel,biases),name = 'fullyconnect_out')
      
    with tf.variable_scope('layer5_fullyconnect'):
        kernel = weight_variable([128,2],0.005)
        biases = bias_variable(0.1,[2])

    softmax_linear = tf.add(tf.matmul(fc4, kernel), biases, name='output')
    #计算损失函数及反向传播

    return softmax_linear
