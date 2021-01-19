import model
import os
import glob
import numpy as np
from goto import with_goto
import collections
import argparse
import input_data
import tensorflow as tf
#注意  要把训练 和 验证的 pb模型分开存储（为了后续量化）
#所以我们实际需要的模型存放在VAL_LOGS_DIR下

#变量的默认值
BATCH_SIZE = 10                             #batch的大小
MAX_STEP = 4000                             #总的训练步数
FILEPATH = './train/'                       #存放训练图片的位置
TRAIN_LOGS_DIR = './train_logs/'            #存放训练模型
VAL_LOGS_DIR = './eval_logs/'
WIDTH= 10                                   #图像压缩后的宽
HEIGHT= 10                                  #图像压缩后的高
CHANNELS=2                                  #图像通道数
LEARNING_RATE = 0.0001                      #学习率
ratio = 0.8                                 #训练集与验证集的比率，一般为82开所以设置为0.8
modeltype = 'NOQUANT'
evaltype = 'TEST'
VALPATH = './eval/'
 
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


@with_goto
def train_net():
    #------进入计算图--------
    x_train,y_train,x_val,y_val = input_data.read_img(FILEPATH,CHANNELS,ratio)
    x_train_batch,y_train_batch = input_data.bulid_batch(x_train,y_train,BATCH_SIZE)
    x_val_batch,y_val_batch = input_data.bulid_batch(x_val,y_val,BATCH_SIZE)
    batch_train_len = x_train_batch.shape[0]
    batch_val_len = x_val_batch.shape[0]

   #定义网络    x为输入占位符      y为输出占位符
    #image_max = tf.reduce_max(x_train, name='image_max')
    #image_min = tf.reduce_min(x_train,name='image_min')
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNELS], name='input')
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels_placeholder')
    softmax_linear = model.build_network(x,True,False)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=softmax_linear, labels=y, name='xentropy_per_example')
    train_loss = tf.reduce_mean(cross_entropy, name='loss')

    if modeltype == 'NOQUANT': goto .end
    #fake quant插入到op之前
    tf.contrib.quantize.create_training_graph(input_graph=tf.get_default_graph(), quant_delay=2000) 
    
    label .end

    train_step = trainning(train_loss,LEARNING_RATE)

    #准确略计算
    correct = tf.nn.in_top_k(softmax_linear, y, 1)
    correct = tf.cast(correct, tf.float16)
    train_acc = tf.reduce_mean(correct)

    #------------结束计算图-------------

    with tf.Session() as sess:

        saver = tf.compat.v1.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        valstep = 0

        #max = sess.run(image_max)
        #min = sess.run(image_min)
        #训练
        try:
            ckpt = tf.train.get_checkpoint_state(TRAIN_LOGS_DIR)
            global_step = 0
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
                    
            for i in range(MAX_STEP + 1):
                
                #if_train = True
                pos = i % batch_train_len
                _,acc,loss = sess.run([train_step,train_acc,train_loss],
                         feed_dict={x : x_train_batch[pos], y : y_train_batch[pos]})

                #每50步打印一次准确率和损失函数
                if  i% 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(i, loss, acc*100.0))

                #每200步用验证集的数据进行验证
                if i%200 == 0:
                    #if_train = False    #量化模式下用变量替代占位符.注意 如果要用tflite的话,if_train不要用占位符！
                    vpos = valstep % batch_val_len
                    val_loss, val_acc = sess.run([train_loss, train_acc],
                                                 feed_dict={x : x_val_batch[vpos], y : y_val_batch[vpos]})

                    valstep = valstep+1
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(i, val_loss, val_acc*100.0))

                #每500步保存一次变量值
                if i%500 == 0:
                    checkpoint_path = os.path.join(TRAIN_LOGS_DIR, 'saved_model.ckpt')
                    tmpstep = i + int(global_step)
                    saver.save(sess, checkpoint_path, global_step=tmpstep)
        
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)



@with_goto
def eval():
    maxstep,image_batch,label_batch = input_data.get_eval_image(VALPATH,BATCH_SIZE)
    with tf.Graph().as_default():
        input_x = tf.placeholder(tf.float32,shape = [BATCH_SIZE,WIDTH,HEIGHT,CHANNELS],name = "input")
        logit = model.build_network(input_x,False,True)
        
        if modeltype == 'NOQUANT': goto .next
        print("create eval quantize")
        tf.contrib.quantize.create_eval_graph()
        label .next

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(TRAIN_LOGS_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            acc_count = 0
            max_count = maxstep*BATCH_SIZE
            for i in range(maxstep):
                output = sess.run(logit,feed_dict = {input_x: image_batch[i]})
                prediction = np.argmax(output, axis=1)
                for j in range(BATCH_SIZE):
                    print("predict label: %s    true label : %s"%(prediction[j],label_batch[i][j]))
                    if str(prediction[j]) == str(label_batch[i][j]) : acc_count = acc_count + 1
            print("final accuracy is :{:.2f}%".format(100 * acc_count/max_count))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '-mt','--MODELTYPE', required=False,default= 'NOQUANT',
                      help='choose to quantify or not',type=str)
    parser.add_argument( '-et','--EVALTYPE', required=False,default= 'TEST',
                      help='choose to test or count max_min',type=str)
    parser.add_argument( '-e','--EVAL', required=False,default=0,
                      help='choose to train or eval',type=int)
    parser.add_argument( '-b','--BATCH', required=False,default=12,
                      help='set batch_size',type=int)
    parser.add_argument( '-s','--STEP', required=False,default=4000,
                      help='set maxstep',type=int)
    parser.add_argument( '-i','--IMGPATH', required=False,default='./train/' ,
                      help='set train image path',type=str)
    parser.add_argument( '-v','--VALPATH', required=False,default='./eval/' ,
                      help='set val image path',type=str)
    parser.add_argument( '-tl','--TRAINLOGS', required=False,default='./train_logs/' ,
                      help='set train logs path',type=str) 
    parser.add_argument( '-vl','--VALLOGS', required=False,default='./eval_logs/' ,
                      help='set val logs path',type=str)
    parser.add_argument( '-wd','--WIDTH', required=False,default=64 ,
                      help='set image width',type=int)
    parser.add_argument( '-ht','--HEIGHT', required=False,default=64 ,
                      help='set image height',type=int)
    parser.add_argument( '-ch','--CHANNELS', required=False,default=3 ,
                      help='set image channels',type=int)
    parser.add_argument( '-lr','--LRATE', required=False,default=0.0001 ,
                      help='set learning rate',type=float)
    parser.add_argument( '-rt','--RATIO', required=False,default=0.8 ,
                      help='set image channels',type=float)
    
    args = parser.parse_args()
    
    #获取参数值
    flag = args.EVAL
    BATCH_SIZE = args.BATCH                                    
    MAX_STEP = args.STEP
    VALPATH = args.VALPATH                                   
    FILEPATH = args.IMGPATH                                 
    TRAIN_LOGS_DIR = args.TRAINLOGS     
    VAL_LOGS_DIR = args.VALLOGS
    WIDTH= args.WIDTH                                                 
    HEIGHT= args.HEIGHT                                               
    CHANNELS=args.CHANNELS                                     
    LEARNING_RATE = args.LRATE                   
    ratio = args.RATIO     
    modeltype = args.MODELTYPE
    evaltype  = args.EVALTYPE

    if flag:   
        eval()
    else:
        train_net()
