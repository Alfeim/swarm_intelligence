import tensorflow as tf
import input_data
import model
import numpy as np


class AlexNet:
    def __init__(self,cpkt_file,input_shape):
        self.input_x = tf.placeholder(tf.float32,shape = input_shape,name = "input")
        self.logit = model.build_network(self.input_x,False,True)
        self.input_shape = input_shape
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(cpkt_file)
        self.sess = tf.Session()
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

    def feed(self,inputs):
        inputs = np.reshape(inputs,(self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3]))
        output = self.sess.run(self.logit,feed_dict = {self.input_x: inputs})
        prediction = np.argmax(output,axis=1)
        return prediction	
