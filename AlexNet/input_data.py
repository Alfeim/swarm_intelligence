import os
import glob
import numpy as np

def read_img(path,CHANNELS,ratio):
    matrix_set_path = path + 'matrix_25000_10_10_2'
    label_set_path = path + 'labels'
    
    raw_matrix = np.fromfile(matrix_set_path,dtype = np.uint8)
    matrix_dim_str = matrix_set_path.split("_")
    matrix_shape = [int(matrix_dim_str[1]),int(matrix_dim_str[2]),int(matrix_dim_str[3]),int(matrix_dim_str[4])]
    matrix_set = np.reshape(raw_matrix,(matrix_shape[0],matrix_shape[1],matrix_shape[2],matrix_shape[3]))

    raw_label = np.fromfile(label_set_path,dtype = np.uint8)
    path_set = np.reshape(raw_label,matrix_shape[0])

    #change type
    print("change type")
    imgs = np.asarray(matrix_set,np.float32)
    labels = np.asarray(path_set,np.int32)
    num_example = imgs.shape[0]
    s = np.int(num_example * ratio)

    #打乱乱序
    print("shuffle!")
    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    np.random.shuffle(labels)
   

    #x_train 训练集图片  y_train 训练集标签
    #x_val 验证集图片  y_al 验证集标签
    x_train = imgs[:s]
    y_train = labels[:s]
    x_val   = imgs[s:]
    y_val   = labels[s:]

    return x_train,y_train,x_val,y_val

def bulid_batch(image, label, batch_size):
    print("build batch")
    #生成batch 多幅图片为一个batch
    image_batch = []
    label_batch = []
    
    border = image.shape[0]
    if border % batch_size != 0:
        border = border - border%batch_size

    maxstep = border//batch_size
    for i in range(maxstep):
        start = i * batch_size
        end = start + batch_size
        label_batch.append(label[start:end])
        image_batch.append(image[start:end])

    return np.asarray(image_batch),np.asarray(label_batch)

def get_eval_image(path,batch_size):
    matrix_set_path = path + 'matrix_2500_10_10_2'
    label_set_path = path + 'labels'
    
    raw_matrix = np.fromfile(matrix_set_path,dtype = np.uint8)
    matrix_dim_str = matrix_set_path.split("_")
    matrix_shape = [int(matrix_dim_str[1]),int(matrix_dim_str[2]),int(matrix_dim_str[3]),int(matrix_dim_str[4])]
    matrix_set = np.reshape(raw_matrix,(matrix_shape[0],matrix_shape[1],matrix_shape[2],matrix_shape[3]))

    raw_label = np.fromfile(label_set_path,dtype = np.uint8)
    path_set = np.reshape(raw_label,matrix_shape[0])

    #change type
    print("change type")
    imgs = np.asarray(matrix_set,np.float32)
    labels = np.asarray(path_set,np.int32)
    imgs_batch,labels_batch = bulid_batch(imgs,labels,10)
    
    border = matrix_shape[0]
    if border % batch_size != 0:
        border = border - border%batch_size

    maxstep = border//batch_size
    return maxstep,imgs_batch,labels_batch
