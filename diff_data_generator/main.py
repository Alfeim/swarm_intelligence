import numpy as np
import os
import dijkstra as path_generator
import time
import copy
import random
import math

diff_threshold = 0.2

def generate_random_matrix(rows,cols,force = True,maxstep = 10):
    matrix_list = []
    print("\n**********start to generate raw matrix set**********\n")
    for i in range(maxstep):
        current_matrix = np.random.randn(rows,cols)
        current_matrix = (np.multiply(current_matrix,255.0/(2.58*2)) + 127.5).astype(np.uint8)
        #current_matrix = generate_etreme_cases(current_matrix,rows,cols,6)
        matrix_list.append(current_matrix)

    matrix_set = np.array(matrix_list)
    dim = matrix_set.shape
    file_name = "matrix_" + str(dim[0]) + "_" + str(dim[1]) + "_" + str(dim[2])
    matrix_set.tofile(file_name)
    print("generate raw matrix success!")
    return file_name,matrix_set


def calculate_diff_rate(path1,path2,matrix_1,matrix_2,rols,cols):
    sum_path_1 = 0
    sum_path_2 = 0
    for number in path1:
        x = (number - 1)//cols
        y = (number - 1)%cols
        sum_path_1 += matrix_1[x][y]

    for number in path2:
        x = (number - 1)//cols
        y = (number - 1)%cols
        sum_path_2 += matrix_2[x][y]
    
    diff_rate = (float)(abs(sum_path_2 - sum_path_1))/(sum_path_1 + 1.0)
    return diff_rate


def add_salt(input_matrix,rows,cols,max_salt_count):
    new_matrix = copy.deepcopy(input_matrix)
    salt_count = random.randint(0,max_salt_count)
    element_count = rows * cols
    salt_added_list = []
    for i in range(element_count):
        if(len(salt_added_list) < max_salt_count):
            salt_added_list.append(i)
        else:
            exchange_number = random.randint(0,max_salt_count-1)
            salt_added_list[exchange_number] = i

    for i in salt_added_list:
        x = i//cols
        y = i%cols
        salt = int(np.random.randn()*255.0/(2.58*2) + 127.5)
        new_matrix[x][y] = salt
        
    return new_matrix


def generate_diff_matrix(raw_matrix,new_matrix,path,rows,cols):
    def get_path_around_sum(number,matrix,rows,cols,used,ttf):
        if(ttf < 0): return 0
        
        x = (number - 1)//cols
        y = (number - 1)%cols
        
        if(x < 0 or x >= rows or y < 0 or y >= cols):return 0

        ret = 0
        ret += matrix[x][y] if number not in used else 0
        used.add(number)

        if(x - 1 >= 0):
            ret += get_path_around_sum(number - 1,matrix,rows,cols,used,ttf-1)
        if(x + 1 < rows):
            ret += get_path_around_sum(number + 1,matrix,rows,cols,used,ttf-1)
        if(y - 1 >= 0):
            ret += get_path_around_sum(number - cols,matrix,rows,cols,used,ttf-1)
        if(y + 1 < cols):
            ret += get_path_around_sum(number + cols,matrix,rows,cols,used,ttf-1)
        
        return ret
   
    def set_diff_matrix(raw_matrix,new_matrix,diff_matrix,rows,cols,used,path_avg_level):
        for i in range(rows):
            for j in range(cols):
                number = i*cols + (j + 1)
                if(number in used):
                    diff_level = abs(int(raw_matrix[i][j]) - int(new_matrix[i][j]))
                    diff_level = int((float(diff_level)/path_avg_level)*255)
                    diff_matrix[(number - 1)*2] = diff_level
                
    
    used = set()
    path_around_sum = 0
    diff_matrix = np.zeros(rows*cols*2,dtype = np.uint8)
    
    if(len(path) == 0):
        return np.reshape(diff_matrix,(rows,cols,2))

    for number in path:
        diff_matrix[2*number - 1] = 255
        path_around_sum += get_path_around_sum(number,raw_matrix,rows,cols,used,1)

    path_avg_level = path_around_sum//len(used)

    set_diff_matrix(raw_matrix,new_matrix,diff_matrix,rows,cols,used,path_avg_level)
    diff_matrix = np.reshape(diff_matrix,(rows,cols,2))
    
    return diff_matrix


def generate_diff_set(file_name):
    raw_matrix = np.fromfile(file_name,dtype = np.uint8)
    dim_str = file_name.split("_")
    shape = [int(dim_str[1]),int(dim_str[2]),int(dim_str[3])]
    matrix_set = np.reshape(raw_matrix,(shape[0],shape[1],shape[2]))
    batch_size = 25
    matrix_max_number = shape[1]*shape[2]
    diff_matrix_set = []
    raw_maxtrix_set = []
    diff_matrix_label = []

    print("generate diff set")
    for i in range(shape[0]):
        if((i + 1) % 10 == 0):
            print("process {}%".format((i+1)/shape[0]*100))
        current_matrix = matrix_set[i]
        for j in range(batch_size):
            new_matrix = add_salt(current_matrix,shape[1],shape[2],matrix_max_number//3)
            k = 0
            while k < 10:
                src_index = random.randint(1,matrix_max_number)
                dst_index = random.randint(1,matrix_max_number)
                if(src_index == dst_index):
                    continue
                vList = path_generator.relax(current_matrix,src_index)
                path = path_generator.get_shortest_path(src_index,dst_index,vList)
                path = path[1:-1]

                new_vList = path_generator.relax(new_matrix,src_index)
                new_path = path_generator.get_shortest_path(src_index,dst_index,new_vList)
                new_path = new_path[1:-1]

                diff_rate = calculate_diff_rate(path,new_path,current_matrix,new_matrix,shape[1],shape[2])
                
                diff_matrix = generate_diff_matrix(current_matrix,new_matrix,path,shape[1],shape[2])
                diff_matrix_set.append(diff_matrix)
                if(diff_rate >= diff_threshold):
                    diff_matrix_label.append(1)
                else:
                    diff_matrix_label.append(0)
                k += 1


    filename_1 = "matrix_" + str(shape[0]*batch_size*10) + "_" + str(shape[1]) + "_" + str(shape[2]) + "_2"
    filename_2 = "labels"
    file1 = np.asarray(diff_matrix_set,dtype = np.uint8)
    file2 = np.asarray(diff_matrix_label,dtype = np.uint8)
    np.asarray(diff_matrix_set,dtype = np.uint8).tofile(filename_1)
    np.asarray(diff_matrix_label,dtype = np.uint8).tofile(filename_2)
    print("complete!")
    return diff_matrix_set,diff_matrix_label


filename,matrix_set = generate_random_matrix(10,10)
generate_diff_set(filename)