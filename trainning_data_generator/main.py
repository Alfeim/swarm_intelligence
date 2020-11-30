import numpy as np
import os
import dijkstra as path_generator
import time
import copy
import random

trainning_unvaliable_value = 255
calculate_unavaliable_value = 8096
max_radium = 4
frequency = 0.3

def generate_etreme_cases(matrix,rows,cols,max_count):
    value = random.random()

    if(value > frequency):
        return matrix

    max_count = random.randint(1,max_count)
    max_number = rows * cols

    for i in range(max_count):
        current_number = random.randint(1,max_number)
        radium = random.randint(1,max_radium)
        x = (current_number - 1) // cols
        y = (current_number - 1) % cols
        matrix[x:min(rows - 1,x + radium - 1),y:min(cols - 1,y + radium - 1)] = trainning_unvaliable_value
    
    return matrix


#generate some matrix as trainning data
def generate_random_matrix(rows,cols,levels,force = True,maxstep = 1):
    matrix_list = []
    milestone = maxstep // 100

    print("\n**********start to generate raw matrix set**********\n")
    for i in range(maxstep):
        current_matrix = np.random.randint(1,levels + 1,(rows,cols))
        current_matrix = generate_etreme_cases(current_matrix,rows,cols,6)
        matrix_list.append(current_matrix)

        if(maxstep < 100):
            print("generation progress: {}%".format((i + 1) *100//maxstep))
        else if((i + 1) %  milestone == 0):
            print("generation progress: {}%".format((i + 1)//milestone))

    matrix_set = np.array(matrix_list)
    dim = matrix_set.shape
    file_name = "matrix_" + str(dim[0]) + "_" + str(dim[1]) + "_" + str(dim[2])
    matrix_set.tofile(file_name)
    return matrix_set


#read saved matrix files and generate matrix which can be used for trainning
def get_matrix_data(file_name = None):
    if(file_name is None):
        files = os.listdir('./')
        for data_file in files:
            if("matrix_" in data_file):
                file_name = data_file
                break
        if(file_name is None):
            print("error:do not find data files!\n")
            return
    
    row_matrix = np.fromfile(file_name,dtype = np.int)
    dim_str = file_name.split("_")
    shape = [int(dim_str[1]),int(dim_str[2]),int(dim_str[3])]
    matrix_set = np.reshape(row_matrix,(shape[0],shape[1],shape[2]))
    return matrix_set


#show the saved matrix files content
def show_generate_matrix():
    files = os.listdir('./')
    matrix_set = None
    dim_str = None

    for data_file in files:
        if("matrix_" in data_file):
            matrix_set = np.fromfile(data_file,dtype = np.int)
            dim_str = data_file.split("_")

    if(matrix_set is None):
        raise Exception("Cannot find generated matrix file")

    shape = [int(dim_str[1]),int(dim_str[2]),int(dim_str[3])]
    matrix_set = np.reshape(matrix_set,(shape[0],shape[1],shape[2]))
    print(matrix_set)
    return


#show the saved path files content
def show_generate_path():
    files = os.listdir('./')
    path_set = None
    dim_str = None

    for data_file in files:
        if("path_" in data_file):
            path_set = np.fromfile(data_file,dtype = np.int)
            dim_str = data_file.split("_")

    if(path_set is None):
        raise Exception("Cannot find generated path file")

    shape = [int(dim_str[1]),int(dim_str[2])]
    path_set = np.reshape(path_set,(shape[0],shape[1]))
    print(path_set)
    return


def clean_old_path():
    files = os.listdir('./')
    for data_file in files:
        if("path_" in data_file):
            os.remove(data_file)

    return


def replace_unavaliable_value(matrix,rows,cols):
    matrix[matrix >= trainning_unvaliable_value] = calculate_unavaliable_value
    return matrix


if __name__ == '__main__':
    generate_random_matrix(100,100,100,False)
    matrix_set = get_matrix_data()
    dim = matrix_set.shape
    max_vertex_number = dim[1] * dim[2]
    path_set = []    
    milestone = dim[0]/100

    time_start = time.time()
    
    print("\n*********start to calculate********** ")
    for i in range(dim[0]):
        current_matrix = replace_unavaliable_value(matrix_set[i],dim[1],dim[2])
        vList = path_generator.relax(current_matrix)
        source_index = path_generator.get_source_number(current_matrix)

        #INDEX i in next_step[] stands for the next step if we want move from index i to source_index(means destination)
        next_step = []
        for j in range(1,max_vertex_number + 1):
            path = path_generator.get_shortest_path(source_index,j,vList)
            if(j == source_index):
                next_step.append(source_index)
            else:
                next_step.append(path[1])

        path_set.append(next_step)
        
        if((i + 1) % milestone == 0):
            print("progress:{}%".format((i + 1)//milestone))
        
        if(dim[0] < 100):
            print("progress:{}%".format((i + 1)*100/dim[0]))



    clean_old_path()

    file_name = "path_" + str(dim[0]) + "_" + str(max_vertex_number)
    path_set = np.array(path_set)
    path_set.tofile(file_name)

    time_end = time.time()
    period = int(time_end - time_start)
    
    hours = period//3600
    minutes = (period - hours*3600)//60
    seconds = period - hours*3600 - minutes*60
    print("totally cost {0}hours,{1}minites,{2}seconds)".format(hours,minutes,seconds))
    #show_generate_matrix()
    #show_generate_path()

