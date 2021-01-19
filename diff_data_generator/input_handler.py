import numpy as np

input_data_path = './matrix_5000_5_5_2'
input_label_path = './labels'


def generate_data_set(matrix_set_path,label_set_path):
    raw_matrix = np.fromfile(matrix_set_path,dtype = np.uint8)
    matrix_dim_str = matrix_set_path.split("_")
    matrix_shape = [int(matrix_dim_str[1]),int(matrix_dim_str[2]),int(matrix_dim_str[3]),int(matrix_dim_str[4])]
    matrix_set = np.reshape(raw_matrix,(matrix_shape[0],matrix_shape[1],matrix_shape[2],matrix_shape[3]))

    raw_label = np.fromfile(label_set_path,dtype = np.uint8)
    path_set = np.reshape(raw_label,matrix_shape[0])

    assert(matrix_shape[0] == len(path_set)),"label and matrix are not matched"

    return matrix_set,path_set

            
if __name__ == "__main__":
    m,l = generate_data_set(input_data_path,input_label_path)

            
        




    












