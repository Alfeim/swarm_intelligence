import threading
import time
import math
import numpy as np
import os
import json
import stradegy
import pathTree
import copy
import random
from loger import Logger



class UAV(threading.Thread):
    """
    @uid: unique id of each UAV
    @position: real space position of current UAV
    @config: config properties
    @max_level: the max level of path planning
    @signal_range_squre: the max commuication distance squre of each UAV
    @one_skip_neighbors: the neigbor of one skip of current UAV
    @two_skip_neighbors: the neibor of two skip of current UAV
    @path_tree: the path tree which will save each path of different levels
    @stradegy_chain: a list to save stradegy of different levels
    @destination: the destination of current UAV
    @maps: a list to save different maps of levels. tips: the order of maps is different to levels,
    if we want to get the map of level x,then it should be self.maps[self.max_level - x][0]
    @last_path_maps: the map corresponding to the time when node update its path
    @move_record: a list to save the path that node has through
    @multi_levels: whether to doing multi level path planing or not
    @addWeight: whether to using Repulsive force of single machine
    @weight_regression: whether to using weight regression or not 
    """
    def __init__(self,uid,config_path,logger):
        super(UAV,self).__init__()       
        assert(os.path.isfile(config_path)), "config file not found."
        self.uid = uid
        self.config = json.loads(open(config_path,"r").read())
        self.pool_step = self.config['pool_step']
        self.max_level = self.config['max_path_tree_level']
        self.signal_range_square = self.config['signal_range_square']
        self.position = np.zeros((2),dtype=np.int32)
        self.one_skip_neighbors = set()
        self.two_skip_neighbors = set()
        self.path_tree = pathTree.PathTree(self.max_level)
        self.stradegy_chain = stradegy.StradegyChain(self.max_level)
        self.destination = 0
        self.maps = []
        self.weight_biases = dict()
        self.time_swap = 0
        self.last_path_maps = None
        self.regression_time = self.config["regression_time"]
        self.regression_value = self.config["regression_value"]

        self.Logger = logger
        self.move_record = []

        self.working = True
        self.multi_levels = True
        self.addWeight = True
        self.weight_regression = True
        self.change_step = self.config['change_step']
        self.openPrint = True

        for i in range(1,self.max_level + 1):
            pool_ratio = pow(self.pool_step,(i - 1))
            width = (int)(self.config['width']//pool_ratio)
            height = (int)(self.config['height']//pool_ratio)
            grid = np.zeros((width,height),dtype=np.uint8)
            time_grid = np.zeros((width,height),dtype=np.uint8)
            self.maps.append([grid,time_grid])
        self.maps.reverse

    """
    To reset destination
    @number: the number of grid of destination
    """
    def reset_destination(self,number):
        self.Logger.log("[notice]uav_%d change_destination from %d to %d"%(self.uid,self.destination,number))
        self.Logger.log("uav {}'s last move record is {}".format(self.uid,self.move_record))
        self.move_record.clear()
        self.working = True
        self.destination = number

    """
    To initialize map of level 0 of current UAV
    """
    def initialize_global_information(self,global_map):
        width = self.config['width']
        height = self.config['height']
        for i in range(width):
            for j in range(height):
                self.maps[0][0][i][j] = global_map[i][j]
                if(self.weight_regression is True):
                    raw_value = self.maps[0][0][i][j]
                    change_value = raw_value*self.change_step
                    final_value = random.randint(0,int(change_value)) - change_value//2 + raw_value
                    final_value = min(255,max(0,final_value))
                    self.maps[0][0][i][j] = final_value

    """
    To update one skip neighbor information
    """
    def update_one_skip_neighbor(self,global_node_list):
        self.one_skip_neighbors.clear()
        for node in global_node_list:
            distance_square = math.pow((self.position[0] - node.position[0]),2) + math.pow((self.position[1] - node.position[1]),2)
            if(node.uid != self.uid and distance_square <= self.signal_range_square):
                self.one_skip_neighbors.add(node)

    """
    To update two skip neighbor information
    """
    def update_two_skip_neighbor(self):
        self.two_skip_neighbors.clear()
        for node in self.one_skip_neighbors:
            for t in node.one_skip_neighbors:
                if(t not in self.one_skip_neighbors):
                    self.two_skip_neighbors.add(t)

    """
    To get grid position by space position
    @map_level: the level of current map
    """
    def get_src_grid_position(self,map_level =  1):
        cols = (int)(self.config['height'])
        grid_size = (int)(self.config['grid_size'])
        x = self.position[0]//grid_size
        y = self.position[1]//grid_size
        step = pow(self.pool_step,self.max_level - map_level)
        _x = x // step
        _y = y // step
        cols //= step
        return _x,_y,_x*cols + _y + 1
    
    """
    To get grid position of destination
    @map_level: the level of current map
    """
    def get_dst_grid_position(self,map_level=1):
        cols = (int)(self.config['height'])
        x = (self.destination - 1)//cols
        y = (self.destination - 1)%cols

        step = pow(self.pool_step,self.max_level - map_level)
        _x = x // step
        _y = y // step
        cols //= step
        return _x,_y,_x*cols + _y + 1

    """
    To update current grid's information
    """
    def update_local_environment_information(self,global_map,global_time):
        grid_x,grid_y,_ = self.get_src_grid_position(self.max_level)     
        basic_grid = self.maps[0][0]
        basic_time_grid = self.maps[0][1]
        basic_grid[grid_x][grid_y] = global_map[grid_x][grid_y]
        basic_time_grid[grid_x][grid_y] = global_time
        self.time_swap = global_time

        bias_live_time = self.config['bias_live_time']
        for subdict in self.weight_biases.values():
            remove_list = []
            for k,v in subdict.items():
                last_time = v
                if(self.time_swap - bias_live_time >= last_time):
                    remove_list.append(k)
            for k in remove_list:
                subdict.pop(k)


    """
    To transmit information of current UAV to others
    """
    def update_environment_information(self):
        def calculate_current_regression_value(current_value,regression_final_value,alpha,delta_time):
            N = alpha * delta_time
            ret = current_value + (regression_final_value - current_value)*float(math.e**N-1)/(math.e**N+1)
            return int(ret)


        def update_matrix(other_node):
            width = self.config['width']
            height = self.config['height']
            basic_grid = self.maps[0][0]
            basic_time_grid = self.maps[0][1]
            for i in range(width):
                for j in range(height):
                    if(basic_time_grid[i][j] > other_node.maps[0][1][i][j]):
                        other_node.maps[0][0][i][j] = basic_grid[i][j]
                        other_node.maps[0][1][i][j] = basic_time_grid[i][j]
        
        if(self.weight_regression is True):
            for i in range(int(self.config['width'])):
                for j in range(int(self.config['height'])):
                    alpha = math.log(2)/self.regression_time
                    delta_time = self.time_swap - self.maps[0][1][i][j]
                    self.maps[0][0][i][j] = calculate_current_regression_value(self.maps[0][0][i][j],self.regression_value,alpha,delta_time)


        for node in self.one_skip_neighbors:
            update_matrix(node)
            for k,subdict in self.weight_biases.items():
                if(k not in node.weight_biases):
                    node.weight_biases[k] = dict()
                for number,time in subdict.items():
                    if(number not in node.weight_biases[k] or time > node.weight_biases[k][number]):
                        node.weight_biases[k][number] = time
    
    """
    To update higher level map by map of level 0
    """
    def update_all_level_maps(self,maxpool=False):
        def get_matrix_sum(matrix,row,col):
            ret = 0
            for i in range(row):
                for j in range(col):
                    ret += matrix[i][j]
            
            return ret
        
        def get_matrix_max(matrix,row,col):
            ret = 0
            for i in range(row):
                for j in range(col):
                   ret = max(ret,matrix[i][j])
            return ret 

        width = self.config['width']
        height = self.config['height']
        for i in range(1,self.max_level):
            last_grid = self.maps[i-1][0]
            last_time_grid = self.maps[i-1][1]
            current_grid = self.maps[i][0]
            current_time_grid = self.maps[i][1]
            width = width//self.pool_step
            height = height//self.pool_step
            for k in range(width):
                for m in range(height):
                    matrix = last_grid[k:k+self.pool_step,m:m+self.pool_step]
                    time_matrix = last_time_grid[k:k+self.pool_step,m:m+self.pool_step]
                    avg = get_matrix_sum(matrix,self.pool_step,self.pool_step)//(self.pool_step*self.pool_step)
                    current_grid[k][m] = get_matrix_max(matrix,self.pool_step,self.pool_step) if maxpool is True else avg
                    current_time_grid[k][m] = get_matrix_max(time_matrix,self.pool_step,self.pool_step)


    """
    UAV will move by path list
    """
    def move(self):
        if(self.working is False):
            return
        
        grid_size = (int)(self.config['grid_size'])
        speed = (int)(self.config['speed'])
        cols = (int)(self.config['height'])

        _,_,current_grid_number = self.get_src_grid_position(self.max_level)
        current_grid_x = self.position[0]
        current_grid_y = self.position[1]

        if(len(self.move_record) == 0):
            self.move_record.append([current_grid_x,current_grid_y,current_grid_number])
        
        next_grid_number = self.path_tree.getNodeNumber(self.max_level,0)
        if(next_grid_number == current_grid_number):
            next_grid_number = self.path_tree.getNodeNumber(self.max_level,1)
        
        if(self.openPrint):
            print("current_grid_number %d, next grid number:%d"%(current_grid_number,next_grid_number))
        next_grid_center_x = ((next_grid_number - 1)//cols)*grid_size + grid_size//2
        next_grid_center_y = ((next_grid_number - 1)%cols)*grid_size + grid_size//2
        
        if(self.openPrint):
            print("node %d move from (%d,%d) to (%d,%d)"%(self.uid,current_grid_x,current_grid_y,next_grid_center_x,next_grid_center_y))
        x_offset = 0
        y_offset = 0
        
        if(next_grid_center_x == current_grid_x):
            y_offset = min(next_grid_center_y - current_grid_y,speed)
        elif(next_grid_center_y == current_grid_y):
            x_offset = min(next_grid_center_x - current_grid_x,speed)
        else:
            distance_square = pow(current_grid_x-next_grid_center_x,2) + pow(current_grid_y-next_grid_center_y,2)
            speed = min(speed,math.sqrt(int(distance_square)))
            alpha = math.atan(abs(current_grid_y - next_grid_center_y)/abs(current_grid_x - next_grid_center_x))
            x_offset = int(speed*math.cos(alpha))
            y_offset = int(speed*math.sin(alpha))
            if(next_grid_center_x < current_grid_x):
                x_offset *= -1
            if(next_grid_center_y < current_grid_y):
                y_offset *= -1


        self.position[0] += x_offset
        self.position[1] += y_offset

        _,_,new_grid_number= self.get_src_grid_position(self.max_level)
       
        self.move_record.append([self.position[0],self.position[1],new_grid_number])

        if(current_grid_number != new_grid_number):
            self.path_tree.removeNode(self.max_level,0)
            if(self.uid not in self.weight_biases):
                self.weight_biases[self.uid] = dict()
            self.weight_biases[self.uid][new_grid_number] = self.time_swap

        if(new_grid_number == self.destination):
            self.working = False
            self.Logger.log("node %d go to destination"%self.uid)
            self.Logger.log("node {}'s move path is {}".format(self.uid,self.move_record))
    
    """
    To determin whether doing path planing or not
    """
    def stradegy_switch(self):
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
    
        if(self.last_path_maps is None):
            self.last_path_maps = self.maps[0][0]
            return True

        rows = self.config['width']
        cols = self.config['height']
        path = self.path_tree.getPath(self.max_level)
        path_around_sum = 0
        used = set()
        diff_matrix = np.zeros(rows*cols*2,dtype = np.uint8)
        

        for number in path:
            diff_matrix[2*number - 1] = 255
            path_around_sum += get_path_around_sum(number,self.last_path_maps,rows,cols,used,1)

        path_avg_level = path_around_sum//len(used)
        set_diff_matrix(self.last_path_maps,self.maps[0][0],diff_matrix,rows,cols,used,path_avg_level)
        diff_matrix = np.reshape(diff_matrix,(rows,cols,2))

        #here feed diff_matrix to the deep learning model to get result
        #if result is 0 then return False else return True
        return True


    """
    To transform global matrix's grid position to local grid position
    """
    def global_to_local(self,current_x,current_y,last_grid_x,last_grid_y):
        x_offset = last_grid_x * self.pool_step
        y_offset = last_grid_y * self.pool_step
        x_local = current_x - x_offset
        y_local = current_y - y_offset
        return x_local,y_local,x_local*y_local+y_local+1
    
    """
    To transform local matrix's grid position to global grid position
    """
    def local_to_global(self,current_x,current_y,last_grid_x,last_grid_y):
        x_offset = last_grid_x * self.pool_step
        y_offset = last_grid_y * self.pool_step
        x_global = current_x + x_offset
        y_global = current_y + y_offset
        return x_global,y_global,x_global*y_global+y_global+1


    """
    @src: source number in current level's map
    @dst: destination number in current levels's map
    @level: current level
    """
    def convergence(self,src,dst,level,matrix_added_weight = None):
        input_matrix = matrix_added_weight if (level == self.max_level) else self.maps[self.max_level-level][0]
        
        if(level == 1 or self.multi_levels is False):
            #if current level is 1,there would not be father grids
            djikstrator = stradegy.Dijkstra(input_matrix,src)
            path = djikstrator.execute(src,dst,self.uid)
            for path_number in path:
                self.path_tree.addNode(level,pathTree.PathTreeNode(path_number))

            return

        """
        get relative position (local position) in the sub grid
        """
        def get_relative_position(current_level_postion,last_level_position):
            x = current_level_postion[0]
            y = current_level_postion[1]
            x_ = last_level_position[0]
            y_ = last_level_position[1]

            relative_x = x - x_*pool_step
            relative_y = y - y_*pool_step

            return [relative_x,relative_y]

        """
        get destination position of current grid by the relationship of current path node and next path node
        of father map 
        """
        def get_dst_position(current_number,next_number,cols):
            if(next_number == current_number - cols):
                return [0,(self.pool_step + 1)//2]
            elif(next_number == current_number + cols):
                return [self.pool_step - 1,(self.pool_step + 1)//2]
            elif(next_number == current_number - 1):
                return [(self.pool_step + 1)//2,0]
            elif(next_number == current_number + 1):
                return [(self.pool_step + 1)//2,self.pool_step - 1]
        
        """
        get source position of current grid by the relationship of current path node and next path node
        of father map 
        """
        def get_src_position(current_number,last_number,cols):
            return get_dst_position(current_number,last_number,cols)

        """
        get real path number of current level by the grid position of father and current local number
        """
        def get_real_position_of_current_level(local_number,father_position):
            father_x = father_position[0]
            father_y = father_position[1]
            real_x = father_x*pool_step + (local_number - 1)//self.pool_step
            real_y = father_y*pool_step + (local_number - 1)%self.pool_step
            return [real_x,real_y]

        pool_step = self.config['pool_step']
        
        current_level_shape = self.maps[self.max_level - level][0].shape
        current_level_cols = current_level_shape[1]
        current_src_position_real = [(src-1)//current_level_cols,(src-1)%current_level_cols]
        current_dst_position_real = [(dst-1)//current_level_cols,(dst-1)%current_level_cols]
        
        last_level_shape = self.maps[self.max_level - level + 1][0].shape
        last_level_cols = last_level_shape[1]
        last_level_path = self.path_tree.getPath(level - 1)
        
        length = len(last_level_path)

        #enumerate each path number in the path of last level(path number is not local)

        for i in range(length):
            path_number = last_level_path[i]
            path_position = [(path_number - 1)//last_level_cols,(path_number - 1)%last_level_cols]

            x_start = path_position[0]*pool_step
            y_start = path_position[1]*pool_step
            x_end = x_start + pool_step
            y_end = y_start + pool_step
            
            local_matrix = copy.deepcopy((input_matrix)[x_start:x_end,y_start:y_end])

            relative_src_position = get_relative_position(current_src_position_real,path_position) if i == 0 else get_src_position(path_number,last_level_path[i-1],last_level_cols) 
            relative_dst_position = get_relative_position(current_dst_position_real,path_position) if i == length - 1 else get_dst_position(path_number,last_level_path[i+1],last_level_cols)

            #local grid is a pool_step*pool_step matrix
            relative_src_number = relative_src_position[0]*pool_step + relative_src_position[1] + 1 
            relative_dst_number = relative_dst_position[0]*pool_step + relative_dst_position[1] + 1

            djikstrator = stradegy.Dijkstra(local_matrix,relative_src_number)
            local_path = djikstrator.execute(relative_src_number,relative_dst_number,self.uid)

            for local_path_number in local_path:
                real_position = get_real_position_of_current_level(local_path_number,path_position)
                real_number = real_position[0]*current_level_cols + real_position[1] + 1
                self.path_tree.addNode(level,pathTree.PathTreeNode(real_number))
            
        
        return



    """
    To doing path planing
    """
    def path_planning(self):
        if not(self.stradegy_switch()):
            return

        #if it is necessary to doing path planning, then the path tree should be reset   
        self.Logger.log("[notice]uav_%d is doing path planing"%self.uid)
        self.path_tree.reset(self.max_level)
        level = 1
        while(level <= self.max_level):
            _,_,src = self.get_src_grid_position(level)
            _,_,dst = self.get_dst_grid_position(level)

            matrix = None if level != self.max_level else copy.deepcopy(self.maps[0][0])
            
            if(level == self.max_level):
                if(self.addWeight is True):
                    for subdict in self.weight_biases.values():
                        for k,v in subdict.items():
                            x = (k - 1)//self.config['height']
                            y = (k - 1)%self.config['height']
                            bias_time_swap = v
                            weight = int(self.config['bias_max']) * (int(self.config['bias_live_time']) - (self.time_swap - bias_time_swap))
                            matrix[x][y] = min(int(weight) + matrix[x][y],255)


            if(self.multi_levels is False and level != self.max_level):
                level += 1
                continue
            
            self.convergence(src,dst,level,matrix)
            level += 1
           

        self.last_path_maps = self.maps[0][0]

        return













