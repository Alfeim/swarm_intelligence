import sys
sys.path.append('./AlexNet/')
import UAV
from UAV import UAV
import time
import random
import math
import os
import json
from loger import Logger
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import AlexNet.deep_learning as deep_learning
from deep_learning import AlexNet

SRC_NUMBER = 1
DST_NUMBER = 1
UAV_COUNTS = 1
UAV_BATCH = 1

CONFIG_PATH = "./config.json"
logger = Logger('log_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
CONFIG = json.loads(open(CONFIG_PATH,"r").read())
destination = CONFIG['width']*CONFIG['height']
turns_of_change = 1
rate_of_change = 0.1
logger.openLogger = False
MOVE_RECORD = dict()
MODEL = AlexNet('./AlexNet/train_logs',[1,CONFIG['width'],CONFIG['height'],2])
def run():
    global turns_of_change
    global rate_of_change
    global destination
    
    time_swap = 1
    MAP_RECORD = []
    UAV_SET = []
    MOVE_RECORD.clear()

    global_map = (np.multiply(np.random.randn(CONFIG['width'],CONFIG['height']),255.0/(2.58*2)) + 127.5).astype(np.uint8)
    observer_uav = UAV(-1,CONFIG_PATH,logger)
    observer_uav.position[0] = 0
    observer_uav.position[1] = 0
    observer_uav.reset_destination(destination)
    observer_uav.openPrint = False
    observer_uav.weight_regression = False
    observer_uav.use_deep_learning = False
    MAP_RECORD.append(copy.deepcopy(global_map))

    #initialization
    for i in range(UAV_COUNTS):
        current_uav = UAV(i+1,CONFIG_PATH,logger)
        current_uav.position[0] = 0
        current_uav.position[1] = 0
        current_uav.initialize_global_information(global_map)
        current_uav.reset_destination(destination)
        current_uav.use_deep_learning = True
        current_uav.openPrint = False
        UAV_SET.append(current_uav)
        
    time_cost_raw = 0
    time_cost_deep_learning = 0
    while(len(UAV_SET)>0 or observer_uav.working is True):        
        #print("**time_swap: %d**"%time_swap)
        change_flag = (random.random() <= 1/turns_of_change)
        if(change_flag is True):
            #print("[Notice]global map info is changed")
            element_count = CONFIG['width']*CONFIG['height']
            change_count = int(element_count*rate_of_change)
            for i in range(change_count):
                number = random.randint(1,element_count)
                x = (number - 1)//int(CONFIG['height'])
                y = (number - 1)%int(CONFIG['height'])
                global_map[x][y] = random.randint(0,255)
                   
        MAP_RECORD.append(copy.deepcopy(global_map))
        
        for uav in UAV_SET:
            uav.initialize_global_information(global_map)
            uav.path_planning(MODEL)
            uav.move()
    
        for uav in UAV_SET:
            if(uav.working is False):
                MOVE_RECORD[uav.uid] = uav.move_record
                UAV_SET.remove(uav)
                time_cost_deep_learning += uav.stradegy_cost 
        

        if(observer_uav.working is True):
            observer_uav.initialize_global_information(global_map)
            observer_uav.path_planning()
            observer_uav.move()
        
        if(observer_uav.working is False):
            time_cost_raw += observer_uav.stradegy_cost

        time_swap += 1

    avg_cost = 0
    for uid,record in MOVE_RECORD.items():
        path_cost = 0
        path_record = []
        position_record = []
        time_step = 0
        for element in record:
            time_step += 1
            x = element[0]
            y = element[1]
            position_record.append([x,y])
            if(len(path_record) > 0 and element[2] == path_record[-1][0]):
                continue
            path_record.append([element[2],time_step])

        for element in path_record:
            grid_x = (element[0] - 1)//int(CONFIG['height'])
            grid_y = (element[0] - 1)%int(CONFIG['height'])
            path_cost += MAP_RECORD[element[1]-1][grid_x][grid_y]
        
        avg_cost += path_cost
        if(observer_uav.openPrint is True):
            path = [element[0] for element in path_record]
            print("uav_{}'s move record is {}".format(uid,position_record))
            print("uav_{}'s move grid's record is {}".format(uid,path))
            print("uav_%d's path cost is %d"%(uid,path_cost))


    observer_path_cost = 0
    observer_path_record = []
    observer_position_record = []
    time_step = 0
    for element in observer_uav.move_record:
        time_step += 1
        x = element[0]
        y = element[1]
        observer_position_record.append([x,y])
        if(len(observer_path_record) > 0 and element[2] == observer_path_record[-1][0]):
            continue
        observer_path_record.append([element[2],time_step])

    for element in observer_path_record:
        grid_x = (element[0] - 1)//int(CONFIG['height'])
        grid_y = (element[0] - 1)%int(CONFIG['height'])
        observer_path_cost += MAP_RECORD[element[1] - 1][grid_x][grid_y]

    if(observer_uav.openPrint is True):
        observer_path = [element[0] for element in observer_path_record]
        print("uav_observer's move record is {}".format(observer_position_record))
        print("uav_observer's move grid's record is {}".format(observer_path))
        print("uav_observer's path cost is %d"%(observer_path_cost))
        print("average uav's path cost is %d"%(avg_cost))

    extra_cost = float(avg_cost - observer_path_cost)/float(observer_path_cost)
    time_saved = float(time_cost_raw - time_cost_deep_learning)/float(time_cost_raw)
    return extra_cost,observer_position_record,time_saved


def experiment(times = 1):
    extra_cost = 0.0
    observer_position_record = None
    i = 0
    time_saved = 0.0

    while(i < times):
        print("process {:.2f}%".format(i+1))
        ret = run()
        extra_cost += ret[0]
        observer_position_record = ret[1]
        time_saved += ret[2]
        i += 1

    #draw_pic(observer_position_record)    
    extra_cost /= times
    time_saved /= times
    return extra_cost,time_saved


def draw_pic(observer_position_record):
    plt.figure(figsize=(10,10))
    #plt.title("[Test case]uav_count=%d, regression_time=%d, bias_max=%d, bias_live_time=%d"%(UAV_BATCH,
    #CONFIG['regression_time'],CONFIG['bias_max'],CONFIG['bias_live_time']))
    plt.title("[Test case]change_step = {:.2f}%".format(CONFIG['change_step']/2*100))
    y_range = CONFIG['width']*CONFIG['grid_size']
    x_range = CONFIG['height']*CONFIG['grid_size']
    plt.xlim(0,x_range)
    plt.ylim(0,y_range)
    x_major_locator=MultipleLocator(2)
    y_major_locator=MultipleLocator(2)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid(color='r',linestyle='-',linewidth=1,alpha=0.2)
    
    for uid,positions in MOVE_RECORD.items():
        for p in positions:
            plt.plot(p[0],p[1],'o',color='#483D8B')

        for i in range(len(positions)-1):
            start = (positions[i][0],positions[i+1][0])
            end = (positions[i][1],positions[i+1][1])
            plt.plot(start,end,color='#483D8B')
    """
    for p in observer_position_record:
        plt.plot(p[0],p[1],'o',color='#FFA07A')
    
    for i in range(len(observer_position_record)-1):
        start = (observer_position_record[i][0],observer_position_record[i+1][0])
        end = (observer_position_record[i][1],observer_position_record[i+1][1])
        plt.plot(start,end,color='#FFA07A')
    """
    plt.show()





if __name__ == '__main__':
    avg_extra_cost,avg_time_saved = experiment(100)
    print("average extra cost is {:.2f}%".format(avg_extra_cost*100))
    print("average time saved using deep learning is {:.2f}%".format(avg_time_saved*100))






