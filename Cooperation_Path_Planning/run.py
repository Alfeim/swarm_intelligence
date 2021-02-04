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

SRC_NUMBER = 1
DST_NUMBER = 1
UAV_COUNTS = 21
UAV_BATCH = 3

CONFIG_PATH = "./config.json"
logger = Logger('log_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
CONFIG = json.loads(open(CONFIG_PATH,"r").read())
destination = CONFIG['width']*CONFIG['height']
turns_of_change = 5
rate_of_change = 0.05


def run():
    global turns_of_change
    global rate_of_change
    global destination
    
    time_swap = 1
    MAP_RECORD = []
    UAV_SET = []
    MOVE_RECORD = dict()

    global_map = (np.multiply(np.random.randn(CONFIG['width'],CONFIG['height']),255.0/(2.58*2)) + 127.5).astype(np.uint8)
    observer_uav = UAV(-1,CONFIG_PATH,logger)
    observer_uav.position[0] = 0
    observer_uav.position[1] = 0
    observer_uav.reset_destination(destination)
    MAP_RECORD.append(copy.deepcopy(global_map))
    #initialization
    for i in range(UAV_COUNTS):
        current_uav = UAV(i+1,CONFIG_PATH,logger)
        current_uav.position[0] = 0
        current_uav.position[1] = 0
        current_uav.initialize_global_information(global_map)
        current_uav.reset_destination(destination)
        UAV_SET.append(current_uav)

    while(len(UAV_SET)>0 or observer_uav.working is True):        
        print("**time_swap: %d**"%time_swap)
        change_flag = (random.random() <= turns_of_change)
        if(change_flag is True):
            print("[Notice]global map info is changed")
            element_count = CONFIG['width']*CONFIG['height']
            change_count = int(element_count*rate_of_change)
            for i in range(change_count):
                number = random.randint(1,element_count)
                x = (number - 1)//int(CONFIG['height'])
                y = (number - 1)%int(CONFIG['height'])
                global_map[x][y] = random.randint(0,255)
        
        MAP_RECORD.append(copy.deepcopy(global_map))
        
        print("[Notice]update neighborhood information")
        for uav in UAV_SET:
            uav.update_one_skip_neighbor(UAV_SET)
        
        for uav in UAV_SET:
            uav.update_two_skip_neighbor()
        
        print("[Notice]update environment information")
        for uav in UAV_SET:
            uav.update_local_environment_information(global_map,time_swap)
        
        for uav in UAV_SET:
            uav.update_environment_information()
        
        print("[Notice]into stradegy period")
        for uav in UAV_SET:
            if((uav.uid-1)//UAV_BATCH < time_swap):
                uav.path_planning()
        
        print("[Notice]into working period")
        for uav in UAV_SET:
            if((uav.uid-1)//UAV_BATCH < time_swap):
                uav.move()

        for uav in UAV_SET:
            if(uav.working is False):
                MOVE_RECORD[uav.uid] = uav.move_record
                UAV_SET.remove(uav)

        if(observer_uav.working is True):
            observer_uav.initialize_global_information(global_map)
            observer_uav.path_planning()
            observer_uav.move()

        time_swap += 1
        time.sleep(0.1)

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
        path = [element[0] for element in path_record]
        print("uav_{}'s move record is {}".format(uid,position_record))
        print("uav_{}'s move grid's record is {}".format(uid,path))
        print("uav_%d's path cost is %d"%(uid,path_cost))

    avg_cost //=len(MOVE_RECORD)

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

    observer_path = [element[0] for element in observer_path_record]
    print("uav_observer's move record is {}".format(observer_position_record))
    print("uav_observer's move grid's record is {}".format(observer_path))
    print("uav_observer's path cost is %d"%(observer_path_cost))
    print("average uav's path cost is %d"%(avg_cost))

    return observer_path_cost,avg_cost


def experiment(times = 1):
    optimal_cost = 0
    average_cost = 0
    i = times
    while(i > 0):
        ret = run()
        optimal_cost += ret[0]
        average_cost += ret[1]
        i -= 1

    optimal_cost //= times
    average_cost //= times
    print("optimal_cost %d"%optimal_cost)
    print("average_cost %d"%average_cost)


if __name__ == '__main__':
    experiment(2)






