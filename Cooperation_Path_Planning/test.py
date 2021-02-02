import stradegy
import pathTree
import numpy as np
import UAV
import time
import random
from loger import Logger

logger = Logger()

"""
uav_1 = UAV.UAV(1,'./config.json',logger)
global_map = np.random.randn(uav_1.config['width'],uav_1.config['height'])
global_map = (np.multiply(global_map,255.0/(2.58*2)) + 127.5).astype(np.uint8)
uav_1.initialize_global_information(global_map)
uav_1.update_all_level_maps()
uav_1.position[0] = 0
uav_1.position[1] = 0

uav_2 = UAV.UAV(2,'./config.json',logger)
uav_2.initialize_global_information(global_map)
uav_2.update_all_level_maps()
uav_2.position[0] = 0
uav_2.position[1] = 0

uav_3 = UAV.UAV(3,'./config.json',logger)
uav_3.initialize_global_information(global_map)
uav_3.update_all_level_maps()
uav_3.position[0] = 0
uav_3.position[1] = 0


uav_4 = UAV.UAV(4,'./config.json',logger)
uav_4.initialize_global_information(global_map)
uav_4.update_all_level_maps()
uav_4.position[0] = 0
uav_4.position[1] = 0


uav_5 = UAV.UAV(5,'./config.json',logger)
uav_5.initialize_global_information(global_map)
uav_5.update_all_level_maps()
uav_5.position[0] = 0
uav_5.position[1] = 0

global_node_list = [uav_1,uav_2,uav_3,uav_4]

print("****global map*****")
print(global_map)
print("*******************")

uav_1.reset_destination(98)
uav_2.reset_destination(98)
uav_3.reset_destination(98)
uav_4.reset_destination(98)
uav_5.reset_destination(98)
uav_5.weight_regression = False

times = 1
path1 = []
path2 = []
path3 = []
path4 = []
path5 = []

sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0
sum5 = 0

uav_5.path_planning()
while(True):

    print("update one skip neighbor")
    if(uav_1.working):
        uav_1.update_one_skip_neighbor(global_node_list)
    if(uav_2.working):
        uav_2.update_one_skip_neighbor(global_node_list)
    if(uav_3.working):
        uav_3.update_one_skip_neighbor(global_node_list)
    if(uav_4.working):
        uav_4.update_one_skip_neighbor(global_node_list)

    print("update two skip neighbor")
    if(uav_1.working):
        uav_1.update_local_environment_information(global_map,times)
    if(uav_2.working):
        uav_2.update_local_environment_information(global_map,times)
    if(uav_3.working):
        uav_3.update_local_environment_information(global_map,times)
    if(uav_4.working):
        uav_4.update_local_environment_information(global_map,times)

    print("update environment information")
    if(uav_1.working):
        uav_1.update_environment_information()
    if(uav_2.working):
        uav_2.update_environment_information()        
    if(uav_3.working):
        uav_3.update_environment_information()    
    if(uav_4.working):
        uav_4.update_environment_information()

    print("time: %d \n"%times)
    
    if(uav_1.working):
        uav_1.path_planning()
        print("uav1's path:")
        uav_1.path_tree.show()
    if(uav_2.working):
        uav_2.path_planning()
        print("uav2's path:")
        uav_2.path_tree.show()
    if(uav_3.working):
        uav_3.path_planning()
        print("uav3's path:")
        uav_3.path_tree.show()
    if(uav_4.working):
        uav_4.path_planning()
        print("uav4's path:")
        uav_4.path_tree.show()
    

    if(uav_1.working and times >= uav_1.uid):
        uav_1.move()
    if(uav_2.working and times >= uav_2.uid):
        uav_2.move()
    if(uav_3.working and times >= uav_3.uid):
        uav_3.move()
    if(uav_4.working and times >= uav_4.uid):
        uav_4.move()
    if(uav_5.working):
        uav_5.move()
    
    if not(uav_1.working or uav_2.working or uav_3.working or uav_4.working or uav_5.working):
        break

    time.sleep(0.1)
    times += 1

path1 = uav_1.move_record
path2 = uav_2.move_record
path3 = uav_3.move_record
path4 = uav_4.move_record
path5 = uav_5.move_record

cols = uav_1.config['height']

for node in path1:
    x = (node[2]-1)//cols
    y = (node[2]-1)%cols
    sum1 += global_map[x][y]

for node in path2:
    x = (node[2]-1)//cols
    y = (node[2]-1)%cols
    sum2 += global_map[x][y]

for node in path3:
    x = (node[2]-1)//cols
    y = (node[2]-1)%cols
    sum3 += global_map[x][y]

for node in path4:
    x = (node[2]-1)//cols
    y = (node[2]-1)%cols
    sum4 += global_map[x][y]

for node in path5:
    x = (node[2]-1)//cols
    y = (node[2]-1)%cols
    sum5 += global_map[x][y]

print("uav_1's path sum: %d "%sum1)
print([node[2] for node in path1])
print("uav_2's path sum: %d "%sum2)
print([node[2] for node in path2])
print("uav_3's path sum: %d "%sum3)
print([node[2] for node in path3])
print("uav_4's path sum: %d "%sum4)
print([node[2] for node in path4])
print("uav_5(shortest)'s path sum: %d "%sum5)
print([node[2] for node in path5])


"""
i = 0
all_extra_cost = 0.0
all_saved_time = 0.0
while(i < 100):
    print("process {}%".format(i+1))
    
    uav_6 = UAV.UAV(6,'./config.json',logger)
    wide = uav_6.config['height']*uav_6.config['grid_size']
    global_map = np.random.randn(uav_6.config['width'],uav_6.config['height'])
    global_map = (np.multiply(global_map,255.0/(2.58*2)) + 127.5).astype(np.uint8)
    uav_6.initialize_global_information(global_map)
    uav_6.update_all_level_maps()
    uav_6.position[0] = random.randint(1,wide-1)
    uav_6.position[1] = random.randint(1,wide-1)

    uav_7 = UAV.UAV(7,'./config.json',logger)
    uav_7.initialize_global_information(global_map)
    uav_7.update_all_level_maps()
    uav_7.position[0] = uav_6.position[0]
    uav_7.position[1] = uav_6.position[1]
    uav_6.multi_levels = False
    uav_7.multi_levels = True


    border =uav_6.config['height']*uav_6.config['height']
    dst = random.randint(1,border)
    uav_6.reset_destination(dst)
    uav_7.reset_destination(dst)

    time6_start = time.time()
    uav_6.path_planning()
    time6_cost = time.time() - time6_start

    time7_start = time.time()
    uav_7.path_planning()
    time7_cost = time.time() - time7_start

    path6 = uav_6.path_tree.getPath(2)
    path7 = uav_7.path_tree.getPath(2)

    sum6 = 0
    sum7 = 0

    uav_6.path_tree.show()
    uav_7.path_tree.show()

    for node in path6:
        x = (node - 1)//uav_6.config['height']
        y = (node - 1)%uav_6.config['height']
        sum6 += global_map[x][y]
    
    for node in path7:
        x = (node - 1)//uav_7.config['height']
        y = (node - 1)%uav_7.config['height']
        sum7 += global_map[x][y]

    extra_path = float(sum7-sum6)/sum6
    saved_time = float(time6_cost - time7_cost)/time6_cost
    all_extra_cost += extra_path
    all_saved_time += saved_time
    print("multi_level path planing result is extra with {:.2f}%".format(100*extra_path))
    print("multi_level path planing save time with {:.2f}%".format(100*saved_time))
    i += 1

print("\n")
print("avg extra cost {:.2f}%".format(all_extra_cost))
print("avg saved time {:.2f}%".format(all_saved_time))










