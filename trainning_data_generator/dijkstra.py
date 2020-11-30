import numpy as np

class Vertex:
    #顶点类
    def __init__(self,vid,outList):
        self.vid = vid           #出边
        self.outList = outList   #出边指向的顶点id的列表，也可以理解为邻接表
        self.know = False        #默认为假
        self.dist = float('inf') #source到该点的距离,默认为无穷大
        self.prev = 0            #上一个顶点的id，默认为0
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.vid == other.vid
        else:
            return False
    def __hash__(self):
        return hash(self.vid)

def add_edges(start,end,value,edges):
    edges[(start,end)] = value

def create_all_vertex(matrix):
    dim = matrix.shape
    max_row = dim[0]
    max_col = dim[1]
    
    vList = []
    edges = dict()
    source = None
    for i in range(max_row):
        for j in range(max_col):
            neighbors = []
            current_number = i*(max_col) + (j+1)
            #add edges and vertexs
            if(i - 1 >= 0):
                neighbor_number = current_number - max_col
                if not(edges.__contains__((current_number,neighbor_number))):
                    add_edges(current_number,neighbor_number,matrix[i - 1][j],edges)
                neighbors.append(neighbor_number)

            if(i + 1 < max_row):
                neighbor_number = current_number + max_col
                if not(edges.__contains__((current_number,neighbor_number))):
                    add_edges(current_number,neighbor_number,matrix[i + 1][j],edges)
                neighbors.append(neighbor_number)

            if(j - 1 >= 0):
                neighbor_number = current_number - 1
                if not(edges.__contains__((current_number,neighbor_number))):
                    add_edges(current_number,neighbor_number,matrix[i][j - 1],edges)
                neighbors.append(neighbor_number)

            if(j + 1 < max_col):
                neighbor_number = current_number + 1
                if not(edges.__contains__((current_number,neighbor_number))):
                    add_edges(current_number,neighbor_number,matrix[i][j + 1],edges)
                neighbors.append(neighbor_number)

            current_vertex = Vertex(current_number,neighbors)
            vList.append(current_vertex)

            if( current_number == get_source_number(matrix)):
                source = current_vertex

    vList.insert(0,False)
    vSet=set(vList[1:])
    
    return vList,vSet,source,edges


def get_source_number(matrix):
    dim = matrix.shape
    source_number = dim[0] * dim[1]
    return source_number

def get_unknown_min(vList,vSet):
    the_min = 0
    the_index = 0
    j = 0
    for i in range(1,len(vList)):
        if(vList[i].know is True):
            continue
        else:
            if(j==0):
                the_min = vList[i].dist
                the_index = i
            else:
                if(vList[i].dist < the_min):
                    the_min = vList[i].dist
                    the_index = i
            j += 1
    #此时已经找到了未知的最小的元素是谁
    vSet.remove(vList[the_index])#相当于执行出队操作
    return vList[the_index]


def relax(matrix):
    vList,vSet,source,edges = create_all_vertex(matrix)
    source.dist = 0

    while(len(vSet)!=0):
        v = get_unknown_min(vList,vSet)
        v.know = True
        for w in v.outList:#w为索引
            if(vList[w].know is True):
                continue
            if(vList[w].dist == float('inf')):
                vList[w].dist = v.dist + edges[(v.vid,w)]
                vList[w].prev = v.vid
            else:
                if((v.dist + edges[(v.vid,w)]) < vList[w].dist):
                    vList[w].dist = v.dist + edges[(v.vid,w)]
                    vList[w].prev = v.vid
                else:#原路径长更小，没有必要更新
                    pass

    return vList

def get_shortest_path(start,index,vList):
    traj_list = []
    def get_traj(index):#参数是顶点在vlist中的索引
        if(index == start):#终点
            traj_list.append(index)
            return
        if(vList[index].dist == float('inf')):
            print('从起点到该顶点根本没有路径')
            return
        traj_list.append(index)
        get_traj(vList[index].prev)

    get_traj(index)
    return traj_list

"""
matrix = np.random.randint(1,100,(20,20))
vList = relax(matrix)
source_index = get_source_number(matrix)
get_shortest_path(source_index,5,vList)
"""
