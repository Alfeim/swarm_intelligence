from Vertex import Vertex

class StradegyNode:
    """
    stradegy node base class
    """
    def __init__(self):
        pass

    def execute(self,src,dst):
        pass


class Dijkstra(StradegyNode):
    """
    Djikstra stradegy
    @source: source vertex
    @edges: the relationship between all neighbouring vertexes
    @vertexes: all path nodes(vertexes)
    @relaxed: the flag to determin whether has been relaxed
    """
    def __init__(self,matrix,src_id):
        self.source = None
        self.map = matrix
        self.edges = dict()
        self.vertexes = []
        self.relaxed = False
        dim = matrix.shape
        max_row = dim[0]
        max_col = dim[1]

        for i in range(max_row):
            for j in range(max_col):
                neighbors = []
                current_number = i*(max_col) + (j+1)
                #add edges and vertexs
                if(i - 1 >= 0):
                    neighbor_number = current_number - max_col
                    if not(self.edges.__contains__((current_number,neighbor_number))):
                        self.edges[current_number,neighbor_number] = matrix[i-1][j]
                    neighbors.append(neighbor_number)
                if(i + 1 < max_row):
                    neighbor_number = current_number + max_col
                    if not(self.edges.__contains__((current_number,neighbor_number))):
                        self.edges[current_number,neighbor_number] = matrix[i+1][j]
                    neighbors.append(neighbor_number)
                if(j - 1 >= 0):
                    neighbor_number = current_number - 1
                    if not(self.edges.__contains__((current_number,neighbor_number))):
                        self.edges[current_number,neighbor_number] = matrix[i][j-1]
                    neighbors.append(neighbor_number)
                if(j + 1 < max_col):
                    neighbor_number = current_number + 1
                    if not(self.edges.__contains__((current_number,neighbor_number))):
                        self.edges[current_number,neighbor_number] = matrix[i][j+1]
                    neighbors.append(neighbor_number)
                current_vertex = Vertex(current_number,neighbors)
                self.vertexes.append(current_vertex)

                if(src_id != None and current_number == src_id):
                    self.source = current_vertex    

        #To facilitate mapping vertex index and offset position, insert a placeholder
        self.vertexes.insert(0,'place_holder')
    
    """
    to reset source node.once the source node is reset,then it is necessary to doing relax operation again
    """
    def reset_source(self,src_id):
        if(src_id == self.source.vid):
            return

        print("[Notice]: reset source")
        for i in range(1,len(self.vertexes)):
            self.vertexes[i].know = False
            self.vertexes[i].dist = float('inf')
            if(self.vertexes[i].vid == src_id):
                self.source = self.vertexes[i]
                self.relaxed = False

    """
    To get the nearest vertex of candidate list
    """
    def get_candidate_vertex(self,vertexMinSet):
        the_min = 0
        the_index = 0
        j = 0
        for i in range(1,len(self.vertexes)):
            if(self.vertexes[i].know is True):
                continue
            else:
                if(j==0):
                    the_min = self.vertexes[i].dist
                    the_index = i
                else:
                    if(self.vertexes[i].dist < the_min):
                        the_min = self.vertexes[i].dist
                        the_index = i
                j += 1
        

        vertexMinSet.remove(self.vertexes[the_index])
        return self.vertexes[the_index]
    
    """
    relax operation
    """
    def relax(self,uid = -1):
        if(self.relaxed is True):
            return

        vertexMinSet = set(self.vertexes[1:])
        #if(uid != -1):
            #print("[Notice]:node %d doing relaxing operation"%uid)
        self.source.dist = 0
        while(len(vertexMinSet)!=0):
            v = self.get_candidate_vertex(vertexMinSet)
            v.know = True
            for w in v.outList:#w stand for index
                if(self.vertexes[w].know is True):
                    continue
                if(self.vertexes[w].dist == float('inf')):
                    self.vertexes[w].dist = v.dist + self.edges[(v.vid,w)]
                    self.vertexes[w].prev = v.vid
                else:
                    if((v.dist + self.edges[(v.vid,w)]) < self.vertexes[w].dist):
                        self.vertexes[w].dist = v.dist + self.edges[(v.vid,w)]
                        self.vertexes[w].prev = v.vid
                    else:
                        pass
        
        self.relaxed = True

    """
    execute operation,which need source index and destination index
    """
    def execute(self,src_index,dst_index,uid = -1):
        if(src_index != self.source.vid):
            self.reset_source(src_index)
        
        self.relax(uid)
        start = self.source.vid
        path_list = []
        def get_traj(index):#参数是顶点在vlist中的索引
            if(index == start):#终点
                path_list.append(index)
                return
            if(self.vertexes[index].dist == float('inf')):
                print('No avaliable path')
                return
            path_list.append(index)
            get_traj(self.vertexes[index].prev)

        get_traj(dst_index)
        path_list.reverse()
        return path_list



class StradegyChain:
    """
    Stradegy Chain class
    """
    def __init__(self,max_chain_len):
        self.chain = []
        for _ in range(max_chain_len):
            self.chain.append(None)
    
    def set_stradegy(self,position,stradegy):
        self.chain[position] = stradegy

    def get_stradegy(self,level):
        return self.chain[level-1]

    def execute(self,level,src,dst):
        return self.chain[level-1].execute(src,dst) 