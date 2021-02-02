class Vertex:
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