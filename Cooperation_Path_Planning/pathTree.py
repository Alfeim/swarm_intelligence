class PathTreeNode:
    def __init__(self,uid = -1,value = None):
        self.uid = uid
        self.value = value
        self.prev = self
        self.next = self
        self.count = 0

    def getNext(self):
        return self.next

    def getPrev(self):
        return self.prev

    def getUid(self):
        return self.uid

class PathTree:
    def __init__(self,max_level):
        assert(max_level >= 1),"max level should more than 1"
        self.levelHeads = []
        for _ in range(max_level):
            head_node = PathTreeNode()
            self.levelHeads.append(head_node)
        
    
    def get_max_level(self):
        return len(self.levelHeads)
    
    def addNode(self,level,node):
        
        max_level = self.get_max_level()
        assert(level <= max_level and level > 0),"level out of range"
        head = self.levelHeads[level - 1]
        tail = head.getPrev()
        tail.next = node
        node.next = head
        head.prev = node
        node.prev = tail

        self.levelHeads[level-1].count += 1
    
    def getPath(self,level):
        max_level = self.get_max_level()
        assert(level <= max_level and level > 0),"level out of range"
        head = self.levelHeads[level - 1]
        node = head.getNext()
        path = []
        while(node != None and node != head):
            path.append(node.getUid())
            node = node.getNext()
        return path            
    
    def show(self):
        level = 1
        for head in self.levelHeads:
            print("[element] level: %d"%(level))
            element_list = []
            node = head.getNext()
            while(node != head and node != None):
                element_list.append(node.uid)
                node = node.getNext()
            
            print(element_list)
            level += 1

    def getNodeNumber(self,level,position):
        max_level = self.get_max_level()
        assert(level <= max_level and level > 0),"level out of range"
        head = self.levelHeads[level - 1]
        if(head.count <= 0):
            return None
        pos = 0
        node = head.getNext()
        while(pos < position):
            node = node.getNext()
            pos += 1
        return node.uid

    def removeNode(self,level,position):
        max_level = self.get_max_level()
        assert(level <= max_level and level > 0),"level out of range"
        head = self.levelHeads[level - 1]
        node = head.getNext()
        if(head.count <= position):
            return
        
        pos = 0
        while(node != None and node != head and pos != position):
            node = node.getNext()
            pos += 1
        
        prev = node.getPrev()
        nxt = node.getNext()
        prev.next = nxt
        nxt.prev = prev
        node.next = None
        node.prev = None
    
    def reset(self,level):
        self.levelHeads.clear()
        for _ in range(level):
            head_node = PathTreeNode()
            self.levelHeads.append(head_node)


        



        
