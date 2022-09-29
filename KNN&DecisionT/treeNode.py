class treeNode:
    def __init__(self,value,left,mid,right):
        self.value = value
        self.left = left
        self.mid = mid
        self.right = right
    
    def setValue(self,value):
        self.value = value
    
    def getLeft(self):
        return self.left

    def getValue(self):
        return self.value
    
    def getRight(self):
        return self.right
    
    def getMid(self):
        return self.mid
    
    def hasLeft(self):
        if self.left!=None:
            return True
        else:
            return False
    
    def hasRight(self):
        if self.right!=None:
            return True
        else:
            return False
        
    def hasMid(self):
        if self.mid!=None:
            return True
        else:
            return False
        
    def printNode(self):
        print(self.value)
