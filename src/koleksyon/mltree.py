import math
import koleksyon.lib as ll
import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', None)

class Bucket:
    def __init__(self, df, s):
        #df - filtered data frame representing only the records that are in this bucket
        #field - the field with the target we are trying to estimate e.g. charges such as HBCharges/PBCharges
        if df is None:
            self.n = 0
        else:
            self.n = len(df) #number of elements in the bucket
        self.s = s #sum of the elements in the bucket
        self.df = df
    def expected_value(self):
        """
        return the maximum likelihood estimate of this bucket based on the data we have
        -- for now, lets just return the average --
        """
        if self.n <= 0:
            return -1.0 #undefined
        elif self.s <= 0.0:
            return 0.0
        else:
            return self.s / self.n
    def __repr__(self):
        rep = "{\'n\': " + str(self.n) \
            + ", \'s\': " + str(self.s) \
            + "}"
        return rep

class Node:
    def __init__(self, id1, tag, level, parent, bucket):
        self.id = id1
        self.tag = tag
        self.level = level
        self.parent = parent  # 1 represents the root of the tree
        self.bucket = bucket
        self.branches = set()
    def __repr__(self):
        return self.__dict__.__repr__()


class LiklihoodTree:
    def __init__(self, df, target="target"):
        self.df = df
        self.target = target
        self.tree = {}  #a hash that contains all the nodes in the tree
        self.idCounter = 0
        self.levels = []
        bucket = Bucket(df, df[target].sum())
        self.rootNode = Node(self.getID(), "root", 0, len(self.levels), bucket)
        self.tree[1] = self.rootNode      
    def getID(self):
        self.idCounter = self.idCounter + 1
        return self.idCounter
    def getLeaves(self):
        leaves = []
        if len(self.levels) == 0:
            leaves.append(1)  #just append the root node and return
        else:
            for node in self.tree.values():
                if node.level == len(self.levels):
                    leaves.append(node.id)
        return leaves
    def addLevel(self, field):
        #get all the leaves that we are going to attach nodes to
        leaves = self.getLeaves()

        #df['L - Region'].value_counts().to_dict()
        new_node_keys = list(self.df[field].value_counts().to_dict().keys())

        for leaf in leaves:
            bucket = self.tree[leaf].bucket
            df = bucket.df
            for node in new_node_keys:
                f = df[field] == node
                s = df[f][self.target].sum()
                if s > 0.0:
                    newBucket = Bucket(df[f], s)
                    newNode = Node(self.getID(), node, len(self.levels) + 1, leaf, newBucket)
                    self.tree[newNode.id] = newNode
                    self.tree[leaf].branches.add(newNode.id)
        #now we have n+1 level so increment
        self.levels.append(field)
    def getCorrectDecendent(self, node, tag):
        """
        Given that we have a node somewhere in the tree, look at the branches from this node and examine the tags, if the tag matches return this node
        """
        #iterate over the decendents
        for i in node.branches:
            nodePtr = self.tree[i]
            if nodePtr.tag == tag:
                return nodePtr
        return None
    def addLevels(self, field_list):
        """
        addLevel, above works fine with 1 or 2 levels, but as the tree gets deeper, it doesn't scale
        this version adds multiple levels at the same time by driving through the data one line at a time
        no matter how many levels you add, adding another level will be O(n*m) where:
        - n is the number of rows
        - m is the number of columns
        """
        df = self.df
        for i in range(len(df)):
            rowdf = df.iloc[i]
            #print(rowdf)
            nodePtr = self.tree[1]  #start at the root
            #the node is not a leaf!  we need to make a leaf coresponding to this new path, follow existing or new branches all the way down
            while nodePtr.level < len(field_list): 
                field = field_list[nodePtr.level]
                tag = rowdf[field]
                #do I need to make a new path or does one exist?
                decendent = self.getCorrectDecendent(nodePtr, tag)
                #if the decendent does not exist, make it
                if decendent is None:
                    newBucket = Bucket(None, 0.0)  #book-keep the charge in the next step
                    #id1, tag, level, parent, bucket
                    newNode = Node(self.getID(), tag, (nodePtr.level + 1), nodePtr.id, newBucket)
                    if newNode.level > len(self.levels):
                        self.levels.append(field)  #we decended into a new level, so need to increase the number of levels we have
                    nodePtr.branches.add(newNode.id)
                    self.tree[newNode.id] = newNode
                    nodePtr = newNode #advance down the path
                else:
                    nodePtr = decendent
            while nodePtr.level > 0:
                #correct the data on the bucket for this node
                bucket = nodePtr.bucket
                bucket.s = bucket.s + rowdf[self.target] 
                bucket.n = bucket.n + 1 #(note, I don't use df here because it would completely crash the memory!) that only really works on small trees
                nodePtr = self.tree[nodePtr.parent] #traverse up to the parent
            #print(self.tree)

    def getMLNode(self, path, node, delta=0):
        """
        return the node that most closely matches the given path.
        e.g. if path = a, b, c
        it will match on a, if no match return the root
        if matches on a, try to match b, if no match return a, else try to match c
        if c matches then return the leaf node that results from traversing a -> b -> c
        """
        #dfs
        if node.level == len(self.levels):  #node is a leaf
            return node

        #else go down a branch to try to get to a leaf
        branches = node.branches
        for branch in branches:
            tnode = self.tree[branch]
            if (tnode.tag == path[0]):
                return self.getMLNode(path[1:], tnode, delta)
        
        #tried to traverse, no data on the branch, go up until there is data
        n = 0
        ptr = node
        while n <= delta: # we can't return a node with no data/not enough data! while that is true, traverse up.
            if ptr.id == 1:  #we are at the root of the tree, return the root
                return ptr
            #traverse up
            ptr = self.tree[ptr.parent]
            n = ptr.bucket.n
        return ptr
    def predict(self, df, delta=0):
        """
        given a dataframe, return the maximum likelihood estimate for each row in the dataframe based on the current tree esimates

        """
        #traverse the tree for the node coresponding to the best match for this event
        predictions = []
        for index, row in df.iterrows():
            #print(row)
            tree_path = []
            for level in self.levels:
                tree_path.append(row[level])
            node = self.getMLNode(tree_path, self.rootNode, delta)
            expected_value = node.bucket.expected_value()
            #print(expected_value)
            predictions.append(expected_value)
        #node = self.tree[nodeID]
        #return float(node.s) / int(node.n)  #we don't add nodes with zero data elements, so this should be just fine
        return predictions
    def __str__(self):
        return self.tree.__str__()
        

   

def init_buckets():
    nodes = df['L - Region'].value_counts().to_dict().keys()
    print(nodes)
    buckets = {}
    buckets["ALL"] = Bucket(df, df.HBCharges.sum())
    for node in nodes:
        f = df['L - Region'] == node
        s = df[f].HBCharges.sum()
        buckets[node] = Bucket(df[f], s)
    print(buckets)
    return buckets


def estimate_error(buckets):
    best_estimate = 0
    best_error = 999999999999
    for estimate in range(0,1000):
        total_error = 0
        for i in buckets:
            #print(i)
            bucket = buckets[i]
            error = bucket.error_based_on_estimate(estimate)
            total_error = error + total_error
        if(total_error < best_error):
            best_error = total_error
            best_estimate = estimate
        print(str(estimate) + " : " + str(total_error))
    print(str(best_estimate) + " : " + str(best_error))

