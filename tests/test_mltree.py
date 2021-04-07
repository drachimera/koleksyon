import unittest
import pandas as pd
import koleksyon.mltree as mltree
pd.set_option('display.max_columns', None)

class TestMLTree(unittest.TestCase):
    def setUp(self):
        self.testfile = "testing_data/mltree_testing_data.csv"
        self.lt = None

    def test_init_tree(self):
        df = pd.read_csv(self.testfile)
        #print(df)
        #print(df.dtypes)
        self.assertEqual(len(df), 10)
        lt = mltree.LiklihoodTree(df, "result1")  #result1 and result2 are the targets on this toy dataset
        #base state is as follows:
        expected = "{1: {'id': 1, 'tag': 'root', 'level': 0, 'parent': 0, 'bucket': {'n': 10, 's': 2650}, 'branches': set()}}"
        #print(lt.tree)
        self.assertEqual(expected,  str(lt.tree))


    def check_single_level_tree(self, lt, target, branchTags):
        """
        lt - the likelihood tree to check...
        """
        root = lt.tree[1]
        df = lt.df
        #compute the leef_ids e.g. something like set([2,3,4,5,6])
        leaf_ids = set(range(2,len(branchTags) + 2))
        #check some basic statistics at the root
        self.assertEqual(root.parent, 0) #root of the root is zero
        self.assertEqual(root.bucket.n, len(df))  #check that the bucket count@root == len(df)
        self.assertEqual(root.bucket.s, df[lt.target].sum()) # sum of everything in the 'target' column (total charges); same as root
        self.assertEqual(root.branches, leaf_ids)  #ids on the branches are same as we expect
        tagsFound = []
        i = 0
        for branch in sorted(root.branches):
            node = lt.tree[branch]
            self.assertEqual(node.parent, 1) #assert that all the nodes point back to the root
            self.assertEqual(len(node.branches),0) #assert that the node is indeed a leaf
            self.assertEqual(node.level, 1) #all of these nodes should be at level 1
            tagsFound.append(node.tag)
            #filter charges/count based on the tag and compare
            #print(node.tag)
            f = df[lt.levels[0]] == node.tag  #filter based on the first level in the tree e.g. 'department' or whatever field was used first in the 'add levels'
            s = df[f][lt.target].sum()
            n = len(df[f])
            self.assertEqual(node.bucket.n, n) #the number of data elements represented by the node are the same as a filter on the dataframe
            self.assertEqual(node.bucket.s, s) #the sum of the data elements represented by the node are the same as the sum of a filtered dataframe
            i = i + 1
        self.assertEqual(set(branchTags), set(tagsFound))  #we have all the tags for the level


    def test_add_layer(self):
        df = pd.read_csv(self.testfile)
        target1 = "result1"  #the target field that we are attempting to esitmate, e.g. charges
        target2 = "result2"  
        lt1 = mltree.LiklihoodTree(df, target1)
        lt2 = mltree.LiklihoodTree(df, target2)
        field = "department"  #adding a single layer from the toy dataset's 'department' column
        lt1.addLevel(field)  #going with 2 different targets, because the trees are slightly different...
        lt2.addLevel(field)
        #print the tree for debugging if any of the following tests fail
        print("Tree1 **************************************")
        print(lt1.__str__())  
        print("Tree2 **************************************")
        print(lt2.__str__())  
        self.assertEqual(lt1.target, target1)  #assert that result1, the charges are the target in the tree
        self.assertEqual(lt2.target, target2)  #result2, same as above
        self.assertEqual(lt1.levels, [field])  #levels are 0 (root) and 1
        self.assertEqual(lt2.levels, [field])
        #print(df['department'])
        #0    Community Pediatric and Adolescent Medicine
        #1                            Laboratory Medicine
        #2                                    Dermatology
        #3                    Community Internal Medicine
        #4                                    Dermatology
        #5                                    Dermatology
        #6                            Laboratory Medicine
        #7                             Orthopedic Surgery
        #8                            Laboratory Medicine
        #9                                    Dermatology
        self.assertEqual(len(lt2.tree), 6)  #the number of nodes in the tree is 1+ unique rows in dataframe, we have 5 departments and a root node
        #note that if you have a target with zeros, nodes for those values won't be included!
        self.assertEqual(len(lt1.tree), 4)  #two departments had zeros!
        #expected branch tags:
        branchTags1 = ["Dermatology","Laboratory Medicine","Community Internal Medicine"]
        self.check_single_level_tree(lt1, target1, branchTags1)
        branchTags2 = ["Dermatology","Laboratory Medicine","Community Pediatric and Adolescent Medicine","Orthopedic Surgery","Community Internal Medicine"]
        self.check_single_level_tree(lt2, target2, branchTags2)
        #TODO: is it a problem that zero in the 'target' currently trims the tree?

    def populate_keys(self, lt):
        #populate the leaf and branches for a 2 level tree
        leaf_keys = []
        branch_keys = []
        for node_id in lt.tree.keys():
            node = lt.tree[node_id]
            if node.level == 2:
                leaf_keys.append(node_id)
            if node.level == 1:
                branch_keys.append(node_id)
        return branch_keys, leaf_keys

    def check_leaves(self, lt, field1, field2):
        branch_keys, leaf_keys = self.populate_keys(lt)
        df = lt.df
        #double check that adding the leaves on the tree are the same as if we do the math on the dataframe
        for lkey in leaf_keys:
            lnode = lt.tree[lkey]
            parent = lt.tree[lnode.parent]
            tag1 = parent.tag
            tag2 = lnode.tag
            #print(field1)
            #print(tag1)
            #print(field2)
            #print(tag2)
            f1 = (df[field1] == tag1)
            dfprime = df[f1]
            f2 = (dfprime[field2] == tag2)
            dfp = dfprime[f2]
            #print(dfp)
            self.assertEqual(lnode.bucket.n, len(dfp))  #assert that filtering the dataframe results in the same number of records as what is in the tree
            self.assertEqual(lnode.bucket.s, dfp[lt.target].sum())   #assert that adding the charges via filtering is the same as what is on the node
            #assert that the values on the parent are the same as filtering
            self.assertEqual(parent.bucket.n, len(dfprime))
            self.assertEqual(parent.bucket.s, dfprime[lt.target].sum())



    def test_add_second_layer(self):
        print("\nTesting adding a second layer with Region")
        df = pd.read_csv(self.testfile)
        target = "result2"
        lt = mltree.LiklihoodTree(df, target)
        field1 = "department"
        field2 = "region"
        lt.addLevel(field1)
        lt.addLevel(field2)
        print(lt.__str__())  #print the tree for debugging if any of the following tests fail
        self.check_leaves(lt, field1, field2)

    def test_multiple_constructions(self):
        print("\nTesting building a tree two different ways to see if they are the same")
        df = pd.read_csv(self.testfile)
        target = "result2"
        levels = ["payer", "specialty", "region"]
        print(df)
        #create the tree by adding the layers one at a time
        lt1 = mltree.LiklihoodTree(df, target)
        lt1.addLevel(levels[0])
        lt1.addLevel(levels[1])
        lt1.addLevel(levels[2])
        print(lt1)


    def tRegion(self, lt):
        print("Testing adding a second layer with Region")
        print(lt.__str__())  #print the tree for debugging if any of the following tests fail
        df = lt.df
        leaf_keys = []
        branch_keys = []
        for node_id in lt.tree.keys():
            node = lt.tree[node_id]
            if node.level == 2:
                leaf_keys.append(node_id)
            if node.level == 1:
                branch_keys.append(node_id)
        field1 = "L - Department"
        field2 = "L - Region"
        for lkey in leaf_keys:
            lnode = lt.tree[lkey]
            parent = lt.tree[lnode.parent]
            tag1 = parent.tag
            tag2 = lnode.tag
            #print(field1)
            #print(tag1)
            #print(field2)
            #print(tag2)
            f1 = (df[field1] == tag1)
            dfprime = df[f1]
            f2 = (dfprime[field2] == tag2)
            dfp = dfprime[f2]
            #print(dfp)
            self.assertEqual(lnode.bucket.n, len(dfp))  #assert that filtering the dataframe results in the same number of records as what is in the tree
            self.assertEqual(lnode.bucket.s, dfp[lt.charges_field].sum())   #assert that adding the charges via filtering is the same as what is on the node
            #assert that the values on the parent are the same as filtering
            self.assertEqual(parent.bucket.n, len(dfprime))
            self.assertEqual(parent.bucket.s, dfprime[lt.charges_field].sum())

        

    #def test_addLevel(self):
    def foo(self):
        print("***TEST: addLevel, a dataframe filtering method for adding tree levels one layer at a time... slow for lots of layers!***")
        df = mltree.load_data(data=self.testfile, rws=1000)
        lt = mltree.LiklihoodTree(df)
        lt.addLevel("L - Department") 
        #Test adding the single level
        self.tDepartment(lt)
        
        lt.addLevel("L - Region")
        self.tRegion(lt)

        #TODO test three layers deep
        #lt.addLevel("M - PayorName")

    #def test_addLevels(self):
    def bar(self):
        print("***TEST: addLevels, a line by line construction method to build the tree based on the entire dataset multiple layers at a time***")
        df = mltree.load_data(data=self.testfile, rws=1000)
        lt = mltree.LiklihoodTree(df)
        lt.addLevels(["L - Department"]) 
        #Test adding the single level
        self.tDepartment(lt)

        #Test adding two levels
        lt = mltree.LiklihoodTree(df)
        lt.addLevels(["L - Department", "L - Region"]) 
        self.tRegion(lt)


if __name__ == '__main__':
    unittest.main()