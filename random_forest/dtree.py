from scipy.stats import mode
#from statistics import mode, mean
# import scipy.stats.mean as mean
# import scipy.stats.mode as mode
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild


    def predict(self, x_test):
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)

    # def predict_(self, x_test):
    #     if x_test[self.col] <= self.split:
    #         return self.lchild.predict_(x_test)
    #     else:
    #         return self.rchild.predict_(x_test)
    #
    # def predict(self,x_test):
    #     row,col = x_test.shape
    #     y = np.zeros(row)
    #     for i in range(row):
    #         y[i] = self.predict_(x_test[i,:])
    #     return y

    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        else:
            return self.rchild.leaf(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        #self.label = self.prediction(y)

    def predict(self, x_test):
        return self.prediction

    def leaf(self, x_test):
        return self
    # def predict_(self, x_test):
    #     #print(x_test)
    #     return self.prediction

    # def predict(self, x_test):
    #     return self.prediction

    # def predict(self, x_test):
    #     return self.label
    # def predict_(self, x_test):
    #     #print(x_test)
    #     return self.label

    # return prediction passed to constructor of LeafNode
    # lines 3,4 from algorithm above


class DecisionTree621:
    def __init__(self, min_samples_leaf=3, max_features = 0.3, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.loss = loss # loss function; either np.std or gini
		
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """

        self.root = self.RFdtreefit(X, y, self.max_features, self.min_samples_leaf)

    def RFbestsplit(self,X,y,loss, max_features):
        col = -1
        split = -1
        loss_value = loss(y)
        real_max_features = int(max_features * X.shape[1])
        vars = np.random.choice(X.shape[1], real_max_features, replace = False)
        for column in vars:
            candidates = np.random.choice(list(set(X[:,column])), min(len(set(X[:,column])), 11), replace = False)
            for split_point in candidates:
                yl = y[X[:,column] <= split_point]
                yr = y[X[:,column] > split_point]
                if len(yl) < self.min_samples_leaf or len(yr) < self.min_samples_leaf:
                    continue
                l = (len(yl)*loss(yl) + len(yr)*loss(yr))/len(y)
                if l == 0:
                    return column,split_point
                if l < loss_value:
                    col = column
                    split = split_point
                    loss_value = l
        return col, split

    # def bestsplit(self,X,y,loss):
    #     col = -1
    #     split = -1
    #     loss_value = loss(y)
    #
    #     for column in range(X.shape[1]):
    #         candidates = np.random.choice(X[:,column],11)
    #         for split_point in candidates:
    #             yl = y[X[:,column] <= split_point]
    #             yr = y[X[:,column] > split_point]
    #             if len(yl) == 0 or len(yr) == 0:
    #                 continue
    #             else:
    #                 l = (len(yl)*loss(yl) + len(yr)*loss(yr))/len(y)
    #                 if l == 0:
    #                     return column,split_point
    #                 elif l < loss_value:
    #                     col = column
    #                     split = split_point
    #                     loss_value = l
    #     return col, split

    def RFdtreefit(self, X, y, max_features, min_samples_leaf):
        if len(X) <= min_samples_leaf:
            return self.create_leaf(y)
        col, split = self.RFbestsplit(X, y, self.loss, self.max_features)
        if col == -1:
            return self.create_leaf(y)
        else:
            lchild = self.RFdtreefit(X[X[:, col] <= split, :], y[X[:, col] <= split],max_features, min_samples_leaf)
            rchild = self.RFdtreefit(X[X[:, col] > split, :], y[X[:, col] > split],max_features, min_samples_leaf)
            return DecisionNode(col, split, lchild, rchild)

    # def fit_(self, X, y):
    #     """
    #     Recursively create and return a decision tree fit to (X,y) for
    #     either a classifier or regressor.  This function should call self.create_leaf(X,y)
    #     to create the appropriate leaf node, which will invoke either
    #     RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
    #     on the type of self.
    #
    #     This function is not part of the class "interface" and is for internal use, but it
    #     embodies the decision tree fitting algorithm.
    #
    #     (Make sure to call fit_() not fit() recursively.)
    #     """
    #     # 返回bestsplit，新建一个decision node返回回去，在这个里面实现那棵数
    #     # 在这里面把整棵树建好，返回结果是root节点
    #
    #     if len(X) <= self.min_samples_leaf:
    #         return self.create_leaf(y)
    #     col,split = self.bestsplit(X,y,self.loss)
    #     if col == -1:
    #         return self.create_leaf(y)
    #     else:
    #         lchild = self.fit_(X[X[:,col] <= split,:],y[X[:,col] <= split])
    #         rchild = self.fit_(X[X[:,col] > split,:],y[X[:,col] > split])
    #         return DecisionNode(col, split, lchild, rchild)


    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        row, col = X_test.shape
        y = np.zeros(row)
        for i in range(row):
            y[i] = self.root.predict(X_test[i, :])
        return y

    def leaf(self, X_test):
        row, col = X_test.shape
        leaves = []
        for i in range(row):
            leaves.append(self.root.leaf(X_test[i, :]))
        return leaves


class RegressionTree621(DecisionTree621):
    def __init__(self, max_features = 0.3, min_samples_leaf=3, oob_idx = 0):
        super().__init__(min_samples_leaf, max_features, loss=np.std)
        self.oob_idx = oob_idx

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y,np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, max_features = 0.3, min_samples_leaf=3, oob_idx = 0):
        super().__init__(min_samples_leaf, max_features, loss=gini)
        self.oob_idx = oob_idx

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        acc_score = accuracy_score(y_test,y_pred)
        return acc_score


    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        #return LeafNode(y, mode)
        # print(int(mode(y)[0]))
        return LeafNode(y,mode(y)[0][0])


def gini(y):
    "Return the gini impurity score for values in y"
    category, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    sum = 0
    for i in counts:
        sum += (i/total)*(i/total)
    return 1- sum


