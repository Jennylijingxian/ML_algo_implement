import numpy as np
from dtree import *
from sklearn.utils import resample
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from typing import Any


class RandomForest621:
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def bootstrap(self, X, y):
        size = X.shape[0]
        idx = np.array(range(len(X)))
        boot_idx = resample(idx, replace=True, n_samples=size)
        oob_idx = np.setxor1d(idx, boot_idx)
        boot_x = X[boot_idx]
        boot_y = y[boot_idx]
        return boot_x,boot_y, oob_idx

    def fit(self, X, y):
        """
        1) Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.
        2) Keep track of the indexes of the OOB records for each tree.
        3) After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.trees = []
        if self.att == "regression":
            for i in range(self.n_estimators):
                boot_x, boot_y, oob_idx = self.bootstrap(X,y)
                dtree = RegressionTree621(self.max_features, self.min_samples_leaf,oob_idx)
                dtree.oob = oob_idx
                #print(dtree.oob)
                dtree.fit(boot_x, boot_y)
                self.trees.append(dtree)
            if self.oob_score:
                self.oob_score_ = self.oobscore(X,y)

        elif self.att == "classifier":
            for i in range(self.n_estimators):
                boot_x, boot_y, oob_idx = self.bootstrap(X, y)
                dtree = ClassifierTree621(self.max_features, self.min_samples_leaf,oob_idx)
                dtree.fit(boot_x, boot_y)
                self.trees.append(dtree)
            if self.oob_score:
                self.oob_score_ = self.oobscore(X, y)

    # def oob_score_regression(self,RF ,oob_x, oob_y):
    #     oob_pred = np.zeros(len(oob_x))
    #     oob_count = np.zeros(len(oob_x))
    #     for t in self.trees:
    #         leaves = t.leaf(oob_x)
    #         count = np.array([leaf.n for leaf in leaves])
    #         pred = np.array([leaf.prediction for leaf in leaves])
    #         oob_pred += count * pred
    #         oob_count += count
    #     weighted_avg = oob_pred / oob_count
    #     return r2_score(oob_y, weighted_avg)
    #
    # def oob_score_classifier(self,RF ,oob_x, oob_y):
    #     table = np.zeros((oob_x.shape[0], oob_x.shape[1]))
    #     for t in self.trees:
    #         leaves = t.leaf(oob_x)
    #         count = [leaf.n for leaf in leaves]
    #         pred = [leaf.prediction for leaf in leaves]
    #         for i in range(len(pred)):
    #             table[i, pred[i]] += count[i]
    #     y_pred = np.argmax(table, axis=1)
    #     return accuracy_score(oob_y, y_pred)
    #
    # if self.oob_score:
    #     if self.att == "regression":
    #         oob_score_regression(fit(X,y),)

class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        #self.trees = super().fit(X,y)
        self.att = "regression"
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features


    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        y_pred = np.zeros(len(X_test))
        y_count = np.zeros(len(X_test))
        for t in self.trees:
            leaves = t.leaf(X_test)
            count = np.array([leaf.n for leaf in leaves])
            pred = np.array([leaf.prediction for leaf in leaves])
            y_pred += np.multiply(count,pred)
            y_count += count
        weighted_avg = y_pred/y_count
        return weighted_avg

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        weighted_avg = self.predict(X_test)
        return r2_score(y_test, weighted_avg)

    def oobscore(self, X, Y):
        y_pred = np.zeros(len(X))
        y_count = np.zeros(len(X))
        count = np.zeros(len(X))
        pred = np.zeros(len(X))
        for t in self.trees:
            oob_idx = t.oob_idx
            #print(oob_idx)
            leaves = t.leaf(X[oob_idx])
            count[oob_idx] = np.array([leaf.n for leaf in leaves])
            pred[oob_idx] = np.array([leaf.prediction for leaf in leaves])
            y_pred += np.multiply(count,pred)
            y_count += count
        oob = y_count>0
        weighted_avg = y_pred[oob] / y_count[oob]
        return r2_score(Y[oob], weighted_avg)

class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.att = "classifier"

    def predict(self, X_test) -> np.ndarray:
        table = np.zeros((X_test.shape[0], X_test.shape[1]))
        for t in self.trees:
            leaves = t.leaf(X_test)
            count = [leaf.n for leaf in leaves]
            pred = [leaf.prediction for leaf in leaves]
            for i in range(len(pred)):
                table[i,pred[i]] += count[i]
        y_pred = np.argmax(table, axis=1)
        return y_pred
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def oobscore(self, X, Y):
        #print(len(Y))
        table = np.zeros((X.shape[0], X.shape[1]))
        count = np.zeros(len(X))
        pred = np.zeros(len(X))
        for t in self.trees:
            oob_idx = t.oob_idx
            leaves = t.leaf(X[oob_idx])
            count[oob_idx] = [leaf.n for leaf in leaves]
            pred[oob_idx] = [leaf.prediction for leaf in leaves]
            # print(table)
            # print(table.shape)
            for i in range(len(pred)):
                table[i, int(pred[i])] += count[i]
        y_pred = np.argmax(table, axis=1)
        return accuracy_score(Y, y_pred)


#
# import numpy as np
# from sklearn.datasets import \
#     load_boston, load_iris, load_diabetes, load_wine, \
#     load_breast_cancer, fetch_california_housing
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import inspect
#
# from rf import RandomForestRegressor621, RandomForestClassifier621
#
# def test_boston_oob():
#     X, y = load_boston(return_X_y=True)
#     run_regression_test(X, y, min_training_score = .86, oob=True)
#
# def test_boston_min_samples_leaf_oob():
#     X, y = load_boston(return_X_y=True)
#     run_regression_test(X, y, ntrials=5, min_samples_leaf=5, grace=0.08, oob=True)
#
# def test_california_housing_oob():
#     X, y = fetch_california_housing(return_X_y=True)
#     run_regression_test(X, y, min_training_score = .79, grace=0.15, oob=True)
#
# def test_iris_oob():
#     X, y = load_iris(return_X_y=True)
#     run_classification_test(X, y, ntrials=5, min_training_score=0.93, oob=True)
#
# def test_wine_oob():
#     X, y = load_wine(return_X_y=True)
#     run_classification_test(X, y, ntrials=5, min_training_score=0.98, oob=True)
#
# def test_wine_min_samples_leaf_oob():
#     X, y = load_wine(return_X_y=True)
#     run_classification_test(X, y, ntrials=10, min_training_score=0.98, min_samples_leaf=5, grace=0.2, oob=True)
#
# def test_breast_cancer_oob():
#     X, y = load_breast_cancer(return_X_y=True)
#     run_classification_test(X, y, ntrials=5, min_training_score=0.96, oob=True)
#
# def run_classification_test(X, y, ntrials=1, min_samples_leaf=3, max_features=0.3, min_training_score=1.0, grace=.07, oob=False, n_estimators=15):
#     stack = inspect.stack()
#     caller_name = stack[1].function[len('test_'):]
#     X = X[:500]
#     y = y[:500]
#
#     scores = []
#     train_scores = []
#     oob_scores = []
#
#     sklearn_scores = []
#     sklearn_train_scores = []
#     sklearn_oob_scores = []
#
#     for i in range(ntrials):
#         X_train, X_test, y_train, y_test = \
#             train_test_split(X, y, test_size=0.20)
#
#         rf = RandomForestClassifier621(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
#         rf.fit(X_train, y_train)
#         score = rf.score(X_train, y_train)
#         train_scores.append(score)
#         score = rf.score(X_test, y_test)
#         scores.append(score)
#         oob_scores.append(rf.oob_score_)
#
#         sklearn_rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
#         sklearn_rf.fit(X_train, y_train)
#         sklearn_score = sklearn_rf.score(X_train, y_train)
#         sklearn_train_scores.append(sklearn_score)
#         sklearn_score = sklearn_rf.score(X_test, y_test)
#         sklearn_scores.append(sklearn_score)
#         if oob:
#             sklearn_oob_scores.append(sklearn_rf.oob_score_)
#         else:
#             sklearn_oob_scores.append(0.0)
#
#     if oob:
#         assert np.abs(np.mean(scores)- np.mean(sklearn_scores)) < grace, \
#                f"OOB accuracy: {np.mean(oob_scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_oob_scores):.2f}"
#     assert np.mean(train_scores) >= min_training_score, \
#            f"Training accuracy: {np.mean(train_scores):.2f} must {min_training_score:.2f}"
#     assert np.mean(scores)+grace >= np.mean(sklearn_scores), \
#            f"Testing accuracy: {np.mean(scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_scores):.2f}"
#
#     print()
#     if oob:
#         print(f"{caller_name}: 621 OOB score {np.mean(oob_scores):.2f} vs sklearn OOB {np.mean(sklearn_oob_scores):.2f}")
#     print(f"{caller_name}: 621 accuracy score {np.mean(train_scores):.2f}, {np.mean(scores):.2f}")
#     print(f"{caller_name}: Sklearn accuracy score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_scores):.2f}")
#
# def run_regression_test(X, y, ntrials=2, min_training_score = .85, min_samples_leaf=3, max_features=0.3, grace=.08, oob=False, n_estimators=18):
#     stack = inspect.stack()
#     caller_name = stack[1].function[len('test_'):]
#     X = X[:500]
#     y = y[:500]
#
#     scores = []
#     train_scores = []
#     oob_scores = []
#
#     sklearn_scores = []
#     sklearn_train_scores = []
#     sklearn_oob_scores = []
#
#     for i in range(ntrials):
#         X_train, X_test, y_train, y_test = \
#             train_test_split(X, y, test_size=0.20)
#
#         rf = RandomForestRegressor621(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
#         rf.fit(X_train, y_train)
#         score = rf.score(X_train, y_train)
#         train_scores.append(score)
#         score = rf.score(X_test, y_test)
#         scores.append(score)
#         oob_scores.append(rf.oob_score_)
#
#         sklearn_rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
#         sklearn_rf.fit(X_train, y_train)
#         sklearn_score = sklearn_rf.score(X_train, y_train)
#         sklearn_train_scores.append(sklearn_score)
#         sklearn_score = sklearn_rf.score(X_test, y_test)
#         sklearn_scores.append(sklearn_score)
#         if oob:
#             sklearn_oob_scores.append(sklearn_rf.oob_score_)
#         else:
#             sklearn_oob_scores.append(0.0)
#
#     print()
#     if oob:
#         print(f"{caller_name}: 621 OOB score {np.mean(oob_scores):.2f} vs sklearn OOB {np.mean(sklearn_oob_scores):.2f}")
#     print(f"{caller_name}: 621     Train R^2 score mean {np.mean(train_scores):.2f}, stddev {np.std(train_scores):3f}")
#     print(f"{caller_name}: Sklearn Train R^2 score mean {np.mean(sklearn_train_scores):.2f}, stddev {np.std(sklearn_train_scores):3f}")
#     print(f"{caller_name}: 621     Test  R^2 score mean {np.mean(scores):.2f}, stddev {np.std(scores):3f}")
#     print(f"{caller_name}: Sklearn Test  R^2 score mean {np.mean(sklearn_scores):.2f}, stddev {np.std(sklearn_scores):3f}")
#
#     assert np.mean(train_scores) >= min_training_score, \
#            f"Training R^2: {np.mean(train_scores):.2f} must be >= {min_training_score}"
#     assert np.mean(scores)+grace >= np.mean(sklearn_scores), \
#            f"Testing R^2: {np.mean(scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_scores):.2f}"
#     if oob:
#         assert np.abs(np.mean(oob_scores) - np.mean(sklearn_oob_scores)) < grace, \
#             f"OOB R^2: {np.mean(oob_scores):.2f} must be within {grace:2f} of sklearn score: {np.mean(sklearn_oob_scores):.2f}"
#
#
# #test_boston_oob()
# #test_boston_min_samples_leaf_oob()
# #test_california_housing_oob()
# # test_iris_oob()
# # test_wine_oob()
# # test_wine_min_samples_leaf_oob()
# test_breast_cancer_oob()