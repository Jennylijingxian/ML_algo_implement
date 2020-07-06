import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import shap

def standard_scaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler.transform(x) 

def pca_plot(score, coeff,y, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                     "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                     labels[i], color='g', ha='center', va='center')

def pearson_rank(x,y):
    N,D = x.shape
    corr_ls = []
    corr_ls = [spearmanr(x[:,feature],y)[0] for feature in range(D)]
    corr_arr = np.array(corr_ls)
    return corr_arr

def ols_featimp(x,y):
    reg = LinearRegression().fit(x, y)
    return reg.coef_

def permutation_importances(model, x_test, y_test, metric):
    prediction = model.predict(x_test)
    if metric == metrics.r2_score:
        baseline = metric(y_test, prediction)
    elif metric == metrics.accuracy_score:
        baseline = metric(y_test, prediction.round(),normalize=False)
    imp = []
    for col in x_test.columns:
        save = x_test.loc[:,col].copy()
        x_test.loc[:,col] = np.random.RandomState(seed=42).permutation(x_test.loc[:,col])
        if metric == metrics.r2_score:
            m = metric(y_test,model.predict(x_test))
        elif metric == metrics.accuracy_score:
            m = metric(y_test,model.predict(x_test).round(), normalize=False)
        x_test.loc[:,col] = save
        imp.append(baseline - m)
    return np.array(imp)

def dropcol_importances(model,x_train, y_train, x_test, y_test, metric):
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(x_train,y_train)
    prediction = model_.predict(x_test)
    if metric == metrics.r2_score:
        baseline = metric(y_test, prediction)
    elif metric == metrics.accuracy_score:
        baseline = metric(y_test, prediction.round(),normalize=False)
    imp = []
    for col in x_train.columns:
        x_train_new = x_train.drop(col,axis = 1)
        x_test_new = x_test.drop(col,axis = 1)
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(x_train_new, y_train)
        prediction_new = model_.predict(x_test_new)
        if metric == metrics.r2_score:
            score = metric(y_test, prediction_new)
        elif metric == metrics.accuracy_score:
            score = metric(y_test,prediction_new.round(), normalize=False)
        difference = baseline - score
        imp.append(difference)
    return imp

def auto_selection(model, x_train, y_train, x_test, y_test, metric):
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(x_train, y_train)
    prediction = model_.predict(x_test)
    baseline = metric(y_test, prediction)
    print("Base line is:" + str(baseline))
    x_train_new = x_train
    x_test_new = x_test
    dropped = []
    for i in range(len(x_train.columns)):
        print("----------------Round" + str(i) + "---------------------" )
        print("Begining feature name" + str(x_train_new.columns))
        dropc_col_imp = dropcol_importances(
            model_, x_train, y_train, x_test, y_test, metrics.r2_score)
#         print(dropc_col_imp)
        min_featimp_idx = dropc_col_imp.index(min(dropc_col_imp))
        x_train_new = x_train_new.drop(x_train_new.columns[min_featimp_idx], axis=1)
        x_test_new = x_test_new.drop(x_test_new.columns[min_featimp_idx], axis=1)
        dropped.append(x_train_new.columns[min_featimp_idx])
        print("Dropped feature name:" + str(x_train_new.columns[min_featimp_idx]))
        model__ = clone(model)
        model__.random_state = 999
        model__.fit(x_train_new, y_train)
        prediction_new = model__.predict(x_test_new)
        score = metric(y_test, prediction_new)
        print("Score:" + str(score))
        if score > baseline:
            baseline = score
        else:
            break
    print("---------------- Final  ---------------------")
    print("Finally we drop:" + str(dropped[:-1]))
    x_train_final = x_train.drop(dropped[:-1], axis=1)
    x_test_final = x_test.drop(dropped[:-1], axis = 1)
    model_final = clone(model)
    model_final.random_state = 999
    model_final.fit(x_train_final, y_train)
    prediction_final = model_final.predict(x_test_final)
    score = metric(y_test, prediction_final)
    print("Final score is:" +str(score))
    dropc_col_imp = dropcol_importances(
            model_, x_train_final, y_train, x_test_final, y_test, metrics.r2_score)
    sns.barplot(x = dropc_col_imp, y = x_train_final.columns.values)
    plt.show()
    return model_

def get_shap_values(model, x, feature_names):
    shap.initjs()
    explainer = shap.TreeExplainer(model,feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(x,check_additivity=False)
    shap.summary_plot(shap_values = shap_values, feature_names = feature_names, plot_type = "bar")
    return shap_values

def mean_variance(model, x_test, y_test, metric, n):
    imp_distribution = np.array([]).reshape((0,x_test.shape[1]))
    size = x_test.shape[0]
    for i in range(n):
        index = np.random.choice(np.arange(size),size, replace = True)
        x_bs = x_test.reset_index().loc[index,:].drop(columns = ['index'])
        y_bs = y_test.reset_index().loc[index,:].drop(columns = ['index'])
        imp = permutation_importances(model, x_bs, y_bs, metric)
        imp_distribution = np.vstack((imp_distribution, imp))
    return imp_distribution

def p_value(model, x_test, y_test, metric, n):
    imp_distribution = permutation_importances(model, x_test, y_test, metric)
    for i in range(n):
        y_bs = np.random.permutation(y_test)
        imp = permutation_importances(model, x_test, y_bs, metric)
        imp_distribution = np.vstack((imp_distribution, imp))
    return imp_distribution



"""
Reference and Citation:
1. PCA part plot cited from: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
2. Drop column and permutation importance referenced from: https://explained.ai/rf-importance/index.html
"""










