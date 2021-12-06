from numpy import mean
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pathlib
from skopt.space import Integer,Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.cluster import KMeans,DBSCAN
from warnings import catch_warnings
from warnings import simplefilter

def bayesian_optimization(data,cluster_method):
    X = data['in']

    if 'groups' in data.columns:
        y = data['groups']  #catagory
        semantic_label='groups'
    else:
        y = data['category']
        semantic_label='category'

    if cluster_method=='dbscan':
        model=DBSCAN
        search_space = [Real(1.2, 2.5, name='eps'), Integer(5, 12, name='min_samples')]
    else:
        model=KMeans
        search_space = [Integer(3,12, name='n_clusters')]



    @use_named_args(search_space)
    def evaluate_model(**params):

        prediction = model(**params).fit_predict(list(X))
        data['prediction']=prediction
        labels=set(prediction)
        result=[]
        for x in labels:
            selected=data[data['prediction']==x]
            valuecount = selected[semantic_label].value_counts()
            mixrating = valuecount.max() / valuecount.sum() #len(selected)
            result.append(mixrating)
        estimate = mean(result)

        return 1.0 - estimate
    # perform optimization
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        result = gp_minimize(evaluate_model, search_space)

    # summarizing finding:
        print('Best Accuracy: %.3f' % (1.0 - result.fun))
    #print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
    if cluster_method=='dbscan':
        print('Best Parameters: eps=%d, minPoint=%d' % (result.x[0], result.x[1]))
        return result.x[0], result.x[1]
    else:

        print('Best Parameters: cluster_num=%d' % (result.x[0]))
        return result.x[0] #result.x=[4]   result.fun=0.0003242453













# DEFAULT_DATA=str(pathlib.Path(__file__).parent.resolve()) + "/data/processed/"
# data=pd.read_pickle(DEFAULT_DATA + "processed_data.pkl")
# x=Bayesian_optimization(data,'kmeans')
# print('last',x)
# # generate 2d classification dataset
# X=data['in']
# y=data['groups']
# #X, y = make_blobs(n_samples=500, centers=3, n_features=2)
# # define the model
# model = DBSCAN
# # define the space of hyperparameters to search
# search_space = [Real(1.2, 2.5, name='eps'), Integer(5, 12, name='min_samples')]
#
#
# # define the function used to evaluate a given configuration
# @use_named_args(search_space)
# def evaluate_model(**params):
#     # something
#
#     prediction = model(**params).fit_predict(list(X))
#     data['prediction']=prediction
#     labels=set(prediction)
#     result=[]
#     for x in labels:
#         selected=data[data['prediction']==x]
#         valuecount = selected.groups.value_counts()
#         mixrating = valuecount.max() / valuecount.sum() #len(selected)
#         result.append(mixrating)
#     estimate = mean(result)
#
#     return 1.0 - estimate
#
#
# # perform optimization
# result = gp_minimize(evaluate_model, search_space)
# # summarizing finding:
# print('Best Accuracy: %.3f' % (1.0 - result.fun))
# print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
#

# example of bayesian optimization with scikit-optimize

