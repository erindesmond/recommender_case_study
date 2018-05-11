import numpy as np
import pandas as pd
from surprise.model_selection import cross_validate, train_test_split
from surprise import Dataset
from surprise import BaselineOnly
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from gridsearch import grid_search

def rmse_change(orig,new):
    return (new-orig)/orig * 100

def best_preditions_model(data):
    # trainset, testset = train_test_split(data, test_size=.2)
    models = (SVD, SVDpp, NMF, BaselineOnly)
    best_rmse = 10000
    for model in models:
        gridsearch = grid_search(model)

        gridsearch.fit(data)
        print(gridsearch.best_score['rmse'])
        if gridsearch.best_score['rmse'] < best_rmse:
            best_model = gridsearch
            best_rmse = gridsearch.best_score['rmse']
            print(best_rmse)
    print('Best score: ',best_model.best_score['rmse'])
    print('Best params: ',best_model.best_params['rmse'])
    return best_model

if __name__ == '__main__':
    data = Dataset.load_builtin('ml-100k')
