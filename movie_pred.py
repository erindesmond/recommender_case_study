import numpy as np
import pandas as pd
from surprise.model_selection import cross_validate, train_test_split
from surprise import Dataset
from surprise import BaselineOnly
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from gridsearch import grid_search
import pickle

def rmse_change(orig,new):
    return (new-orig)/orig * 100

def best_preditions_model(model_list,data):
    # trainset, testset = train_test_split(data, test_size=.2)
    models = model_list
    best_rmse = 10000
    for model in models:
        gridsearch = grid_search(model)
        gridsearch.fit(data)
        print(gridsearch.best_score['mae'])
        if gridsearch.best_score['rmse'] < best_rmse:
            best_model = gridsearch
            best_rmse = gridsearch.best_score['rmse']
            print(best_rmse)
    print('Best score: ',best_model.best_score['rmse'])
    print('Best params: ',best_model.best_params['rmse'])
    return best_model

def gen_predictions(n,user,movies,all_ratings,all_movies):
    best_collab = pickle.load(open('best.p','rb'))
    tset = []
    for movie in movies:
        if not(((all_ratings['userId']==user)&(all_ratings['movieId']==movie)).any()):
            tset.append((user,movie,0))
    pred= best_collab.test(tset)
    test_predict = np.array([(p.iid,p.est) for p in pred])
    pred_merge = pd.DataFrame(test_predict,columns=['movieId','rating'])
    pred_merge.movieId = pred_merge.movieId.astype(int)
    out = pd.merge(left=pred_merge,right=all_movies,on='movieId')
    return out[['title','rating']].loc[:n]



if __name__ == '__main__':
    data = Dataset.load_builtin('ml-100k')
    all_ratings = pd.read_csv('movies/ratings.csv')
    all_movies = pd.read_csv('movies/movies.csv')
