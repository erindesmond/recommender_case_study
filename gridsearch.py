from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF, SVD


'''The SVD++ algorithm, an extension of SVD taking into account implicit ratings.

The prediction r̂ ui
 is set as:

r̂ ui=μ+bu+bi+qTi(pu+|Iu|−12∑j∈Iuyj)
Where the yj
 terms are a new set of item factors that capture implicit ratings. Here, an implicit rating describes the fact that a user u
 rated an item j
, regardless of the rating value.'''



def grid_search(surprise_model):

    if type(surprise_model()) == type(SVDpp()):

        param_grid = {'n_factors':[20] , 'n_epochs':[20], 'lr_all':[0.005, 0.007, 0.05, 0.07, 0.5, 0.7, 1.0], 'reg_all':[0.02, 0.05, 0.2, 0.5]}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3)

    elif type(surprise_model()) == type(SVD()):

        param_grid = {'n_epochs':[20], 'lr_all':[0.005, 0.007, 0.05, 0.07, 0.5, 0.7, 1.0], 'reg_all':[0.02, 0.05, 0.2, 0.5]}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3)

    elif type(surprise_model()) == type(NMF()):

        param_grid = {'n_epochs':[20], 'reg_pu':[0.02, 0.04, 0.06, 0.08, 0.2], 'reg_qi':[0.02, 0.04, 0.06, 0.08, 0.2]}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3)

    elif type(surprise_model()) == type(BaselineOnly()):
        param_grid = {'bsl_options': {'method': ['als', 'sgd'], 'reg': [1, 2], 'learning_rate': [0.005, 0.05, 0.5, 1.0]}}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3)

    return gs






if __name__ == '__main__':

    # Use movielens-100K
    data = Dataset.load_builtin('ml-100k')

    #SVD = SVD()
    test = grid_search(SVD)

            # print('Best score for NMF: ', gs_nmf.best_score['rmse'])
            # # combination of parameters that gave the best RMSE score
            # print('Best parameters for NMF: ', gs_nmf.best_params['rmse'])





#
