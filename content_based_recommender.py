import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender(object):

    def __init__(self):
        self.df = None #set in fit
        self.genre_matrix = None
        self.cos_sim_matrix = None

    def fit(self, df, target_column='genres', desired_row_index='movieId'):
        self.df = df
        self.genre_matrix = self.make_genre_matrix(self.df, target_column, desired_row_index)
        self.cos_sim_matrix = self.make_cos_sim_matrix_df(self.genre_matrix)

    def make_genre_matrix(self, df, target_column='genres', desired_row_index='movieId'):
        '''one hot encode genres that apply to each movie'''
        topic_list = np.unique(np.concatenate(df[target_column]))
        num_obs = len(df[target_column])
        num_topics = len(topic_list)
        results_array = np.empty((num_obs, num_topics))
        for row_idx in range(num_obs):
            results_array[row_idx]= np.isin(topic_list, df.iloc[row_idx, 2])

        results_df = pd.DataFrame(results_array, columns=topic_list,\
                                        index = df[desired_row_index])
        return results_df

    def make_cos_sim_matrix_df(self, df):
        '''find pairwise cosine similarity matrix for dataframe of
        observations and features'''
        similarities = cosine_similarity(df)
        similarities_df = pd.DataFrame(similarities, index = df.index, columns = df.index)
        return similarities_df

    def find_the_most_similar(self, movie_idx, num_to_find=20):
        '''
        idx: integer type, index of the movie we care about
        num_to_find: integer type, number of recommendations requested
        sim_mtx: similarity matrix of all movies compared to each other
        movie_mtx: original matrix of movies with names
        '''
        sim_mtx = self.cos_sim_matrix
        movie_mtx = self.df
        row = sim_mtx[movie_idx]
        original_movie = movie_mtx[movie_mtx['movieId']==movie_idx]
        top_n_idx = np.argsort(row)[-1*(num_to_find+1):-1]
        top_n_ids = sim_mtx.columns[top_n_idx]

        top_n_movies = movies[np.isin(movies['movieId'],top_n_ids.values)]
        return original_movie, top_n_movies

if __name__ == '__main__':
    movies = pd.read_csv('data/movies/movies.csv')
    movies['genres'] = movies.genres.str.split("|")
    movies['year'] = movies.title.str.split(' \(').str[1]
    movies['title'] = movies.title.str.split(' \(').str[0]
    movies.year = movies.year.str.split('\)').str[0]
    c = ContentBasedRecommender()
    c.fit(movies)
    movie, recommendations = c.find_the_most_similar(2, 20)
