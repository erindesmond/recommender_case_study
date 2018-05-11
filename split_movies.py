import pandas as pd


if __name__ == '__main__':
    movies = pd.read_csv('data/movies/movies.csv')
    movies['genres'] = movies.genres.str.split("|")
    movies['year'] = movies.title.str.split(' \(').str[1]
    movies['title'] = movies.title.str.split(' \(').str[0]
    movies.year = movies.year.str.split('\)').str[0]
