import pandas as pd

def load_data():
    movies = pd.read_csv("ecommerce_recommender_system/data/movies.csv")
    ratings = pd.read_csv("ecommerce_recommender_system/data/ratings.csv")
    return movies, ratings

