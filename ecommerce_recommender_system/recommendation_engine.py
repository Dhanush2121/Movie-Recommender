import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def hybrid_recommendation(user_id, movies, ratings):
    # Create user-item matrix
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Collaborative: find similar users
    user_sim_matrix = cosine_similarity(user_movie_matrix)
    sim_users = list(enumerate(user_sim_matrix[user_id - 1]))
    sim_users = sorted(sim_users, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 similar users

    similar_user_ids = [user[0] + 1 for user in sim_users]
    similar_users_ratings = ratings[ratings['userId'].isin(similar_user_ids)]

    # Content: use genre tags
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(movies['genres'])
    genre_sim = cosine_similarity(genre_matrix)

    movie_scores = {}
    for movie_id in ratings['movieId'].unique():
        if movie_id not in user_movie_matrix.columns:
            continue
        avg_rating = similar_users_ratings[similar_users_ratings['movieId'] == movie_id]['rating'].mean()
        movie_index = movies[movies['movieId'] == movie_id].index[0]
        genre_sim_sum = genre_sim[movie_index].sum()
        score = (avg_rating if not np.isnan(avg_rating) else 0) + genre_sim_sum * 0.01
        movie_scores[movie_id] = score

    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, _ in sorted_movies[:10]]
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()
