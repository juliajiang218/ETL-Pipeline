"""
Filename: content_Rec.py
Author: Julia Jiang
Date: 2024-11-09
Description:
    Content-based movie recommendation system.
Disclaimer: 
    This code is provided for educational and informational purposes only. It is intended to demonstrate basic
    algorithms for computing user preference vectors and movie recommendation scores. The code is not optimized 
    for production use and may require modifications for deployment in a high-performance environment.

    No warranty or guarantee is provided regarding the functionality, reliability, or accuracy of this code.
    The author is not liable for any damages, losses, or issues arising from the use or misuse of this code.
    Users assume full responsibility for testing, modifying, and adapting this code to meet their specific needs.

    By using this code, you agree to hold the author harmless for any direct or indirect consequences resulting 
    from its use. Use at your own risk.
"""

import numpy as np
import sqlite3

def get_movie_genres(movie_id, conn):
    """
    Retrieves genres for a given movie from the database.
    
    Args:
        movie_id (int): The movie ID.
        conn (sqlite3.Connection): SQLite database connection.
        
    Returns:
        list: List of genre names associated with the movie.
    """
    query = """
    SELECT g.name 
    FROM categories c
    JOIN genre g ON c.genre_id = g.id
    WHERE c.movie_id = ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (movie_id,))
    return [row[0] for row in cursor.fetchall()]

def encode_movie_genre_matrix(movie_ids, all_genres, conn):
    """
    Encodes a movie-genre matrix where each row represents a movie and each column represents a genre.
    A value of 1 indicates that the movie belongs to that genre, and 0 means it does not.
    
    Args:
        movie_ids (list): List of movie IDs.
        all_genres (list): List of all unique genres in sorted order.
        conn (sqlite3.Connection): SQLite database connection.
        
    Returns:
        numpy.ndarray: Binary matrix representing movies and their corresponding genres.
    """
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    genre_index = {genre: idx for idx, genre in enumerate(all_genres)}
    movie_genre_matrix = np.zeros((len(movie_ids), len(all_genres)), dtype=int)
    
    for movie_id in movie_ids:
        genres = get_movie_genres(movie_id, conn)
        for genre_name in genres:
            if movie_id in movie_index and genre_name in genre_index:
                row_idx = movie_index[movie_id]
                col_idx = genre_index[genre_name]
                movie_genre_matrix[row_idx, col_idx] = 1
    
    return movie_genre_matrix

def compute_user_preference(user_ratings, movie_genre_matrix):
    """
    Computes the user-preference vector based on the user's movie ratings.
    
    Args:
        user_ratings (numpy array): A 1D array of user's ratings for each movie (0 means unrated).
        movie_genre_matrix (numpy array): A 2D binary array where rows represent movies and columns represent genres.
    
    Returns:
        numpy array: The user-preference vector based on genre preferences.
    """
    rated_movies_mask = user_ratings >= 0
    rated_movies_genres = movie_genre_matrix[rated_movies_mask]
    weighted_genre_sum = np.dot(user_ratings[rated_movies_mask], rated_movies_genres)
    
    if np.sum(weighted_genre_sum) != 0:
        user_preference_vector = weighted_genre_sum / np.sum(weighted_genre_sum)
    else:
        user_preference_vector = np.zeros_like(weighted_genre_sum)
    
    return np.round(user_preference_vector, 2)

def recommend_movies(user_preference_vector, movie_genre_matrix):
    """
    Generates movie recommendations based on the user-preference vector.
    
    Args:
        user_preference_vector (numpy array): The user-preference vector based on genre preferences.
        movie_genre_matrix (numpy array): A 2D binary array where rows represent movies and columns represent genres.
    
    Returns:
        numpy array: A recommendation score for each movie.
    """
    recommendation_scores = np.dot(user_preference_vector, movie_genre_matrix.T)
    return np.round(recommendation_scores, 2)

def get_user_ratings(user_id, conn):
    """
    Retrieves the user's rated movies and their scores from the database.
    
    Args:
        user_id (int): The user ID.
        conn (sqlite3.Connection): SQLite database connection.
        
    Returns:
        list: A list of (movie_id, rating) tuples or an empty list if no ratings are found.
    """
    query = "SELECT movie_id, score FROM user_rating WHERE userID = ?"
    cursor = conn.cursor()
    cursor.execute(query, (user_id,))
    return cursor.fetchall()

def get_all_genres(conn):
    """
    Retrieves all unique genres from the database.
    
    Args:
        conn (sqlite3.Connection): SQLite database connection.
        
    Returns:
        list: Sorted list of all unique genres.
    """
    query = "SELECT DISTINCT name FROM genre"
    cursor = conn.cursor()
    cursor.execute(query)
    return sorted([row[0] for row in cursor.fetchall()])

def get_popular_movies(conn, limit=5):
    """
    Retrieves a list of popular movies based on the highest number of votes or highest ratings.
    
    Args:
        conn (sqlite3.Connection): SQLite database connection.
        limit (int): Number of popular movies to retrieve.
        
    Returns:
        list: List of popular movie titles.
    """
    query = """
    SELECT orig_title 
    FROM movie 
    ORDER BY votes DESC, revenue DESC 
    LIMIT ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (limit,))
    return [row[0] for row in cursor.fetchall()]

def content_based_recommendation(user_id, db_path):
    """
    Generates content-based movie recommendations for a user.
    
    Args:
        user_id (int): The user ID.
        db_path (str): The path to the SQLite database.
        
    Returns:
        list: List of top recommended movie titles or fallback movies.
    """
    conn = sqlite3.connect(db_path)
    
    rated_movies = get_user_ratings(user_id, conn)
    if not rated_movies:
        print("No ratings found for this user. Showing popular movies instead.")
        popular_movies = get_popular_movies(conn)
        conn.close()
        return popular_movies
    
    rated_movie_ids = [movie_id for movie_id, _ in rated_movies]
    user_ratings_vector = convert_to_user_rating_vector(rated_movies)
    
    all_genres = get_all_genres(conn)
    rated_movie_genre_matrix = encode_movie_genre_matrix(rated_movie_ids, all_genres, conn)
    
    user_preference_vector = compute_user_preference(user_ratings_vector, rated_movie_genre_matrix)
    
    query_unrated_movies = "SELECT id FROM movie WHERE id NOT IN ({seq}) LIMIT 100".format(seq=','.join(['?'] * len(rated_movie_ids)))
    cursor = conn.cursor()
    cursor.execute(query_unrated_movies, rated_movie_ids)
    unrated_movie_ids = [row[0] for row in cursor.fetchall()]
    
    if not unrated_movie_ids:
        print("No unrated movies found for recommendations.")
        conn.close()
        return []
    
    unrated_movie_genre_matrix = encode_movie_genre_matrix(unrated_movie_ids, all_genres, conn)
    recommendation_scores = recommend_movies(user_preference_vector, unrated_movie_genre_matrix)
    
    movie_titles_query = "SELECT id, orig_title FROM movie WHERE id IN ({seq})".format(seq=','.join(['?'] * len(unrated_movie_ids)))
    cursor.execute(movie_titles_query, unrated_movie_ids)
    movie_titles = {row[0]: row[1] for row in cursor.fetchall()}
    
    recommendations = sorted(zip(unrated_movie_ids, recommendation_scores), key=lambda x: -x[1])
    top_recommendations = [movie_titles[movie_id] for movie_id, _ in recommendations[:5]]
    
    conn.close()
    return top_recommendations

def convert_to_user_rating_vector(rated_movies):
    """
    Converts the scores of rated movie IDs to a user rating vector.
    
    Args:
        rated_movies (list): List of (movie_id, score) tuples for movies rated by the user.
        
    Returns:
        np.array: User rating vector.
    """
    movie_ids = sorted(set(movie_id for movie_id, _ in rated_movies))
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    user_ratings = np.zeros(len(movie_ids))
    for movie_id, score in rated_movies:
        if movie_id in movie_index:
            user_ratings[movie_index[movie_id]] = score
    return user_ratings

