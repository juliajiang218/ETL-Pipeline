"""
Filename: recommendation.py
Author: Julia Jiang
Date: 2024-11-09
Description:
    Collaborative filtering movie recommendation system with matrix factorization using the Surprise library,
    cosine similarity calculation, and user-based filtering.
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
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from typing import Dict, List



def load_feedback_matrix(database_path: str) -> pd.DataFrame:
    """
    Connect to the database and load the feedback matrix.
    :param database_path: Path to the SQLite database file.
    :return: Feedback matrix as a DataFrame.
    """
    conn = sqlite3.connect(database_path)
    query = """
    select
        ur.userID,
        ur.movie_id,
        ur.score
    from user_rating ur
    join movie m on ur.movie_id = m.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    feedback_matrix = df.pivot_table(index='userID', columns='movie_id', values='score', fill_value=0)
    return feedback_matrix


def load_movie_titles(database_path: str) -> Dict[int, str]:
    """
    Connect to the database and load movie titles.
    :param database_path: Path to the SQLite database file.
    :return: Dictionary mapping movie IDs to their English titles.
    """
    conn = sqlite3.connect(database_path)
    query = """
    SELECT 
        m.id as movie_id,
        m.eng_title
    FROM movie m
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    movie_titles = df.set_index('movie_id')['eng_title'].to_dict()
    return movie_titles


def matrix_factorization(feedback_matrix: pd.DataFrame, n_factors: int = 100, n_epochs: int = 20, 
                         lr_all: float = 0.005, reg_all: float = 0.02, test_size: float = 0.1) -> SVD:
    """
    Perform matrix factorization using the optimized SVD algorithm from the Surprise library.
    
    :param feedback_matrix: A Pandas DataFrame representing the user-movie feedback matrix.
    :param n_factors: Number of latent factors to use in the SVD model.
    :param n_epochs: Number of epochs for training.
    :param lr_all: Learning rate for all parameters.
    :param reg_all: Regularization term for all parameters.
    :param test_size: Proportion of the data to use for testing.
    :return: Trained SVD model.
    """
    # Convert the feedback matrix to a long format DataFrame
    data = feedback_matrix.stack().reset_index()
    data.columns = ['userID', 'movie_id', 'score']
    data = data[data['score'] > 0]  # Filter out unrated entries

    # Define a Reader object with the appropriate rating scale
    reader = Reader(rating_scale=(data['score'].min(), data['score'].max()))

    # Load the data into a Surprise dataset
    dataset = Dataset.load_from_df(data[['userID', 'movie_id', 'score']], reader)

    # Split the dataset into training and testing sets
    trainset, testset = train_test_split(dataset, test_size=test_size)

    # Configure and create the SVD model with optimized hyperparameters
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

    # Train the algorithm on the training set
    algo.fit(trainset)

    return algo


def predict_ratings(algo: SVD, user_id: int, movie_titles: Dict[int, str], top_n: int = 3) -> List[str]:
    """
    Generate predictions for a specific user using the trained SVD model.
    
    :param algo: Trained SVD model from Surprise.
    :param user_id: ID of the user for whom to generate recommendations.
    :param movie_titles: Dictionary mapping movie IDs to titles.
    :param top_n: Number of top recommendations to return.
    :return: List of recommended movie titles.
    """
    all_movie_ids = movie_titles.keys()
    predictions = [(movie_id, algo.predict(user_id, movie_id).est) for movie_id in all_movie_ids]

    # Sort predictions by estimated score in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get the top N movie titles
    top_movies = [movie_titles[movie_id] for movie_id, _ in predictions[:top_n]]
    return top_movies


def calculate_user_similarity(U: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between users.
    :param U: User matrix (embedding matrix) where each row represents a user vector.
    :return: A user similarity matrix, where each entry [i][j] indicates the similarity between user i and user j.
    """
    return cosine_similarity(U)


def recommend_movies(user_id: int, algo: SVD, movie_titles: Dict[int, str], top_n: int = 3) -> List[str]:
    """
    Generate movie recommendations for a target user using the trained SVD model.
    :param user_id: ID of the target user.
    :param algo: Trained SVD model from the Surprise library.
    :param movie_titles: Dictionary mapping movie IDs to titles.
    :param top_n: Number of top recommendations to return.
    :return: List of recommended movie titles.
    """
    recommend_movies = predict_ratings(algo, user_id, movie_titles, top_n)
    return recommend_movies
