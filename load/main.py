"""
Filename: main.py
Author: Julia Jiang
Date: 2024-11-09
Description:
    Command-line interface for the collaborative filtering movie recommendation system with profiling support.
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

import argparse
import cProfile
import pstats
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transform.recommendation import load_feedback_matrix, load_movie_titles, matrix_factorization, recommend_movies

def log_recommendations(user_id: int, recommendations: list, top_n: int) -> None:
    """
    Log movie recommendations to the results file.
    :param user_id: ID of the user for whom recommendations were generated.
    :param recommendations: List of recommended movie titles.
    :param top_n: Number of recommendations requested.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "result.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] User ID: {user_id}, Top N: {top_n}, Recommendations: {', '.join(recommendations)}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def main(database_path: str, user_id: int, top_n: int) -> None:
    """
    Main function to generate and print movie recommendations for a given user.
    :param database_path: Path to the SQLite database file.
    :param user_id: ID of the user for whom to generate recommendations.
    :param top_n: Number of top recommendations to return.
    """
    # Load feedback matrix and movie titles
    feedback_matrix = load_feedback_matrix(database_path)
    movie_titles = load_movie_titles(database_path)

    # Train the SVD model using matrix factorization
    algo = matrix_factorization(feedback_matrix)

    # Generate and print recommendations for the given user ID
    recommendations = recommend_movies(user_id, algo, movie_titles, top_n)
    print(f"Recommended Movies for User {user_id}: {', '.join(recommendations)}")

    # Log the recommendations to file
    log_recommendations(user_id, recommendations, top_n)

if __name__ == "__main__":
    # Use relative path to database
    database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract", "data_sources", "movies.db")

    parser = argparse.ArgumentParser(description="Movie recommendation system using collaborative filtering with profiling.")
    parser.add_argument('user_id', type=int, help="User ID for whom to generate movie recommendations")
    parser.add_argument('--top_n', type=int, default=3, help="Number of top recommendations to return (default is 3)")
    parser.add_argument('--profile', action='store_true', help="Enable profiling with cProfile")

    args = parser.parse_args()

    if args.profile:
        # Profile the main function
        profiler = cProfile.Profile()
        profiler.enable()
        main(database_path, args.user_id, args.top_n)
        profiler.disable()

        # Print the profiling results
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Print top 20 lines
    else:
        main(database_path, args.user_id, args.top_n)