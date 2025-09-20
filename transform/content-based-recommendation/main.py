"""
Filename: main.py
Author: Julia Jiang
Date: 2024-10-27
Description: Main Code
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
import sqlite3


def main():
    # parser = argparse.ArgumentParser(description="Movie Recommender System")
    # parser.add_argument("do_file", help="Path to the data base.")
    # parser.add_argument('-t', '--test', help=)
    """Testing"""
    #------baseline
    baseline_test.create_baseline(20)
    #------good
    pre_selected_users = [20, 70]
    MRS_good.good_test(pre_selected_users)


if __name__ == "__main__":
    main()
    # Connect to the database
    conn = sqlite3.connect('movies_copy.db')
    ## ------------- excellent part ----------------------------------
    # Select a random user or specify a particular user ID for testing
    user_id = recommendation.get_random_user(conn)  # You can also replace this with a specific user ID, e.g., 123
    
    # Define test scenarios for varying numbers of rated and unrated movies
    # num_rated_movies_variants = [1, 10, 50, 100, 200, 400, 800] #, 1600, 3200, 6400, 12800, 25600, 56200, 100000, 1000000, 10000000]   # Different sizes of rated movies
    # num_unrated_movies_variants = [1, 10, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 56200, 100000, 1000000, 10000000]  # Different sizes of unrated movies
    
    # Call the stress test function
    # print(f"Starting stress tests for User ID: {user_id}")
    # stress_test_recommender_system(
    #     user_id=user_id, 
    #     conn=conn, 
    #     num_rated_movies_variants=num_rated_movies_variants, 
    #     num_unrated_movies_variants=num_unrated_movies_variants
    # )

    # Close the database connection
    conn.close()