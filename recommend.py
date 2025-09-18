"""
Movie Recommendation Engine
Provides multiple recommendation algorithms with sub-second response times through pre-computation.
"""

import sqlite3
import numpy as np
import pandas as pd
import logging
import json
import argparse
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class RecommendationEngine:
    """Advanced movie recommendation engine with multiple algorithms and caching."""
    
    def __init__(self, db_path: str, cache_dir: str = "cache/"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Recommendation models
        self.content_model = None
        self.collaborative_model = None
        self.hybrid_weights = {'content': 0.3, 'collaborative': 0.7}
        
        # Cache for fast lookups
        self.movie_features_cache = {}
        self.user_profiles_cache = {}
        self.precomputed_recommendations = {}
        
        # Performance tracking
        self.recommendation_stats = {
            'content_based_calls': 0,
            'collaborative_calls': 0,
            'hybrid_calls': 0,
            'cache_hits': 0,
            'total_calls': 0
        }
    
    def initialize_models(self, force_retrain: bool = False):
        """Initialize and train recommendation models."""
        self.logger.info("Initializing recommendation models")
        
        # Check for cached models
        content_model_path = self.cache_dir / "content_model.pkl"
        collaborative_model_path = self.cache_dir / "collaborative_model.pkl"
        
        if not force_retrain and content_model_path.exists() and collaborative_model_path.exists():
            self.logger.info("Loading cached models")
            try:
                with open(content_model_path, 'rb') as f:
                    self.content_model = pickle.load(f)
                with open(collaborative_model_path, 'rb') as f:
                    self.collaborative_model = pickle.load(f)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to load cached models: {e}")
        
        # Train new models
        self.logger.info("Training new recommendation models")
        
        try:
            # Train content-based model
            self._train_content_model()
            
            # Train collaborative filtering model
            self._train_collaborative_model()
            
            # Cache models
            with open(content_model_path, 'wb') as f:
                pickle.dump(self.content_model, f)
            with open(collaborative_model_path, 'wb') as f:
                pickle.dump(self.collaborative_model, f)
            
            self.logger.info("Models trained and cached successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            return False
    
    def _train_content_model(self):
        """Train content-based recommendation model."""
        self.logger.info("Training content-based model")
        
        # Load movie features from database
        query = """
        SELECT 
            mf.movie_id,
            mf.feature_data,
            m.english_title,
            m.release_year,
            m.imdb_rating,
            m.popularity_score
        FROM movie_features mf
        JOIN movie m ON mf.movie_id = m.id
        WHERE mf.feature_data IS NOT NULL
        """
        
        with sqlite3.connect(self.db_path) as conn:
            movie_features_df = pd.read_sql_query(query, conn)
        
        if movie_features_df.empty:
            self.logger.warning("No movie features found, using basic features")
            self._create_basic_content_features()
            return
        
        # Process feature data
        feature_vectors = []
        movie_ids = []
        
        for _, row in movie_features_df.iterrows():
            try:
                features = json.loads(row['feature_data'])
                
                # Extract numeric features for similarity calculation
                feature_vector = []
                
                # Genre features (one-hot encoded)
                genre_features = [v for k, v in features.items() if k.startswith('genre_onehot_')]
                feature_vector.extend(genre_features)
                
                # Numeric features
                numeric_features = [
                    features.get('Runtime', 0) / 200.0,  # Normalize runtime
                    features.get('imdb_rating', 0) / 10.0,  # Normalize rating
                    features.get('popularity_score', 0) / 1000.0,  # Normalize popularity
                    features.get('release_year', 2000) / 2024.0  # Normalize year
                ]
                feature_vector.extend(numeric_features)
                
                if feature_vector:
                    feature_vectors.append(feature_vector)
                    movie_ids.append(row['movie_id'])
                    
            except Exception as e:
                self.logger.warning(f"Error processing features for movie {row['movie_id']}: {e}")
        
        if feature_vectors:
            self.content_model = {
                'feature_matrix': np.array(feature_vectors),
                'movie_ids': movie_ids,
                'similarity_matrix': cosine_similarity(feature_vectors)
            }
            self.logger.info(f"Content model trained with {len(movie_ids)} movies")
        else:
            self.logger.error("No valid feature vectors created")
    
    def _create_basic_content_features(self):
        """Create basic content features from movie data."""
        query = """
        SELECT 
            m.id as movie_id,
            m.english_title,
            m.release_year,
            m.runtime,
            m.imdb_rating,
            m.popularity_score,
            GROUP_CONCAT(g.name, '|') as genres
        FROM movie m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genre g ON mg.genre_id = g.id
        GROUP BY m.id
        """
        
        with sqlite3.connect(self.db_path) as conn:
            movies_df = pd.read_sql_query(query, conn)
        
        # Create simple feature vectors
        feature_vectors = []
        movie_ids = []
        
        for _, row in movies_df.iterrows():
            feature_vector = [
                row['runtime'] / 200.0 if row['runtime'] else 1.5,
                row['imdb_rating'] / 10.0 if row['imdb_rating'] else 0.5,
                row['popularity_score'] / 1000.0 if row['popularity_score'] else 0.1,
                row['release_year'] / 2024.0 if row['release_year'] else 0.8
            ]
            
            feature_vectors.append(feature_vector)
            movie_ids.append(row['movie_id'])
        
        if feature_vectors:
            self.content_model = {
                'feature_matrix': np.array(feature_vectors),
                'movie_ids': movie_ids,
                'similarity_matrix': cosine_similarity(feature_vectors)
            }
            self.logger.info(f"Basic content model created with {len(movie_ids)} movies")
    
    def _train_collaborative_model(self):
        """Train collaborative filtering model using SVD."""
        self.logger.info("Training collaborative filtering model")
        
        # Load ratings data
        query = """
        SELECT user_id, movie_id, rating
        FROM user_rating
        WHERE rating BETWEEN 1.0 AND 10.0
        """
        
        with sqlite3.connect(self.db_path) as conn:
            ratings_df = pd.read_sql_query(query, conn)
        
        if ratings_df.empty:
            self.logger.error("No ratings data found for collaborative filtering")
            return
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        
        # Train-test split
        trainset, testset = train_test_split(data, test_size=0.1, random_state=42)
        
        # Train SVD model
        self.collaborative_model = SVD(
            n_factors=100,
            n_epochs=30,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )
        
        self.collaborative_model.fit(trainset)
        
        # Evaluate model
        predictions = self.collaborative_model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        
        self.logger.info(f"Collaborative model trained with RMSE: {rmse:.4f}")
    
    def get_content_recommendations(self, user_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get content-based recommendations for a user."""
        if not self.content_model:
            self.logger.error("Content model not initialized")
            return []
        
        self.recommendation_stats['content_based_calls'] += 1
        self.recommendation_stats['total_calls'] += 1
        
        try:
            # Get user's rating history
            user_ratings = self._get_user_ratings(user_id)
            
            if not user_ratings:
                # Cold start - recommend popular movies
                return self._get_popular_movies(top_n)
            
            # Calculate user preference vector based on rated movies
            user_vector = self._calculate_user_content_profile(user_ratings)
            
            if user_vector is None:
                return self._get_popular_movies(top_n)
            
            # Calculate similarities with all movies
            similarities = cosine_similarity([user_vector], self.content_model['feature_matrix'])[0]
            
            # Get movies user hasn't rated
            rated_movie_ids = {r['movie_id'] for r in user_ratings}
            recommendations = []
            
            # Sort by similarity and filter out rated movies
            movie_similarity_pairs = list(zip(self.content_model['movie_ids'], similarities))
            movie_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for movie_id, similarity in movie_similarity_pairs:
                if movie_id not in rated_movie_ids and len(recommendations) < top_n:
                    movie_info = self._get_movie_info(movie_id)
                    if movie_info:
                        recommendations.append({
                            'movie_id': movie_id,
                            'title': movie_info['title'],
                            'prediction_score': float(similarity),
                            'reason': 'content_based',
                            'explanation': f"Similar to movies you rated highly"
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Content-based recommendation failed: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations for a user."""
        if not self.collaborative_model:
            self.logger.error("Collaborative model not initialized")
            return []
        
        self.recommendation_stats['collaborative_calls'] += 1
        self.recommendation_stats['total_calls'] += 1
        
        try:
            # Get all movies
            all_movies = self._get_all_movies()
            
            # Get user's rating history
            user_ratings = self._get_user_ratings(user_id)
            rated_movie_ids = {r['movie_id'] for r in user_ratings}
            
            # Generate predictions for unrated movies
            predictions = []
            
            for movie_id in all_movies:
                if movie_id not in rated_movie_ids:
                    prediction = self.collaborative_model.predict(user_id, movie_id)
                    predictions.append({
                        'movie_id': movie_id,
                        'predicted_rating': prediction.est,
                        'confidence': prediction.details.get('was_impossible', False)
                    })
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            # Format recommendations
            recommendations = []
            for pred in predictions[:top_n]:
                movie_info = self._get_movie_info(pred['movie_id'])
                if movie_info:
                    recommendations.append({
                        'movie_id': pred['movie_id'],
                        'title': movie_info['title'],
                        'prediction_score': pred['predicted_rating'],
                        'reason': 'collaborative',
                        'explanation': f"Users with similar tastes enjoyed this movie"
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Collaborative recommendation failed: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining content and collaborative filtering."""
        self.recommendation_stats['hybrid_calls'] += 1
        self.recommendation_stats['total_calls'] += 1
        
        try:
            # Get recommendations from both models
            content_recs = self.get_content_recommendations(user_id, top_n * 2)
            collaborative_recs = self.get_collaborative_recommendations(user_id, top_n * 2)
            
            # Combine and weight recommendations
            combined_scores = {}
            
            # Add content-based scores
            for rec in content_recs:
                movie_id = rec['movie_id']
                combined_scores[movie_id] = {
                    'title': rec['title'],
                    'content_score': rec['prediction_score'] * self.hybrid_weights['content'],
                    'collaborative_score': 0.0,
                    'total_score': 0.0
                }
            
            # Add collaborative scores
            for rec in collaborative_recs:
                movie_id = rec['movie_id']
                if movie_id in combined_scores:
                    combined_scores[movie_id]['collaborative_score'] = rec['prediction_score'] * self.hybrid_weights['collaborative']
                else:
                    combined_scores[movie_id] = {
                        'title': rec['title'],
                        'content_score': 0.0,
                        'collaborative_score': rec['prediction_score'] * self.hybrid_weights['collaborative'],
                        'total_score': 0.0
                    }
            
            # Calculate total scores
            for movie_id, scores in combined_scores.items():
                scores['total_score'] = scores['content_score'] + scores['collaborative_score']
            
            # Sort by total score
            sorted_recommendations = sorted(
                combined_scores.items(),
                key=lambda x: x[1]['total_score'],
                reverse=True
            )
            
            # Format final recommendations
            recommendations = []
            for movie_id, scores in sorted_recommendations[:top_n]:
                recommendations.append({
                    'movie_id': movie_id,
                    'title': scores['title'],
                    'prediction_score': scores['total_score'],
                    'content_score': scores['content_score'],
                    'collaborative_score': scores['collaborative_score'],
                    'reason': 'hybrid',
                    'explanation': f"Combines content similarity and user behavior patterns"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Hybrid recommendation failed: {e}")
            return []
    
    def precompute_recommendations(self, top_n: int = 50):
        """Precompute recommendations for all users to enable sub-second responses."""
        self.logger.info("Starting recommendation precomputation")
        
        try:
            # Get all users with sufficient ratings
            query = """
            SELECT user_id, COUNT(*) as rating_count
            FROM user_rating
            GROUP BY user_id
            HAVING COUNT(*) >= 5
            ORDER BY rating_count DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                users_df = pd.read_sql_query(query, conn)
            
            total_users = len(users_df)
            self.logger.info(f"Precomputing recommendations for {total_users} users")
            
            # Store precomputed recommendations in database
            self._clear_recommendation_cache()
            
            batch_size = 100
            for i in range(0, total_users, batch_size):
                batch_users = users_df.iloc[i:i + batch_size]
                
                for _, user_row in batch_users.iterrows():
                    user_id = user_row['user_id']
                    
                    try:
                        # Generate hybrid recommendations
                        recommendations = self.get_hybrid_recommendations(user_id, top_n)
                        
                        # Store in database
                        self._store_precomputed_recommendations(user_id, recommendations)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to precompute recommendations for user {user_id}: {e}")
                
                # Log progress
                progress = min(i + batch_size, total_users)
                self.logger.info(f"Precomputed recommendations: {progress}/{total_users} users")
            
            self.logger.info("Recommendation precomputation completed")
            
        except Exception as e:
            self.logger.error(f"Precomputation failed: {e}")
    
    def get_recommendations(self, user_id: int, top_n: int = 10, 
                          algorithm: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Get movie recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to return
            algorithm: 'content', 'collaborative', or 'hybrid'
            
        Returns:
            List of movie recommendations
        """
        # Check for cached recommendations first
        cached_recs = self._get_cached_recommendations(user_id, algorithm)
        if cached_recs:
            self.recommendation_stats['cache_hits'] += 1
            return cached_recs[:top_n]
        
        # Generate recommendations based on algorithm
        if algorithm == 'content':
            recommendations = self.get_content_recommendations(user_id, top_n)
        elif algorithm == 'collaborative':
            recommendations = self.get_collaborative_recommendations(user_id, top_n)
        else:  # hybrid
            recommendations = self.get_hybrid_recommendations(user_id, top_n)
        
        return recommendations
    
    def _get_user_ratings(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's rating history."""
        query = """
        SELECT movie_id, rating, rating_date
        FROM user_rating
        WHERE user_id = ?
        ORDER BY rating_date DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
        
        return [
            {
                'movie_id': row[0],
                'rating': row[1],
                'rating_date': row[2]
            }
            for row in rows
        ]
    
    def _get_movie_info(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Get basic movie information."""
        query = """
        SELECT english_title, release_year, imdb_rating
        FROM movie
        WHERE id = ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (movie_id,))
            row = cursor.fetchone()
        
        if row:
            return {
                'title': row[0],
                'year': row[1],
                'rating': row[2]
            }
        return None
    
    def _get_all_movies(self) -> List[int]:
        """Get list of all movie IDs."""
        query = "SELECT id FROM movie"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        
        return [row[0] for row in rows]
    
    def _get_popular_movies(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get popular movies for cold start users."""
        query = """
        SELECT 
            m.id,
            m.english_title,
            m.imdb_rating,
            COUNT(ur.user_id) as rating_count,
            AVG(ur.rating) as avg_rating
        FROM movie m
        LEFT JOIN user_rating ur ON m.id = ur.movie_id
        WHERE m.imdb_rating IS NOT NULL
        GROUP BY m.id
        HAVING COUNT(ur.user_id) >= 10
        ORDER BY (COUNT(ur.user_id) * AVG(ur.rating)) DESC
        LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()
        
        recommendations = []
        for row in rows:
            recommendations.append({
                'movie_id': row[0],
                'title': row[1],
                'prediction_score': row[4] or 0.0,  # avg_rating
                'reason': 'popular',
                'explanation': f"Popular movie with {row[3]} ratings"
            })
        
        return recommendations
    
    def _calculate_user_content_profile(self, user_ratings: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Calculate user's content preference profile."""
        if not self.content_model or not user_ratings:
            return None
        
        # Get feature vectors for movies user has rated
        rated_movie_vectors = []
        rating_weights = []
        
        for rating_info in user_ratings:
            movie_id = rating_info['movie_id']
            rating = rating_info['rating']
            
            # Find movie in content model
            try:
                movie_index = self.content_model['movie_ids'].index(movie_id)
                movie_vector = self.content_model['feature_matrix'][movie_index]
                
                # Weight by rating (higher ratings have more influence)
                weight = (rating - 5.5) / 4.5  # Normalize to [-1, 1] range
                rated_movie_vectors.append(movie_vector * weight)
                rating_weights.append(abs(weight))
                
            except ValueError:
                # Movie not in content model
                continue
        
        if not rated_movie_vectors:
            return None
        
        # Calculate weighted average preference vector
        weighted_vectors = np.array(rated_movie_vectors)
        weights = np.array(rating_weights)
        
        if np.sum(weights) == 0:
            return np.mean(weighted_vectors, axis=0)
        else:
            return np.average(weighted_vectors, axis=0, weights=weights)
    
    def _get_cached_recommendations(self, user_id: int, algorithm: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendations from database."""
        query = """
        SELECT movie_id, prediction_score, explanation
        FROM user_recommendations
        WHERE user_id = ? AND recommendation_type = ? AND expires_at > ?
        ORDER BY rank_position
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id, algorithm, datetime.now().isoformat()))
            rows = cursor.fetchall()
        
        if rows:
            recommendations = []
            for row in rows:
                movie_info = self._get_movie_info(row[0])
                if movie_info:
                    recommendations.append({
                        'movie_id': row[0],
                        'title': movie_info['title'],
                        'prediction_score': row[1],
                        'reason': algorithm,
                        'explanation': row[2] or f"{algorithm} recommendation"
                    })
            return recommendations
        
        return None
    
    def _store_precomputed_recommendations(self, user_id: int, recommendations: List[Dict[str, Any]]):
        """Store precomputed recommendations in database."""
        if not recommendations:
            return
        
        query = """
        INSERT OR REPLACE INTO user_recommendations 
        (user_id, movie_id, prediction_score, recommendation_type, rank_position, explanation, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        # Set expiration to 24 hours from now
        expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for i, rec in enumerate(recommendations):
                cursor.execute(query, (
                    user_id,
                    rec['movie_id'],
                    rec['prediction_score'],
                    'hybrid',
                    i + 1,
                    rec['explanation'],
                    expires_at
                ))
            
            conn.commit()
    
    def _clear_recommendation_cache(self):
        """Clear expired recommendations from cache."""
        query = "DELETE FROM user_recommendations WHERE expires_at <= ?"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (datetime.now().isoformat(),))
            conn.commit()
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get recommendation system statistics."""
        return {
            'model_status': {
                'content_model_loaded': self.content_model is not None,
                'collaborative_model_loaded': self.collaborative_model is not None,
            },
            'performance_stats': self.recommendation_stats.copy(),
            'cache_hit_rate': self.recommendation_stats['cache_hits'] / max(1, self.recommendation_stats['total_calls'])
        }


def main():
    """Main entry point for recommendation system."""
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--db', required=True, help='Path to database file')
    parser.add_argument('--user-id', type=int, help='User ID for recommendations')
    parser.add_argument('--algorithm', choices=['content', 'collaborative', 'hybrid'], 
                       default='hybrid', help='Recommendation algorithm')
    parser.add_argument('--top-n', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--precompute', action='store_true', help='Precompute recommendations for all users')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize recommendation engine
    engine = RecommendationEngine(args.db)
    
    # Initialize models
    if not engine.initialize_models(force_retrain=args.retrain):
        logger.error("Failed to initialize models")
        return
    
    if args.precompute:
        # Precompute recommendations
        engine.precompute_recommendations()
        logger.info("Recommendation precomputation completed")
        
    elif args.user_id:
        # Generate recommendations for specific user
        start_time = datetime.now()
        
        recommendations = engine.get_recommendations(
            user_id=args.user_id,
            top_n=args.top_n,
            algorithm=args.algorithm
        )
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\n=== Top {len(recommendations)} {args.algorithm.title()} Recommendations for User {args.user_id} ===")
        print(f"Response time: {response_time:.3f} seconds\n")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']}")
                print(f"   Score: {rec['prediction_score']:.3f}")
                print(f"   Reason: {rec['explanation']}")
                print()
        else:
            print("No recommendations found")
        
        # Display stats
        stats = engine.get_recommendation_stats()
        print(f"Recommendation Stats: {stats}")
        
    else:
        print("Please specify --user-id for recommendations or --precompute for batch processing")


if __name__ == '__main__':
    main()