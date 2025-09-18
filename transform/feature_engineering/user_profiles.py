"""
User Profile Feature Engineering Module
Creates comprehensive user preference vectors and behavioral features.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class UserProfileBuilder:
    """Advanced user profile and preference vector generation."""
    
    def __init__(self, min_ratings_per_user: int = 5, profile_dimensions: int = 50):
        self.logger = logging.getLogger(__name__)
        self.min_ratings_per_user = min_ratings_per_user
        self.profile_dimensions = profile_dimensions
        
        # Fitted transformers
        self.rating_scaler = None
        self.feature_scaler = None
        self.svd_model = None
        
        # User statistics and profiles
        self.user_statistics = {}
        self.user_profiles = {}
        self.genre_preferences = {}
        
    def build_user_profiles(self, ratings_df: pd.DataFrame, 
                           movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build comprehensive user profiles from ratings and movie data.
        
        Args:
            ratings_df: User ratings DataFrame
            movies_df: Movies DataFrame with genre features
            
        Returns:
            Tuple of (user profiles DataFrame, profile metadata)
        """
        self.logger.info(f"Building user profiles from {len(ratings_df)} ratings and {len(movies_df)} movies")
        
        # Calculate basic user statistics
        self.user_statistics = self._calculate_user_statistics(ratings_df)
        
        # Filter users with minimum ratings
        active_users = self._filter_active_users(ratings_df)
        self.logger.info(f"Found {len(active_users)} active users with >= {self.min_ratings_per_user} ratings")
        
        # Build various types of user profiles
        profiles_data = {}
        
        # 1. Rating-based profiles
        rating_profiles = self._build_rating_profiles(ratings_df, active_users)
        profiles_data.update(rating_profiles)
        
        # 2. Genre preference profiles
        genre_profiles = self._build_genre_profiles(ratings_df, movies_df, active_users)
        profiles_data.update(genre_profiles)
        
        # 3. Behavioral profiles
        behavioral_profiles = self._build_behavioral_profiles(ratings_df, active_users)
        profiles_data.update(behavioral_profiles)
        
        # 4. Temporal profiles
        temporal_profiles = self._build_temporal_profiles(ratings_df, active_users)
        profiles_data.update(temporal_profiles)
        
        # 5. Similarity-based profiles (using matrix factorization)
        similarity_profiles = self._build_similarity_profiles(ratings_df, movies_df, active_users)
        profiles_data.update(similarity_profiles)
        
        # Create unified user profiles DataFrame
        user_profiles_df = pd.DataFrame.from_dict(profiles_data, orient='index')
        user_profiles_df.index.name = 'UserID'
        user_profiles_df = user_profiles_df.reset_index()
        
        # Generate metadata
        profile_metadata = {
            'total_users': len(self.user_statistics),
            'active_users': len(active_users),
            'profile_features': len(user_profiles_df.columns) - 1,  # -1 for UserID
            'feature_categories': {
                'rating_features': len([c for c in user_profiles_df.columns if 'rating_' in c]),
                'genre_features': len([c for c in user_profiles_df.columns if 'genre_' in c]),
                'behavioral_features': len([c for c in user_profiles_df.columns if 'behavior_' in c]),
                'temporal_features': len([c for c in user_profiles_df.columns if 'temporal_' in c]),
                'similarity_features': len([c for c in user_profiles_df.columns if 'similarity_' in c])
            },
            'user_statistics': self.user_statistics
        }
        
        self.logger.info(f"User profiles completed: {len(user_profiles_df)} users, {len(user_profiles_df.columns)-1} features")
        return user_profiles_df, profile_metadata
    
    def _calculate_user_statistics(self, ratings_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive user statistics."""
        user_stats = {}
        
        for user_id, user_ratings in ratings_df.groupby('UserID'):
            stats = {
                'total_ratings': len(user_ratings),
                'avg_rating': user_ratings['Rating'].mean(),
                'rating_std': user_ratings['Rating'].std(),
                'min_rating': user_ratings['Rating'].min(),
                'max_rating': user_ratings['Rating'].max(),
                'rating_range': user_ratings['Rating'].max() - user_ratings['Rating'].min(),
            }
            
            # Rating distribution
            rating_counts = user_ratings['Rating'].value_counts().sort_index()
            for rating in range(1, 11):  # Assuming 1-10 scale
                stats[f'rating_{rating}_count'] = rating_counts.get(rating, 0)
                stats[f'rating_{rating}_pct'] = rating_counts.get(rating, 0) / len(user_ratings)
            
            # Temporal statistics if date available
            if 'Date' in user_ratings.columns:
                try:
                    dates = pd.to_datetime(user_ratings['Date'])
                    stats['date_span_days'] = (dates.max() - dates.min()).days
                    stats['avg_ratings_per_month'] = len(user_ratings) / max(1, stats['date_span_days'] / 30)
                except:
                    pass
            
            user_stats[user_id] = stats
        
        return user_stats
    
    def _filter_active_users(self, ratings_df: pd.DataFrame) -> List[int]:
        """Filter users with minimum number of ratings."""
        user_counts = ratings_df['UserID'].value_counts()
        active_users = user_counts[user_counts >= self.min_ratings_per_user].index.tolist()
        return active_users
    
    def _build_rating_profiles(self, ratings_df: pd.DataFrame, active_users: List[int]) -> Dict[str, Dict[str, float]]:
        """Build rating-based user profiles."""
        profiles = {}
        
        for user_id in active_users:
            if user_id not in self.user_statistics:
                continue
                
            user_stats = self.user_statistics[user_id]
            profile = {}
            
            # Basic rating statistics
            profile['rating_avg'] = user_stats['avg_rating']
            profile['rating_std'] = user_stats['rating_std']
            profile['rating_range'] = user_stats['rating_range']
            profile['rating_count'] = user_stats['total_ratings']
            
            # Rating tendency (harsh vs generous)
            overall_avg = np.mean([stats['avg_rating'] for stats in self.user_statistics.values()])
            profile['rating_tendency'] = user_stats['avg_rating'] - overall_avg
            
            # Rating consistency
            profile['rating_consistency'] = 1 / (1 + user_stats['rating_std'])  # Higher = more consistent
            
            # Rating extremity (tendency to use extreme ratings)
            high_ratings = sum(user_stats[f'rating_{i}_count'] for i in range(8, 11))  # 8-10
            low_ratings = sum(user_stats[f'rating_{i}_count'] for i in range(1, 4))   # 1-3
            total_ratings = user_stats['total_ratings']
            profile['rating_extremity'] = (high_ratings + low_ratings) / total_ratings
            
            # Rating positivity
            positive_ratings = sum(user_stats[f'rating_{i}_count'] for i in range(6, 11))  # 6-10
            profile['rating_positivity'] = positive_ratings / total_ratings
            
            profiles[user_id] = profile
        
        return {f'rating_{k}': {user: profiles[user][k] for user in profiles} 
                for k in profiles[list(profiles.keys())[0]].keys()}
    
    def _build_genre_profiles(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                             active_users: List[int]) -> Dict[str, Dict[str, float]]:
        """Build genre preference profiles."""
        profiles = {}
        
        # Merge ratings with movie genre information
        genre_columns = [col for col in movies_df.columns if col.startswith('genre_onehot_')]
        
        if not genre_columns:
            self.logger.warning("No genre one-hot columns found in movies data")
            return {}
        
        # Create ratings-movies merge
        ratings_with_genres = ratings_df.merge(
            movies_df[['MovieID'] + genre_columns], 
            on='MovieID', 
            how='left'
        )
        
        for user_id in active_users:
            user_ratings = ratings_with_genres[ratings_with_genres['UserID'] == user_id]
            if len(user_ratings) == 0:
                continue
            
            profile = {}
            
            # Genre preference scores (weighted by ratings)
            for genre_col in genre_columns:
                genre_name = genre_col.replace('genre_onehot_', '')
                
                # Movies of this genre that user rated
                genre_movies = user_ratings[user_ratings[genre_col] == 1]
                
                if len(genre_movies) > 0:
                    # Average rating for this genre
                    profile[f'genre_avg_{genre_name}'] = genre_movies['Rating'].mean()
                    
                    # Count of movies rated in this genre
                    profile[f'genre_count_{genre_name}'] = len(genre_movies)
                    
                    # Preference score (rating weighted by frequency)
                    total_movies = len(user_ratings)
                    genre_frequency = len(genre_movies) / total_movies
                    avg_rating = genre_movies['Rating'].mean()
                    preference_score = (avg_rating - 5.5) * genre_frequency  # Center around 5.5
                    profile[f'genre_pref_{genre_name}'] = preference_score
                else:
                    profile[f'genre_avg_{genre_name}'] = 0.0
                    profile[f'genre_count_{genre_name}'] = 0
                    profile[f'genre_pref_{genre_name}'] = 0.0
            
            # Genre diversity
            genres_rated = sum(1 for col in genre_columns if user_ratings[col].sum() > 0)
            profile['genre_diversity'] = genres_rated / len(genre_columns)
            
            profiles[user_id] = profile
        
        # Transpose for output format
        all_keys = set()
        for profile in profiles.values():
            all_keys.update(profile.keys())
        
        return {key: {user: profiles[user].get(key, 0.0) for user in profiles} 
                for key in all_keys}
    
    def _build_behavioral_profiles(self, ratings_df: pd.DataFrame, 
                                  active_users: List[int]) -> Dict[str, Dict[str, float]]:
        """Build behavioral user profiles."""
        profiles = {}
        
        for user_id in active_users:
            user_ratings = ratings_df[ratings_df['UserID'] == user_id]
            profile = {}
            
            # Rating frequency patterns
            if 'Date' in user_ratings.columns:
                try:
                    dates = pd.to_datetime(user_ratings['Date'])
                    
                    # Temporal patterns
                    profile['behavior_rating_frequency'] = len(user_ratings) / max(1, (dates.max() - dates.min()).days / 30)
                    
                    # Day of week patterns (if we have enough data)
                    if len(dates) >= 7:
                        dow_counts = dates.dt.dayofweek.value_counts()
                        profile['behavior_weekend_preference'] = (dow_counts.get(5, 0) + dow_counts.get(6, 0)) / len(dates)
                    else:
                        profile['behavior_weekend_preference'] = 0.0
                        
                except:
                    profile['behavior_rating_frequency'] = 0.0
                    profile['behavior_weekend_preference'] = 0.0
            else:
                profile['behavior_rating_frequency'] = 0.0
                profile['behavior_weekend_preference'] = 0.0
            
            # Rating pattern consistency
            ratings = user_ratings['Rating'].values
            if len(ratings) > 1:
                # Consecutive rating differences
                rating_diffs = np.abs(np.diff(ratings))
                profile['behavior_rating_volatility'] = np.std(rating_diffs)
                
                # Trend in ratings over time
                correlation = np.corrcoef(range(len(ratings)), ratings)[0, 1]
                profile['behavior_rating_trend'] = correlation if not np.isnan(correlation) else 0.0
            else:
                profile['behavior_rating_volatility'] = 0.0
                profile['behavior_rating_trend'] = 0.0
            
            # Discovery behavior (preference for popular vs niche content)
            # This would require movie popularity data - placeholder for now
            profile['behavior_mainstream_preference'] = 0.5  # Neutral
            
            profiles[user_id] = profile
        
        return {f'behavior_{k.replace("behavior_", "")}': {user: profiles[user][k] for user in profiles} 
                for k in profiles[list(profiles.keys())[0]].keys()}
    
    def _build_temporal_profiles(self, ratings_df: pd.DataFrame, 
                                active_users: List[int]) -> Dict[str, Dict[str, float]]:
        """Build temporal behavior profiles."""
        profiles = {}
        
        if 'Date' not in ratings_df.columns:
            return {}
        
        for user_id in active_users:
            user_ratings = ratings_df[ratings_df['UserID'] == user_id]
            profile = {}
            
            try:
                dates = pd.to_datetime(user_ratings['Date'])
                ratings = user_ratings['Rating'].values
                
                # Temporal statistics
                date_span = (dates.max() - dates.min()).days
                profile['temporal_activity_span'] = date_span
                profile['temporal_activity_density'] = len(user_ratings) / max(1, date_span)
                
                # Rating evolution over time
                if len(dates) >= 5:  # Need minimum data points
                    sorted_indices = dates.argsort()
                    sorted_ratings = ratings[sorted_indices]
                    
                    # Early vs late rating averages
                    early_ratings = sorted_ratings[:len(sorted_ratings)//3]
                    late_ratings = sorted_ratings[-len(sorted_ratings)//3:]
                    
                    profile['temporal_rating_evolution'] = np.mean(late_ratings) - np.mean(early_ratings)
                    
                    # Rating trend slope
                    time_numeric = np.arange(len(sorted_ratings))
                    slope = np.polyfit(time_numeric, sorted_ratings, 1)[0]
                    profile['temporal_rating_slope'] = slope
                else:
                    profile['temporal_rating_evolution'] = 0.0
                    profile['temporal_rating_slope'] = 0.0
                
                # Seasonal patterns (month-based)
                if len(dates) >= 12:
                    monthly_ratings = {}
                    for month in range(1, 13):
                        month_ratings = ratings[dates.dt.month == month]
                        monthly_ratings[month] = np.mean(month_ratings) if len(month_ratings) > 0 else 5.5
                    
                    profile['temporal_seasonal_variance'] = np.std(list(monthly_ratings.values()))
                else:
                    profile['temporal_seasonal_variance'] = 0.0
                
            except Exception as e:
                self.logger.warning(f"Error processing temporal data for user {user_id}: {e}")
                profile = {
                    'temporal_activity_span': 0.0,
                    'temporal_activity_density': 0.0,
                    'temporal_rating_evolution': 0.0,
                    'temporal_rating_slope': 0.0,
                    'temporal_seasonal_variance': 0.0
                }
            
            profiles[user_id] = profile
        
        return {k: {user: profiles[user][k] for user in profiles} 
                for k in profiles[list(profiles.keys())[0]].keys()}
    
    def _build_similarity_profiles(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                                  active_users: List[int]) -> Dict[str, Dict[str, float]]:
        """Build similarity-based profiles using matrix factorization."""
        profiles = {}
        
        try:
            # Create user-movie matrix
            user_movie_matrix = ratings_df.pivot_table(
                index='UserID', 
                columns='MovieID', 
                values='Rating', 
                fill_value=0
            )
            
            # Filter to active users
            active_user_matrix = user_movie_matrix.loc[
                user_movie_matrix.index.intersection(active_users)
            ]
            
            if len(active_user_matrix) < 10:  # Need minimum users
                self.logger.warning("Insufficient users for similarity profiles")
                return {}
            
            # Apply SVD for dimensionality reduction
            self.svd_model = TruncatedSVD(n_components=min(self.profile_dimensions, len(active_user_matrix)))
            user_factors = self.svd_model.fit_transform(active_user_matrix)
            
            # Create similarity features
            for i, user_id in enumerate(active_user_matrix.index):
                profile = {}
                user_vector = user_factors[i]
                
                # SVD components as features
                for j, component in enumerate(user_vector):
                    profile[f'similarity_component_{j}'] = component
                
                # Vector magnitude (overall preference strength)
                profile['similarity_preference_strength'] = np.linalg.norm(user_vector)
                
                # Dominant component (primary preference dimension)
                profile['similarity_dominant_component'] = np.argmax(np.abs(user_vector))
                
                profiles[user_id] = profile
            
        except Exception as e:
            self.logger.error(f"Error building similarity profiles: {e}")
            return {}
        
        # Ensure all active users have profiles (fill missing with zeros)
        all_keys = set()
        for profile in profiles.values():
            all_keys.update(profile.keys())
        
        for user_id in active_users:
            if user_id not in profiles:
                profiles[user_id] = {key: 0.0 for key in all_keys}
        
        return {key: {user: profiles[user].get(key, 0.0) for user in profiles} 
                for key in all_keys}
    
    def transform_new_users(self, ratings_df: pd.DataFrame, 
                           movies_df: pd.DataFrame) -> pd.DataFrame:
        """Transform new user data using fitted transformers."""
        if not hasattr(self, 'user_statistics') or not self.user_statistics:
            raise ValueError("User profile builder not fitted. Call build_user_profiles first.")
        
        # Build profiles for new users using same methods
        new_user_stats = self._calculate_user_statistics(ratings_df)
        active_users = self._filter_active_users(ratings_df)
        
        profiles_data = {}
        
        # Use fitted transformers where available
        rating_profiles = self._build_rating_profiles(ratings_df, active_users)
        profiles_data.update(rating_profiles)
        
        genre_profiles = self._build_genre_profiles(ratings_df, movies_df, active_users)
        profiles_data.update(genre_profiles)
        
        behavioral_profiles = self._build_behavioral_profiles(ratings_df, active_users)
        profiles_data.update(behavioral_profiles)
        
        temporal_profiles = self._build_temporal_profiles(ratings_df, active_users)
        profiles_data.update(temporal_profiles)
        
        # For similarity profiles, transform using fitted SVD
        if self.svd_model is not None:
            similarity_profiles = self._transform_similarity_profiles(ratings_df, active_users)
            profiles_data.update(similarity_profiles)
        
        user_profiles_df = pd.DataFrame.from_dict(profiles_data, orient='index')
        user_profiles_df.index.name = 'UserID'
        user_profiles_df = user_profiles_df.reset_index()
        
        return user_profiles_df
    
    def _transform_similarity_profiles(self, ratings_df: pd.DataFrame, 
                                      active_users: List[int]) -> Dict[str, Dict[str, float]]:
        """Transform new users using fitted SVD model."""
        if self.svd_model is None:
            return {}
        
        profiles = {}
        
        try:
            # Create user-movie matrix for new users
            user_movie_matrix = ratings_df.pivot_table(
                index='UserID', 
                columns='MovieID', 
                values='Rating', 
                fill_value=0
            )
            
            # Transform using fitted SVD
            user_factors = self.svd_model.transform(user_movie_matrix)
            
            for i, user_id in enumerate(user_movie_matrix.index):
                if user_id in active_users:
                    profile = {}
                    user_vector = user_factors[i]
                    
                    # SVD components
                    for j, component in enumerate(user_vector):
                        profile[f'similarity_component_{j}'] = component
                    
                    profile['similarity_preference_strength'] = np.linalg.norm(user_vector)
                    profile['similarity_dominant_component'] = np.argmax(np.abs(user_vector))
                    
                    profiles[user_id] = profile
                    
        except Exception as e:
            self.logger.error(f"Error transforming similarity profiles: {e}")
        
        all_keys = set()
        for profile in profiles.values():
            all_keys.update(profile.keys())
        
        return {key: {user: profiles[user].get(key, 0.0) for user in profiles} 
                for key in all_keys}
    
    def get_user_profile_summary(self) -> Dict[str, Any]:
        """Get summary of user profile features and statistics."""
        if not hasattr(self, 'user_statistics') or not self.user_statistics:
            return {'error': 'User profiles not built'}
        
        summary = {
            'total_users': len(self.user_statistics),
            'profile_dimensions': self.profile_dimensions,
            'min_ratings_per_user': self.min_ratings_per_user,
            'feature_categories': [
                'rating_features',
                'genre_features', 
                'behavioral_features',
                'temporal_features',
                'similarity_features'
            ],
            'user_activity_distribution': {
                'min_ratings': min(stats['total_ratings'] for stats in self.user_statistics.values()),
                'max_ratings': max(stats['total_ratings'] for stats in self.user_statistics.values()),
                'avg_ratings': np.mean([stats['total_ratings'] for stats in self.user_statistics.values()]),
                'median_ratings': np.median([stats['total_ratings'] for stats in self.user_statistics.values()])
            },
            'rating_tendencies': {
                'avg_user_rating': np.mean([stats['avg_rating'] for stats in self.user_statistics.values()]),
                'rating_std_avg': np.mean([stats['rating_std'] for stats in self.user_statistics.values()]),
                'harsh_users': len([u for u, s in self.user_statistics.items() if s['avg_rating'] < 4.0]),
                'generous_users': len([u for u, s in self.user_statistics.items() if s['avg_rating'] > 7.0])
            }
        }
        
        return summary