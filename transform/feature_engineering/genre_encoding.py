"""
Genre Encoding and Feature Engineering Module
Handles one-hot encoding for genres and creates genre-based features for recommendation systems.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json


class GenreEncoder:
    """Advanced genre encoding with multiple representation strategies."""
    
    def __init__(self, min_genre_frequency: int = 5, max_genres: int = 50):
        self.logger = logging.getLogger(__name__)
        self.min_genre_frequency = min_genre_frequency
        self.max_genres = max_genres
        
        # Fitted encoders and transformers
        self.multi_label_binarizer = None
        self.tfidf_vectorizer = None
        self.genre_embeddings = None
        self.genre_statistics = {}
        
        # Genre mappings and hierarchies
        self.genre_mapping = self._initialize_genre_mapping()
        self.genre_hierarchy = self._initialize_genre_hierarchy()
        
    def _initialize_genre_mapping(self) -> Dict[str, str]:
        """Initialize genre name standardization mapping."""
        return {
            # Common variations and standardizations
            'sci-fi': 'Science Fiction',
            'scifi': 'Science Fiction',
            'romantic': 'Romance',
            'biopic': 'Biography',
            'docu': 'Documentary',
            'doc': 'Documentary',
            'kids': 'Family',
            'children': 'Family',
            'adult': 'Adult',
            'historical': 'History',
            'period': 'History',
            'psychological': 'Thriller',
            'suspense': 'Thriller',
            'slasher': 'Horror',
            'supernatural': 'Horror',
            'martial arts': 'Action',
            'superhero': 'Action',
            'heist': 'Crime',
            'neo-noir': 'Crime',
            'court': 'Drama',
            'legal': 'Drama',
            'medical': 'Drama',
            'sports': 'Sport',
            'athletics': 'Sport'
        }
    
    def _initialize_genre_hierarchy(self) -> Dict[str, List[str]]:
        """Initialize genre hierarchy for parent-child relationships."""
        return {
            'Action': ['Adventure', 'Thriller'],
            'Drama': ['Biography', 'History', 'Romance'],
            'Comedy': ['Family', 'Romance'],
            'Horror': ['Thriller'],
            'Science Fiction': ['Adventure', 'Action'],
            'Crime': ['Thriller', 'Drama'],
            'Documentary': ['Biography', 'History'],
            'Fantasy': ['Adventure', 'Family'],
            'Music': ['Musical', 'Biography'],
            'Mystery': ['Crime', 'Thriller']
        }
    
    def fit_transform_movies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit genre encoders and transform movie data with multiple genre representations.
        
        Args:
            df: Movies DataFrame with 'Genres' column
            
        Returns:
            Tuple of (transformed DataFrame, encoding metadata)
        """
        if 'Genres' not in df.columns:
            self.logger.warning("No 'Genres' column found in movies data")
            return df, {}
        
        self.logger.info(f"Fitting genre encoders on {len(df)} movies")
        
        # Parse and clean genres
        genre_lists = self._parse_genre_strings(df['Genres'])
        
        # Calculate genre statistics
        self.genre_statistics = self._calculate_genre_statistics(genre_lists)
        
        # Filter genres by frequency
        filtered_genre_lists = self._filter_genres_by_frequency(genre_lists)
        
        # Fit and transform with multiple encoding strategies
        df_encoded = df.copy()
        
        # 1. One-hot encoding
        onehot_features, onehot_columns = self._fit_transform_onehot(filtered_genre_lists)
        
        # 2. TF-IDF encoding
        tfidf_features, tfidf_columns = self._fit_transform_tfidf(df['Genres'])
        
        # 3. Genre embeddings (simple average-based)
        embedding_features, embedding_columns = self._fit_transform_embeddings(filtered_genre_lists)
        
        # 4. Genre hierarchy features
        hierarchy_features, hierarchy_columns = self._transform_hierarchy_features(filtered_genre_lists)
        
        # Add all encoded features to DataFrame
        for i, col in enumerate(onehot_columns):
            df_encoded[f'genre_onehot_{col}'] = onehot_features[:, i]
            
        for i, col in enumerate(tfidf_columns):
            df_encoded[f'genre_tfidf_{col}'] = tfidf_features[:, i]
            
        for i, col in enumerate(embedding_columns):
            df_encoded[f'genre_emb_{col}'] = embedding_features[:, i]
            
        for col, values in hierarchy_features.items():
            df_encoded[f'genre_hier_{col}'] = values
        
        # Add genre count and diversity features
        df_encoded['genre_count'] = [len(genres) for genres in filtered_genre_lists]
        df_encoded['genre_diversity'] = self._calculate_genre_diversity(filtered_genre_lists)
        df_encoded['genre_popularity'] = self._calculate_genre_popularity(filtered_genre_lists)
        
        # Create encoding metadata
        encoding_metadata = {
            'total_unique_genres': len(self.genre_statistics['genre_counts']),
            'filtered_genres': len(onehot_columns),
            'onehot_columns': onehot_columns,
            'tfidf_columns': tfidf_columns,
            'embedding_dimensions': len(embedding_columns),
            'hierarchy_features': list(hierarchy_features.keys()),
            'genre_statistics': self.genre_statistics
        }
        
        self.logger.info(f"Genre encoding completed: {len(onehot_columns)} one-hot features, "
                        f"{len(tfidf_columns)} TF-IDF features, {len(embedding_columns)} embedding features")
        
        return df_encoded, encoding_metadata
    
    def transform_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new movie data using fitted encoders.
        
        Args:
            df: Movies DataFrame with 'Genres' column
            
        Returns:
            Transformed DataFrame with genre features
        """
        if self.multi_label_binarizer is None:
            raise ValueError("Encoders not fitted. Call fit_transform_movies first.")
        
        if 'Genres' not in df.columns:
            self.logger.warning("No 'Genres' column found in movies data")
            return df
        
        df_encoded = df.copy()
        
        # Parse genres
        genre_lists = self._parse_genre_strings(df['Genres'])
        filtered_genre_lists = self._filter_genres_by_frequency(genre_lists)
        
        # Transform with fitted encoders
        onehot_features = self.multi_label_binarizer.transform(filtered_genre_lists)
        tfidf_features = self.tfidf_vectorizer.transform(df['Genres'].fillna(''))
        embedding_features, embedding_columns = self._transform_embeddings(filtered_genre_lists)
        hierarchy_features, hierarchy_columns = self._transform_hierarchy_features(filtered_genre_lists)
        
        # Add features to DataFrame
        onehot_columns = self.multi_label_binarizer.classes_
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        for i, col in enumerate(onehot_columns):
            df_encoded[f'genre_onehot_{col}'] = onehot_features[:, i]
            
        if hasattr(tfidf_features, 'toarray'):
            tfidf_array = tfidf_features.toarray()
        else:
            tfidf_array = tfidf_features
            
        for i, col in enumerate(tfidf_feature_names):
            df_encoded[f'genre_tfidf_{col}'] = tfidf_array[:, i]
            
        for i, col in enumerate(embedding_columns):
            df_encoded[f'genre_emb_{col}'] = embedding_features[:, i]
            
        for col, values in hierarchy_features.items():
            df_encoded[f'genre_hier_{col}'] = values
        
        # Add derived features
        df_encoded['genre_count'] = [len(genres) for genres in filtered_genre_lists]
        df_encoded['genre_diversity'] = self._calculate_genre_diversity(filtered_genre_lists)
        df_encoded['genre_popularity'] = self._calculate_genre_popularity(filtered_genre_lists)
        
        return df_encoded
    
    def _parse_genre_strings(self, genre_series: pd.Series) -> List[List[str]]:
        """Parse genre strings into lists of standardized genre names."""
        genre_lists = []
        
        for genre_string in genre_series:
            if pd.isna(genre_string) or genre_string == '':
                genre_lists.append([])
                continue
            
            # Split by common delimiters
            raw_genres = []
            for delimiter in ['|', ',', ';']:
                if delimiter in str(genre_string):
                    raw_genres = str(genre_string).split(delimiter)
                    break
            else:
                raw_genres = [str(genre_string)]
            
            # Clean and standardize genres
            clean_genres = []
            for genre in raw_genres:
                clean_genre = genre.strip().title()
                
                # Apply mapping
                standardized_genre = self.genre_mapping.get(clean_genre.lower(), clean_genre)
                
                if standardized_genre and standardized_genre not in clean_genres:
                    clean_genres.append(standardized_genre)
            
            genre_lists.append(clean_genres)
        
        return genre_lists
    
    def _calculate_genre_statistics(self, genre_lists: List[List[str]]) -> Dict[str, Any]:
        """Calculate comprehensive genre statistics."""
        all_genres = [genre for genres in genre_lists for genre in genres]
        genre_counts = pd.Series(all_genres).value_counts().to_dict()
        
        # Movie count per genre
        movies_per_genre = {}
        for genres in genre_lists:
            for genre in set(genres):  # Use set to avoid counting same movie multiple times
                movies_per_genre[genre] = movies_per_genre.get(genre, 0) + 1
        
        # Genre co-occurrence matrix
        genre_cooccurrence = {}
        unique_genres = list(genre_counts.keys())
        
        for genre1 in unique_genres:
            genre_cooccurrence[genre1] = {}
            for genre2 in unique_genres:
                if genre1 != genre2:
                    cooccurrence_count = sum(
                        1 for genres in genre_lists 
                        if genre1 in genres and genre2 in genres
                    )
                    genre_cooccurrence[genre1][genre2] = cooccurrence_count
        
        return {
            'genre_counts': genre_counts,
            'movies_per_genre': movies_per_genre,
            'total_movies': len(genre_lists),
            'average_genres_per_movie': np.mean([len(genres) for genres in genre_lists]),
            'genre_cooccurrence': genre_cooccurrence,
            'most_common_genres': list(genre_counts.keys())[:10],
            'least_common_genres': list(genre_counts.keys())[-10:]
        }
    
    def _filter_genres_by_frequency(self, genre_lists: List[List[str]]) -> List[List[str]]:
        """Filter out rare genres based on minimum frequency."""
        if not hasattr(self, 'genre_statistics') or not self.genre_statistics:
            return genre_lists
        
        frequent_genres = set(
            genre for genre, count in self.genre_statistics['genre_counts'].items()
            if count >= self.min_genre_frequency
        )
        
        # Also limit to top N genres if specified
        if self.max_genres > 0:
            top_genres = list(self.genre_statistics['genre_counts'].keys())[:self.max_genres]
            frequent_genres = frequent_genres.intersection(set(top_genres))
        
        filtered_lists = []
        for genres in genre_lists:
            filtered_genres = [g for g in genres if g in frequent_genres]
            filtered_lists.append(filtered_genres)
        
        return filtered_lists
    
    def _fit_transform_onehot(self, genre_lists: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform using multi-label binarizer (one-hot encoding)."""
        self.multi_label_binarizer = MultiLabelBinarizer()
        onehot_features = self.multi_label_binarizer.fit_transform(genre_lists)
        
        return onehot_features, list(self.multi_label_binarizer.classes_)
    
    def _fit_transform_tfidf(self, genre_series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform using TF-IDF vectorizer."""
        # Prepare genre text for TF-IDF (treat each genre as a word)
        genre_texts = genre_series.fillna('').apply(lambda x: str(x).replace('|', ' ').replace(',', ' '))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_genres,
            lowercase=True,
            token_pattern=r'\b[A-Za-z]+\b',  # Only alphabetic tokens
            stop_words=None
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(genre_texts)
        feature_names = list(self.tfidf_vectorizer.get_feature_names_out())
        
        return tfidf_features, feature_names
    
    def _fit_transform_embeddings(self, genre_lists: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform using simple genre embeddings."""
        # Create simple embeddings based on genre co-occurrence
        unique_genres = list(set(genre for genres in genre_lists for genre in genres))
        embedding_dim = min(50, len(unique_genres))  # Limit embedding dimension
        
        # Simple random embeddings (in production, use pre-trained embeddings)
        np.random.seed(42)  # For reproducibility
        self.genre_embeddings = {
            genre: np.random.normal(0, 0.1, embedding_dim)
            for genre in unique_genres
        }
        
        # Transform genre lists to average embeddings
        embedding_features = []
        for genres in genre_lists:
            if genres:
                genre_embeds = [self.genre_embeddings.get(genre, np.zeros(embedding_dim)) for genre in genres]
                avg_embedding = np.mean(genre_embeds, axis=0)
            else:
                avg_embedding = np.zeros(embedding_dim)
            embedding_features.append(avg_embedding)
        
        embedding_features = np.array(embedding_features)
        embedding_columns = [f'dim_{i}' for i in range(embedding_dim)]
        
        return embedding_features, embedding_columns
    
    def _transform_embeddings(self, genre_lists: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
        """Transform genre lists using fitted embeddings."""
        if not self.genre_embeddings:
            return np.array([]), []
        
        embedding_dim = len(list(self.genre_embeddings.values())[0])
        embedding_features = []
        
        for genres in genre_lists:
            if genres:
                genre_embeds = [
                    self.genre_embeddings.get(genre, np.zeros(embedding_dim)) 
                    for genre in genres
                ]
                avg_embedding = np.mean(genre_embeds, axis=0)
            else:
                avg_embedding = np.zeros(embedding_dim)
            embedding_features.append(avg_embedding)
        
        embedding_features = np.array(embedding_features)
        embedding_columns = [f'dim_{i}' for i in range(embedding_dim)]
        
        return embedding_features, embedding_columns
    
    def _transform_hierarchy_features(self, genre_lists: List[List[str]]) -> Tuple[Dict[str, List], List[str]]:
        """Transform genres using hierarchy-based features."""
        hierarchy_features = {}
        
        # Parent genre presence
        for parent, children in self.genre_hierarchy.items():
            parent_presence = []
            for genres in genre_lists:
                # Check if any child genre is present
                has_child = any(child in genres for child in children)
                # Or if parent genre is directly present
                has_parent = parent in genres
                parent_presence.append(int(has_child or has_parent))
            
            hierarchy_features[f'has_{parent.lower().replace(" ", "_")}'] = parent_presence
        
        # Genre depth (how specific the genres are)
        genre_depth = []
        for genres in genre_lists:
            depths = []
            for genre in genres:
                # Check if genre is a parent or child
                if genre in self.genre_hierarchy:
                    depths.append(1)  # Parent genre
                elif any(genre in children for children in self.genre_hierarchy.values()):
                    depths.append(2)  # Child genre
                else:
                    depths.append(1)  # Standalone genre
            
            avg_depth = np.mean(depths) if depths else 0
            genre_depth.append(avg_depth)
        
        hierarchy_features['genre_depth'] = genre_depth
        
        return hierarchy_features, list(hierarchy_features.keys())
    
    def _calculate_genre_diversity(self, genre_lists: List[List[str]]) -> List[float]:
        """Calculate genre diversity score for each movie."""
        diversity_scores = []
        
        for genres in genre_lists:
            if len(genres) <= 1:
                diversity_scores.append(0.0)
                continue
            
            # Calculate diversity based on genre co-occurrence
            total_pairs = len(genres) * (len(genres) - 1) / 2
            diverse_pairs = 0
            
            for i, genre1 in enumerate(genres):
                for genre2 in genres[i+1:]:
                    # Check if genres rarely co-occur
                    cooccurrence = self.genre_statistics.get('genre_cooccurrence', {}).get(genre1, {}).get(genre2, 0)
                    if cooccurrence < self.min_genre_frequency:
                        diverse_pairs += 1
            
            diversity = diverse_pairs / total_pairs if total_pairs > 0 else 0
            diversity_scores.append(diversity)
        
        return diversity_scores
    
    def _calculate_genre_popularity(self, genre_lists: List[List[str]]) -> List[float]:
        """Calculate average genre popularity score for each movie."""
        popularity_scores = []
        total_movies = self.genre_statistics.get('total_movies', 1)
        
        for genres in genre_lists:
            if not genres:
                popularity_scores.append(0.0)
                continue
            
            genre_popularities = []
            for genre in genres:
                genre_count = self.genre_statistics.get('movies_per_genre', {}).get(genre, 0)
                popularity = genre_count / total_movies
                genre_popularities.append(popularity)
            
            avg_popularity = np.mean(genre_popularities)
            popularity_scores.append(avg_popularity)
        
        return popularity_scores
    
    def get_genre_features_summary(self) -> Dict[str, Any]:
        """Get summary of all genre features created."""
        if not self.multi_label_binarizer:
            return {'error': 'Genre encoder not fitted'}
        
        summary = {
            'encoding_strategies': ['one_hot', 'tfidf', 'embeddings', 'hierarchy'],
            'one_hot_features': len(self.multi_label_binarizer.classes_),
            'tfidf_features': len(self.tfidf_vectorizer.get_feature_names_out()) if self.tfidf_vectorizer else 0,
            'embedding_dimensions': len(list(self.genre_embeddings.values())[0]) if self.genre_embeddings else 0,
            'hierarchy_features': len(self.genre_hierarchy) + 1,  # +1 for genre_depth
            'additional_features': ['genre_count', 'genre_diversity', 'genre_popularity'],
            'total_features': (
                len(self.multi_label_binarizer.classes_) +
                (len(self.tfidf_vectorizer.get_feature_names_out()) if self.tfidf_vectorizer else 0) +
                (len(list(self.genre_embeddings.values())[0]) if self.genre_embeddings else 0) +
                len(self.genre_hierarchy) + 1 + 3  # hierarchy + additional
            ),
            'genre_statistics': self.genre_statistics
        }
        
        return summary