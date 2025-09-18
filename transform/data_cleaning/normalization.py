"""
Data Normalization and Standardization Module
Handles data standardization, format normalization, and type conversions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime
import unicodedata


class DataNormalizer:
    """Comprehensive data normalization and standardization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.normalization_stats = {}
    
    def normalize_movies_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize movie data with standardized formats and clean values.
        
        Args:
            df: Raw movies DataFrame
            
        Returns:
            Normalized movies DataFrame
        """
        df_normalized = df.copy()
        stats = {'original_rows': len(df)}
        
        # Normalize movie IDs
        if 'MovieID' in df_normalized.columns:
            df_normalized['MovieID'] = self._normalize_ids(df_normalized['MovieID'])
        
        # Normalize titles
        for title_col in ['OriginalTitle', 'EnglishTitle', 'Series_Title']:
            if title_col in df_normalized.columns:
                df_normalized[title_col] = self._normalize_text(df_normalized[title_col])
        
        # Normalize dates
        if 'ReleaseDate' in df_normalized.columns:
            df_normalized['ReleaseDate'] = self._normalize_dates(df_normalized['ReleaseDate'])
        elif 'Released_Year' in df_normalized.columns:
            df_normalized['ReleaseDate'] = self._normalize_year_to_date(df_normalized['Released_Year'])
        
        # Normalize numeric fields
        numeric_fields = ['Runtime', 'Budget', 'Revenue', 'bugt_amt']
        for field in numeric_fields:
            if field in df_normalized.columns:
                df_normalized[field] = self._normalize_numeric(df_normalized[field])
        
        # Normalize genres
        if 'Genres' in df_normalized.columns:
            df_normalized['Genres'] = self._normalize_genres(df_normalized['Genres'])
        elif 'Genre' in df_normalized.columns:
            df_normalized['Genres'] = self._normalize_genres(df_normalized['Genre'])
        
        # Normalize countries and languages
        if 'ProductionCountries' in df_normalized.columns:
            df_normalized['ProductionCountries'] = self._normalize_countries(df_normalized['ProductionCountries'])
        
        if 'SpokenLanguages' in df_normalized.columns:
            df_normalized['SpokenLanguages'] = self._normalize_languages(df_normalized['SpokenLanguages'])
        
        # Add standard columns if missing
        df_normalized = self._add_standard_movie_columns(df_normalized)
        
        stats['final_rows'] = len(df_normalized)
        self.normalization_stats['movies'] = stats
        
        self.logger.info(f"Normalized movies data: {stats['original_rows']} -> {stats['final_rows']} rows")
        return df_normalized
    
    def normalize_ratings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ratings data with standardized user IDs and rating scales.
        
        Args:
            df: Raw ratings DataFrame
            
        Returns:
            Normalized ratings DataFrame
        """
        df_normalized = df.copy()
        stats = {'original_rows': len(df)}
        
        # Normalize IDs
        if 'UserID' in df_normalized.columns:
            df_normalized['UserID'] = self._normalize_ids(df_normalized['UserID'])
        
        if 'MovieID' in df_normalized.columns:
            df_normalized['MovieID'] = self._normalize_ids(df_normalized['MovieID'])
        
        # Normalize rating scale (convert to 1-10 scale)
        if 'Rating' in df_normalized.columns:
            df_normalized['Rating'] = self._normalize_rating_scale(df_normalized['Rating'])
        
        # Normalize dates
        if 'Date' in df_normalized.columns:
            df_normalized['Date'] = self._normalize_dates(df_normalized['Date'])
        
        # Remove invalid ratings
        if 'Rating' in df_normalized.columns:
            original_count = len(df_normalized)
            df_normalized = df_normalized[
                (df_normalized['Rating'] >= 1.0) & 
                (df_normalized['Rating'] <= 10.0) &
                (df_normalized['Rating'].notna())
            ]
            removed_count = original_count - len(df_normalized)
            if removed_count > 0:
                self.logger.warning(f"Removed {removed_count} invalid ratings")
        
        stats['final_rows'] = len(df_normalized)
        stats['removed_invalid'] = stats['original_rows'] - stats['final_rows']
        self.normalization_stats['ratings'] = stats
        
        self.logger.info(f"Normalized ratings data: {stats['original_rows']} -> {stats['final_rows']} rows")
        return df_normalized
    
    def normalize_persons_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize persons/cast data with standardized names and roles.
        
        Args:
            df: Raw persons DataFrame
            
        Returns:
            Normalized persons DataFrame
        """
        df_normalized = df.copy()
        stats = {'original_rows': len(df)}
        
        # Normalize IDs
        for id_col in ['CastID', 'MovieID']:
            if id_col in df_normalized.columns:
                df_normalized[id_col] = self._normalize_ids(df_normalized[id_col])
        
        # Normalize names
        if 'Name' in df_normalized.columns:
            df_normalized['Name'] = self._normalize_person_names(df_normalized['Name'])
        
        # Normalize characters
        if 'Character' in df_normalized.columns:
            df_normalized['Character'] = self._normalize_text(df_normalized['Character'])
        
        # Normalize gender values
        if 'Gender' in df_normalized.columns:
            df_normalized['Gender'] = self._normalize_gender(df_normalized['Gender'])
        
        stats['final_rows'] = len(df_normalized)
        self.normalization_stats['persons'] = stats
        
        self.logger.info(f"Normalized persons data: {stats['original_rows']} -> {stats['final_rows']} rows")
        return df_normalized
    
    def normalize_imdb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize IMDb data to match standard movie schema.
        
        Args:
            df: Raw IMDb DataFrame
            
        Returns:
            Normalized IMDb DataFrame matching movies schema
        """
        df_normalized = df.copy()
        stats = {'original_rows': len(df)}
        
        # Map IMDb columns to standard schema
        column_mapping = {
            'Series_Title': 'OriginalTitle',
            'Released_Year': 'ReleaseYear',
            'Runtime': 'Runtime',
            'Genre': 'Genres',
            'IMDB_Rating': 'IMDBRating',
            'No_of_Votes': 'Votes',
            'Director': 'Director',
            'Gross': 'Revenue',
            'Certificate': 'Certificate',
            'Poster_Link': 'PosterURL',
            'Overview': 'Overview'
        }
        
        # Rename columns
        df_normalized = df_normalized.rename(columns=column_mapping)
        
        # Add English title (same as original for IMDb)
        if 'OriginalTitle' in df_normalized.columns:
            df_normalized['EnglishTitle'] = df_normalized['OriginalTitle']
        
        # Normalize text fields
        for col in ['OriginalTitle', 'EnglishTitle', 'Overview']:
            if col in df_normalized.columns:
                df_normalized[col] = self._normalize_text(df_normalized[col])
        
        # Normalize numeric fields
        if 'Runtime' in df_normalized.columns:
            df_normalized['Runtime'] = self._normalize_numeric(df_normalized['Runtime'])
        
        if 'Revenue' in df_normalized.columns:
            # Clean revenue (remove commas, convert to numeric)
            df_normalized['Revenue'] = df_normalized['Revenue'].astype(str).str.replace(',', '')
            df_normalized['Revenue'] = self._normalize_numeric(df_normalized['Revenue'])
        
        # Normalize ratings
        if 'IMDBRating' in df_normalized.columns:
            df_normalized['IMDBRating'] = self._normalize_numeric(df_normalized['IMDBRating'])
        
        # Normalize genres
        if 'Genres' in df_normalized.columns:
            df_normalized['Genres'] = self._normalize_genres(df_normalized['Genres'], separator=', ')
        
        stats['final_rows'] = len(df_normalized)
        self.normalization_stats['imdb'] = stats
        
        self.logger.info(f"Normalized IMDb data: {stats['original_rows']} -> {stats['final_rows']} rows")
        return df_normalized
    
    def _normalize_ids(self, series: pd.Series) -> pd.Series:
        """Normalize ID columns to consistent integer format."""
        # Convert to numeric, coercing errors to NaN
        numeric_ids = pd.to_numeric(series, errors='coerce')
        
        # Convert to integer where possible
        return numeric_ids.astype('Int64')
    
    def _normalize_text(self, series: pd.Series) -> pd.Series:
        """Normalize text fields with consistent formatting."""
        normalized = series.copy()
        
        # Handle NaN values
        normalized = normalized.fillna('')
        
        # Convert to string and strip whitespace
        normalized = normalized.astype(str).str.strip()
        
        # Remove extra whitespace
        normalized = normalized.str.replace(r'\s+', ' ', regex=True)
        
        # Normalize unicode characters
        normalized = normalized.apply(lambda x: unicodedata.normalize('NFKD', x))
        
        # Convert empty strings back to NaN
        normalized = normalized.replace('', np.nan)
        
        return normalized
    
    def _normalize_person_names(self, series: pd.Series) -> pd.Series:
        """Normalize person names with proper capitalization."""
        normalized = self._normalize_text(series)
        
        # Proper case for names
        normalized = normalized.str.title()
        
        # Handle common name particles
        name_particles = ['von', 'van', 'de', 'del', 'della', 'di', 'da', 'du', 'le', 'la']
        for particle in name_particles:
            normalized = normalized.str.replace(f' {particle.title()} ', f' {particle} ')
        
        return normalized
    
    def _normalize_dates(self, series: pd.Series) -> pd.Series:
        """Normalize date columns to consistent datetime format."""
        # Try multiple date formats
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y',
            '%B %d, %Y',
            '%b %d, %Y'
        ]
        
        normalized_dates = pd.Series(index=series.index, dtype='datetime64[ns]')
        
        for date_format in date_formats:
            mask = normalized_dates.isna()
            if mask.any():
                try:
                    parsed = pd.to_datetime(series[mask], format=date_format, errors='coerce')
                    normalized_dates[mask] = parsed
                except:
                    continue
        
        # Final attempt with flexible parsing
        mask = normalized_dates.isna()
        if mask.any():
            normalized_dates[mask] = pd.to_datetime(series[mask], errors='coerce')
        
        return normalized_dates
    
    def _normalize_year_to_date(self, series: pd.Series) -> pd.Series:
        """Convert year column to date format (January 1st of that year)."""
        numeric_years = pd.to_numeric(series, errors='coerce')
        
        # Create dates from years
        dates = pd.to_datetime(numeric_years.astype(str) + '-01-01', errors='coerce')
        
        return dates
    
    def _normalize_numeric(self, series: pd.Series) -> pd.Series:
        """Normalize numeric columns with consistent format."""
        # Remove non-numeric characters except decimal points
        cleaned = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        
        # Convert to numeric
        numeric = pd.to_numeric(cleaned, errors='coerce')
        
        return numeric
    
    def _normalize_rating_scale(self, series: pd.Series) -> pd.Series:
        """Normalize rating scale to 1-10 range."""
        numeric_ratings = pd.to_numeric(series, errors='coerce')
        
        # Detect current scale based on max value
        max_rating = numeric_ratings.max()
        
        if max_rating <= 5.0:
            # Scale from 0.5-5.0 to 1-10
            normalized_ratings = numeric_ratings * 2
        elif max_rating <= 10.0:
            # Already in 1-10 scale
            normalized_ratings = numeric_ratings
        elif max_rating <= 100:
            # Scale from 0-100 to 1-10
            normalized_ratings = (numeric_ratings / 100) * 9 + 1
        else:
            # Unknown scale, keep as-is
            normalized_ratings = numeric_ratings
            self.logger.warning(f"Unknown rating scale detected (max: {max_rating})")
        
        return normalized_ratings
    
    def _normalize_genres(self, series: pd.Series, separator: str = '|') -> pd.Series:
        """Normalize genre lists with consistent format and standard names."""
        # Genre name standardization mapping
        genre_mapping = {
            'sci-fi': 'Science Fiction',
            'scifi': 'Science Fiction',
            'adventure': 'Adventure',
            'action': 'Action',
            'comedy': 'Comedy',
            'drama': 'Drama',
            'horror': 'Horror',
            'thriller': 'Thriller',
            'romance': 'Romance',
            'fantasy': 'Fantasy',
            'animation': 'Animation',
            'family': 'Family',
            'crime': 'Crime',
            'mystery': 'Mystery',
            'war': 'War',
            'western': 'Western',
            'musical': 'Musical',
            'documentary': 'Documentary',
            'biography': 'Biography',
            'history': 'History',
            'sport': 'Sport'
        }
        
        def normalize_genre_list(genre_string):
            if pd.isna(genre_string) or genre_string == '':
                return np.nan
            
            # Split by common separators
            genres = re.split(r'[|,;]', str(genre_string))
            
            # Clean and normalize each genre
            normalized_genres = []
            for genre in genres:
                clean_genre = genre.strip().lower()
                
                # Map to standard name
                standard_genre = genre_mapping.get(clean_genre, genre.strip().title())
                
                if standard_genre and standard_genre not in normalized_genres:
                    normalized_genres.append(standard_genre)
            
            return '|'.join(normalized_genres) if normalized_genres else np.nan
        
        return series.apply(normalize_genre_list)
    
    def _normalize_countries(self, series: pd.Series) -> pd.Series:
        """Normalize country codes and names."""
        def normalize_country_list(country_string):
            if pd.isna(country_string) or country_string == '':
                return np.nan
            
            # Split by pipe
            countries = str(country_string).split('|')
            normalized_countries = []
            
            for country in countries:
                country = country.strip()
                if '-' in country:
                    # Format: code-name
                    code, name = country.split('-', 1)
                    normalized_countries.append(f"{code.strip().upper()}-{name.strip().title()}")
                else:
                    # Just country name or code
                    normalized_countries.append(country.title())
            
            return '|'.join(normalized_countries) if normalized_countries else np.nan
        
        return series.apply(normalize_country_list)
    
    def _normalize_languages(self, series: pd.Series) -> pd.Series:
        """Normalize language codes and names."""
        def normalize_language_list(language_string):
            if pd.isna(language_string) or language_string == '':
                return np.nan
            
            # Split by pipe
            languages = str(language_string).split('|')
            normalized_languages = []
            
            for language in languages:
                language = language.strip()
                if '-' in language:
                    # Format: code-endonym
                    code, endonym = language.split('-', 1)
                    normalized_languages.append(f"{code.strip().lower()}-{endonym.strip()}")
                else:
                    # Just language name
                    normalized_languages.append(language.title())
            
            return '|'.join(normalized_languages) if normalized_languages else np.nan
        
        return series.apply(normalize_language_list)
    
    def _normalize_gender(self, series: pd.Series) -> pd.Series:
        """Normalize gender values to standard codes."""
        gender_mapping = {
            'female': 1,
            'f': 1,
            '1': 1,
            1: 1,
            'male': 2,
            'm': 2,
            '2': 2,
            2: 2,
            'other': 3,
            'non-binary': 3,
            'nb': 3,
            '3': 3,
            3: 3
        }
        
        def normalize_gender_value(value):
            if pd.isna(value):
                return np.nan
            
            clean_value = str(value).strip().lower()
            return gender_mapping.get(clean_value, np.nan)
        
        return series.apply(normalize_gender_value)
    
    def _add_standard_movie_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing standard columns with default values."""
        standard_columns = {
            'Budget': 0,
            'Revenue': 0,
            'Homepage': '',
            'Votes': 0,
            'Certificate': '',
            'PosterURL': '',
            'Overview': ''
        }
        
        for col, default_value in standard_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        return df
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get statistics about the normalization process."""
        return {
            'timestamp': datetime.now().isoformat(),
            'datasets_processed': len(self.normalization_stats),
            'details': self.normalization_stats
        }