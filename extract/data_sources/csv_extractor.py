"""
CSV Data Extractor for Movie Recommendation ETL Pipeline
Handles extraction from multiple CSV data sources with validation and error handling.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml


class CSVExtractor:
    """Multi-format CSV file ingestion with data validation and error handling."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def extract_movies_data(self, csv_path: str) -> pd.DataFrame:
        """
        Extract movie metadata from CSV file.
        
        Args:
            csv_path: Path to movies CSV file
            
        Returns:
            DataFrame with movie data
        """
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            self.logger.info(f"Extracted {len(df)} movies from {csv_path}")
            
            # Basic data validation
            required_columns = ['MovieID', 'OriginalTitle', 'EnglishTitle']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract movies data from {csv_path}: {e}")
            raise
    
    def extract_ratings_data(self, csv_path: str) -> pd.DataFrame:
        """
        Extract user ratings data from CSV file.
        
        Args:
            csv_path: Path to ratings CSV file
            
        Returns:
            DataFrame with ratings data
        """
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            self.logger.info(f"Extracted {len(df)} ratings from {csv_path}")
            
            # Validate rating values
            if 'Rating' in df.columns:
                df = df[df['Rating'].notna()]
                df = df[df['Rating'].between(0.5, 5.0)]
                
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract ratings data from {csv_path}: {e}")
            raise
    
    def extract_persons_data(self, csv_path: str) -> pd.DataFrame:
        """
        Extract persons (actors/directors) data from CSV file.
        
        Args:
            csv_path: Path to persons CSV file
            
        Returns:
            DataFrame with persons data
        """
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            self.logger.info(f"Extracted {len(df)} persons from {csv_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract persons data from {csv_path}: {e}")
            raise
    
    def extract_imdb_data(self, csv_path: str) -> pd.DataFrame:
        """
        Extract IMDb top movies data from CSV file.
        
        Args:
            csv_path: Path to IMDb CSV file
            
        Returns:
            DataFrame with IMDb data
        """
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            self.logger.info(f"Extracted {len(df)} IMDb movies from {csv_path}")
            
            # Clean runtime column
            if 'Runtime' in df.columns:
                df['Runtime'] = df['Runtime'].str.replace(' min', '').astype('Int64')
            
            # Clean gross revenue column
            if 'Gross' in df.columns:
                df['Gross'] = df['Gross'].str.replace(',', '').replace('', None)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract IMDb data from {csv_path}: {e}")
            raise
    
    def extract_all_sources(self, data_sources: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Extract data from all configured sources.
        
        Args:
            data_sources: Dictionary mapping source names to file paths
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        results = {}
        
        for source_name, file_path in data_sources.items():
            if not Path(file_path).exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                if 'movies' in source_name.lower():
                    results[source_name] = self.extract_movies_data(file_path)
                elif 'ratings' in source_name.lower():
                    results[source_name] = self.extract_ratings_data(file_path)
                elif 'persons' in source_name.lower():
                    results[source_name] = self.extract_persons_data(file_path)
                elif 'imdb' in source_name.lower():
                    results[source_name] = self.extract_imdb_data(file_path)
                else:
                    # Generic CSV extraction
                    results[source_name] = pd.read_csv(file_path, encoding='utf-8')
                    
            except Exception as e:
                self.logger.error(f"Failed to extract from {source_name}: {e}")
                continue
        
        return results
    
    def get_extraction_stats(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about extracted data.
        
        Args:
            dataframes: Dictionary of extracted DataFrames
            
        Returns:
            Dictionary of statistics for each source
        """
        stats = {}
        
        for source_name, df in dataframes.items():
            stats[source_name] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'null_counts': df.isnull().sum().to_dict()
            }
        
        return stats