"""
API Connector for External Movie Data Sources
Handles integration with external APIs for real-time movie and rating data.
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


class APIConnector:
    """External API integration for movie metadata and ratings."""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.api_config = api_config
        self.session = requests.Session()
        self.rate_limit_delay = api_config.get('rate_limit_delay', 1.0)
        
    def _make_request(self, url: str, params: Dict[str, Any] = None, 
                     headers: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with error handling and rate limiting.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            JSON response data or None if failed
        """
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {url}: {e}")
            return None
    
    def fetch_movie_metadata(self, movie_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch movie metadata from external API.
        
        Args:
            movie_ids: List of movie IDs to fetch
            
        Returns:
            List of movie metadata dictionaries
        """
        movies_data = []
        base_url = self.api_config.get('movie_api_url', '')
        api_key = self.api_config.get('api_key', '')
        
        if not base_url or not api_key:
            self.logger.warning("Movie API configuration missing")
            return movies_data
        
        for movie_id in movie_ids:
            url = f"{base_url}/{movie_id}"
            params = {'api_key': api_key}
            
            data = self._make_request(url, params)
            if data:
                movies_data.append({
                    'external_id': movie_id,
                    'title': data.get('title', ''),
                    'release_date': data.get('release_date', ''),
                    'genre': data.get('genres', []),
                    'runtime': data.get('runtime', 0),
                    'budget': data.get('budget', 0),
                    'revenue': data.get('revenue', 0),
                    'overview': data.get('overview', ''),
                    'poster_url': data.get('poster_path', ''),
                    'vote_average': data.get('vote_average', 0.0),
                    'vote_count': data.get('vote_count', 0),
                    'fetch_timestamp': datetime.now().isoformat()
                })
                
        self.logger.info(f"Fetched metadata for {len(movies_data)} movies")
        return movies_data
    
    def fetch_trending_movies(self, time_window: str = 'week', 
                             page: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch trending movies from external API.
        
        Args:
            time_window: 'day' or 'week'
            page: Page number for pagination
            
        Returns:
            List of trending movie data
        """
        trending_url = self.api_config.get('trending_api_url', '')
        api_key = self.api_config.get('api_key', '')
        
        if not trending_url or not api_key:
            self.logger.warning("Trending API configuration missing")
            return []
        
        params = {
            'api_key': api_key,
            'time_window': time_window,
            'page': page
        }
        
        data = self._make_request(trending_url, params)
        if not data or 'results' not in data:
            return []
        
        trending_movies = []
        for movie in data['results']:
            trending_movies.append({
                'external_id': movie.get('id'),
                'title': movie.get('title', ''),
                'popularity': movie.get('popularity', 0.0),
                'vote_average': movie.get('vote_average', 0.0),
                'vote_count': movie.get('vote_count', 0),
                'release_date': movie.get('release_date', ''),
                'fetch_timestamp': datetime.now().isoformat(),
                'trending_rank': len(trending_movies) + 1
            })
            
        self.logger.info(f"Fetched {len(trending_movies)} trending movies")
        return trending_movies
    
    def fetch_user_ratings(self, user_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch user ratings from external rating service.
        
        Args:
            user_ids: List of user IDs to fetch ratings for
            
        Returns:
            List of user rating data
        """
        ratings_data = []
        ratings_url = self.api_config.get('ratings_api_url', '')
        api_key = self.api_config.get('api_key', '')
        
        if not ratings_url or not api_key:
            self.logger.warning("Ratings API configuration missing")
            return ratings_data
        
        for user_id in user_ids:
            url = f"{ratings_url}/user/{user_id}/ratings"
            params = {'api_key': api_key}
            
            data = self._make_request(url, params)
            if data and 'ratings' in data:
                for rating in data['ratings']:
                    ratings_data.append({
                        'user_id': user_id,
                        'movie_id': rating.get('movie_id'),
                        'rating': rating.get('rating', 0.0),
                        'timestamp': rating.get('timestamp', ''),
                        'fetch_timestamp': datetime.now().isoformat()
                    })
                    
        self.logger.info(f"Fetched {len(ratings_data)} ratings from API")
        return ratings_data
    
    def to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert API data to pandas DataFrame.
        
        Args:
            data: List of dictionaries from API
            
        Returns:
            DataFrame with API data
        """
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)
    
    def batch_fetch_movies(self, movie_ids: List[str], 
                          batch_size: int = 20) -> pd.DataFrame:
        """
        Fetch movie data in batches to respect API limits.
        
        Args:
            movie_ids: List of movie IDs
            batch_size: Size of each batch
            
        Returns:
            DataFrame with all fetched movie data
        """
        all_movies = []
        
        for i in range(0, len(movie_ids), batch_size):
            batch = movie_ids[i:i + batch_size]
            batch_data = self.fetch_movie_metadata(batch)
            all_movies.extend(batch_data)
            
            # Progress logging
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(movie_ids) + batch_size - 1)//batch_size}")
        
        return self.to_dataframe(all_movies)
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Check API status and rate limits.
        
        Returns:
            Dictionary with API status information
        """
        status_url = self.api_config.get('status_api_url', '')
        api_key = self.api_config.get('api_key', '')
        
        if not status_url:
            return {'status': 'unknown', 'message': 'Status URL not configured'}
        
        params = {'api_key': api_key}
        data = self._make_request(status_url, params)
        
        if data:
            return {
                'status': 'active',
                'rate_limit_remaining': data.get('rate_limit_remaining', 'unknown'),
                'rate_limit_reset': data.get('rate_limit_reset', 'unknown')
            }
        else:
            return {'status': 'error', 'message': 'Failed to fetch status'}