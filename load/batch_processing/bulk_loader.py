"""
High-Performance Bulk Data Loader
Optimized bulk insertion for large-scale ETL operations with error handling and monitoring.
"""

import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple, Generator
import numpy as np
from datetime import datetime
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time


@dataclass
class LoadResult:
    """Result of a bulk load operation."""
    table_name: str
    records_attempted: int
    records_inserted: int
    records_updated: int
    records_failed: int
    execution_time: float
    errors: List[str]
    batch_results: List[Dict[str, Any]]


class BulkLoader:
    """High-performance bulk data loader with parallel processing and error recovery."""
    
    def __init__(self, db_path: str, batch_size: int = 1000, max_workers: int = 4):
        self.db_path = db_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.load_statistics = {}
        self.error_queue = queue.Queue()
        
        # Connection pool for parallel loading
        self.connection_pool = queue.Queue(maxsize=max_workers)
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for parallel operations."""
        for _ in range(self.max_workers):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
                conn.execute("PRAGMA synchronous = NORMAL")  # Balance safety and performance
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                conn.execute("PRAGMA temp_store = memory")
                self.connection_pool.put(conn)
            except Exception as e:
                self.logger.error(f"Failed to create database connection: {e}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        return self.connection_pool.get()
    
    def _return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        self.connection_pool.put(conn)
    
    def bulk_load_movies(self, movies_df: pd.DataFrame, 
                        update_existing: bool = True) -> LoadResult:
        """
        Bulk load movies data with optimized insertion strategy.
        
        Args:
            movies_df: Movies DataFrame
            update_existing: Whether to update existing records
            
        Returns:
            LoadResult with operation statistics
        """
        self.logger.info(f"Starting bulk load of {len(movies_df)} movies")
        start_time = time.time()
        
        # Prepare data for insertion
        prepared_data = self._prepare_movies_data(movies_df)
        
        # Execute bulk load
        result = self._execute_bulk_load(
            table_name='movie',
            data=prepared_data,
            update_existing=update_existing,
            conflict_columns=['id']
        )
        
        result.execution_time = time.time() - start_time
        self.load_statistics['movies'] = result
        
        self.logger.info(f"Movies bulk load completed: {result.records_inserted} inserted, "
                        f"{result.records_updated} updated, {result.records_failed} failed")
        return result
    
    def bulk_load_ratings(self, ratings_df: pd.DataFrame,
                         update_existing: bool = False) -> LoadResult:
        """
        Bulk load user ratings with duplicate handling.
        
        Args:
            ratings_df: Ratings DataFrame
            update_existing: Whether to update existing ratings
            
        Returns:
            LoadResult with operation statistics
        """
        self.logger.info(f"Starting bulk load of {len(ratings_df)} ratings")
        start_time = time.time()
        
        # Prepare data for insertion
        prepared_data = self._prepare_ratings_data(ratings_df)
        
        # Execute bulk load with special handling for user-movie uniqueness
        result = self._execute_bulk_load(
            table_name='user_rating',
            data=prepared_data,
            update_existing=update_existing,
            conflict_columns=['user_id', 'movie_id']
        )
        
        result.execution_time = time.time() - start_time
        self.load_statistics['ratings'] = result
        
        self.logger.info(f"Ratings bulk load completed: {result.records_inserted} inserted, "
                        f"{result.records_updated} updated, {result.records_failed} failed")
        return result
    
    def bulk_load_persons(self, persons_df: pd.DataFrame,
                         cast_df: Optional[pd.DataFrame] = None,
                         crew_df: Optional[pd.DataFrame] = None) -> Dict[str, LoadResult]:
        """
        Bulk load persons and their movie relationships.
        
        Args:
            persons_df: Persons DataFrame
            cast_df: Optional cast relationships DataFrame
            crew_df: Optional crew relationships DataFrame
            
        Returns:
            Dictionary of LoadResults for each table
        """
        results = {}
        
        # Load persons first
        if not persons_df.empty:
            self.logger.info(f"Loading {len(persons_df)} persons")
            prepared_persons = self._prepare_persons_data(persons_df)
            results['persons'] = self._execute_bulk_load(
                table_name='person',
                data=prepared_persons,
                update_existing=True,
                conflict_columns=['id']
            )
        
        # Load cast relationships
        if cast_df is not None and not cast_df.empty:
            self.logger.info(f"Loading {len(cast_df)} cast relationships")
            prepared_cast = self._prepare_cast_data(cast_df)
            results['cast'] = self._execute_bulk_load(
                table_name='movie_cast',
                data=prepared_cast,
                update_existing=False,
                conflict_columns=['movie_id', 'person_id', 'role_type']
            )
        
        # Load crew relationships
        if crew_df is not None and not crew_df.empty:
            self.logger.info(f"Loading {len(crew_df)} crew relationships")
            prepared_crew = self._prepare_crew_data(crew_df)
            results['crew'] = self._execute_bulk_load(
                table_name='movie_crew',
                data=prepared_crew,
                update_existing=False,
                conflict_columns=['movie_id', 'person_id', 'job_title']
            )
        
        return results
    
    def bulk_load_genres(self, movies_df: pd.DataFrame) -> Dict[str, LoadResult]:
        """
        Bulk load genres and movie-genre relationships.
        
        Args:
            movies_df: Movies DataFrame with genre information
            
        Returns:
            Dictionary of LoadResults
        """
        results = {}
        
        # Extract unique genres
        genres_data = self._extract_genres_from_movies(movies_df)
        
        if genres_data:
            self.logger.info(f"Loading {len(genres_data)} genres")
            results['genres'] = self._execute_bulk_load(
                table_name='genre',
                data=genres_data,
                update_existing=True,
                conflict_columns=['name']
            )
        
        # Load movie-genre relationships
        genre_relationships = self._prepare_movie_genre_relationships(movies_df)
        
        if genre_relationships:
            self.logger.info(f"Loading {len(genre_relationships)} movie-genre relationships")
            results['movie_genres'] = self._execute_bulk_load(
                table_name='movie_genres',
                data=genre_relationships,
                update_existing=False,
                conflict_columns=['movie_id', 'genre_id']
            )
        
        return results
    
    def bulk_load_features(self, user_profiles_df: Optional[pd.DataFrame] = None,
                          movie_features_df: Optional[pd.DataFrame] = None) -> Dict[str, LoadResult]:
        """
        Bulk load pre-computed features for ML models.
        
        Args:
            user_profiles_df: User profiles DataFrame
            movie_features_df: Movie features DataFrame
            
        Returns:
            Dictionary of LoadResults
        """
        results = {}
        
        # Load user profiles
        if user_profiles_df is not None and not user_profiles_df.empty:
            self.logger.info(f"Loading {len(user_profiles_df)} user profiles")
            prepared_profiles = self._prepare_user_profiles_data(user_profiles_df)
            results['user_profiles'] = self._execute_bulk_load(
                table_name='user_profiles',
                data=prepared_profiles,
                update_existing=True,
                conflict_columns=['user_id']
            )
        
        # Load movie features
        if movie_features_df is not None and not movie_features_df.empty:
            self.logger.info(f"Loading {len(movie_features_df)} movie feature sets")
            prepared_features = self._prepare_movie_features_data(movie_features_df)
            results['movie_features'] = self._execute_bulk_load(
                table_name='movie_features',
                data=prepared_features,
                update_existing=True,
                conflict_columns=['movie_id']
            )
        
        return results
    
    def _prepare_movies_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare movies data for bulk insertion."""
        prepared_data = []
        
        for _, row in df.iterrows():
            record = {
                'id': int(row.get('MovieID', 0)),
                'original_title': str(row.get('OriginalTitle', '')),
                'english_title': str(row.get('EnglishTitle', row.get('OriginalTitle', ''))),
                'release_date': self._parse_date(row.get('ReleaseDate')),
                'runtime': self._safe_int(row.get('Runtime')),
                'budget': self._safe_float(row.get('Budget', row.get('bugt_amt', 0))),
                'revenue': self._safe_float(row.get('Revenue', 0)),
                'homepage': str(row.get('Homepage', '')),
                'overview': str(row.get('Overview', '')),
                'poster_url': str(row.get('PosterURL', row.get('Poster_Link', ''))),
                'imdb_rating': self._safe_float(row.get('IMDBRating', row.get('IMDB_Rating'))),
                'vote_count': self._safe_int(row.get('Votes', row.get('No_of_Votes', 0))),
                'certificate': str(row.get('Certificate', '')),
                'popularity_score': self._calculate_popularity_score(row),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            prepared_data.append(record)
        
        return prepared_data
    
    def _prepare_ratings_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare ratings data for bulk insertion."""
        prepared_data = []
        
        for _, row in df.iterrows():
            record = {
                'user_id': int(row.get('UserID', 0)),
                'movie_id': int(row.get('MovieID', 0)),
                'rating': float(row.get('Rating', 0)),
                'rating_date': self._parse_datetime(row.get('Date')),
                'source': 'bulk_import',
                'confidence': 1.0
            }
            
            # Validate rating range
            if 1.0 <= record['rating'] <= 10.0:
                prepared_data.append(record)
            else:
                self.logger.warning(f"Invalid rating value: {record['rating']} for user {record['user_id']}")
        
        return prepared_data
    
    def _prepare_persons_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare persons data for bulk insertion."""
        prepared_data = []
        
        for _, row in df.iterrows():
            record = {
                'id': str(row.get('CastID', row.get('PersonID', ''))),
                'name': str(row.get('Name', '')),
                'gender': self._safe_int(row.get('Gender')),
                'birth_date': self._parse_date(row.get('BirthDate')),
                'biography': str(row.get('Biography', '')),
                'profile_image': str(row.get('ProfileImage', '')),
                'popularity_score': self._safe_float(row.get('Popularity', 0)),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            prepared_data.append(record)
        
        return prepared_data
    
    def _prepare_cast_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare cast relationship data."""
        prepared_data = []
        
        for _, row in df.iterrows():
            record = {
                'movie_id': int(row.get('MovieID', 0)),
                'person_id': str(row.get('CastID', row.get('PersonID', ''))),
                'character_name': str(row.get('Character', '')),
                'cast_order': self._safe_int(row.get('CastOrder', 999)),
                'role_type': 'actor'
            }
            prepared_data.append(record)
        
        return prepared_data
    
    def _prepare_crew_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare crew relationship data."""
        prepared_data = []
        
        for _, row in df.iterrows():
            record = {
                'movie_id': int(row.get('MovieID', 0)),
                'person_id': str(row.get('PersonID', '')),
                'job_title': str(row.get('Job', row.get('JobTitle', ''))),
                'department': str(row.get('Department', ''))
            }
            prepared_data.append(record)
        
        return prepared_data
    
    def _prepare_user_profiles_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare user profiles data for insertion."""
        prepared_data = []
        
        for _, row in df.iterrows():
            # Separate key features from full profile data
            profile_data = row.to_dict()
            user_id = profile_data.pop('UserID', None)
            
            if user_id is not None:
                record = {
                    'user_id': int(user_id),
                    'profile_data': json.dumps(profile_data, default=str),
                    'feature_version': '1.0',
                    'avg_rating': self._safe_float(row.get('rating_avg')),
                    'rating_count': self._safe_int(row.get('rating_count')),
                    'genre_diversity': self._safe_float(row.get('genre_diversity')),
                    'activity_level': self._determine_activity_level(row.get('rating_count', 0)),
                    'primary_genre': self._determine_primary_genre(row),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                prepared_data.append(record)
        
        return prepared_data
    
    def _prepare_movie_features_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare movie features data for insertion."""
        prepared_data = []
        
        for _, row in df.iterrows():
            feature_data = row.to_dict()
            movie_id = feature_data.pop('MovieID', None)
            
            if movie_id is not None:
                # Extract genre vector if present
                genre_vector = self._extract_genre_vector(row)
                
                record = {
                    'movie_id': int(movie_id),
                    'feature_data': json.dumps(feature_data, default=str),
                    'feature_version': '1.0',
                    'popularity_score': self._safe_float(row.get('popularity_score', 0)),
                    'quality_score': self._safe_float(row.get('quality_score', 0)),
                    'genre_vector': json.dumps(genre_vector) if genre_vector else None,
                    'content_rating': str(row.get('Certificate', '')),
                    'target_audience': self._determine_target_audience(row),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                prepared_data.append(record)
        
        return prepared_data
    
    def _execute_bulk_load(self, table_name: str, data: List[Dict[str, Any]],
                          update_existing: bool, conflict_columns: List[str]) -> LoadResult:
        """Execute bulk load with parallel processing and error handling."""
        if not data:
            return LoadResult(
                table_name=table_name,
                records_attempted=0,
                records_inserted=0,
                records_updated=0,
                records_failed=0,
                execution_time=0.0,
                errors=[],
                batch_results=[]
            )
        
        start_time = time.time()
        
        # Split data into batches
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        # Process batches in parallel
        batch_results = []
        total_inserted = 0
        total_updated = 0
        total_failed = 0
        all_errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(
                    self._process_batch, 
                    table_name, 
                    batch, 
                    update_existing, 
                    conflict_columns,
                    i
                ): i for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    batch_result = future.result()
                    batch_results.append(batch_result)
                    total_inserted += batch_result['inserted']
                    total_updated += batch_result['updated']
                    total_failed += batch_result['failed']
                    all_errors.extend(batch_result['errors'])
                    
                except Exception as e:
                    error_msg = f"Batch {batch_index} failed: {str(e)}"
                    self.logger.error(error_msg)
                    all_errors.append(error_msg)
                    total_failed += len(batches[batch_index])
        
        execution_time = time.time() - start_time
        
        return LoadResult(
            table_name=table_name,
            records_attempted=len(data),
            records_inserted=total_inserted,
            records_updated=total_updated,
            records_failed=total_failed,
            execution_time=execution_time,
            errors=all_errors,
            batch_results=batch_results
        )
    
    def _process_batch(self, table_name: str, batch_data: List[Dict[str, Any]],
                      update_existing: bool, conflict_columns: List[str],
                      batch_index: int) -> Dict[str, Any]:
        """Process a single batch of data."""
        conn = self._get_connection()
        
        try:
            cursor = conn.cursor()
            inserted = 0
            updated = 0
            failed = 0
            errors = []
            
            # Generate SQL based on table and conflict resolution strategy
            insert_sql, update_sql = self._generate_sql(table_name, batch_data[0], conflict_columns, update_existing)
            
            for record in batch_data:
                try:
                    if update_existing:
                        # Try insert first, then update if conflict
                        try:
                            cursor.execute(insert_sql, list(record.values()))
                            inserted += 1
                        except sqlite3.IntegrityError:
                            # Record exists, try update
                            if update_sql:
                                cursor.execute(update_sql, list(record.values()) + [record[col] for col in conflict_columns])
                                updated += cursor.rowcount
                            else:
                                failed += 1
                    else:
                        # Insert only, ignore conflicts
                        cursor.execute(insert_sql, list(record.values()))
                        inserted += 1
                        
                except Exception as e:
                    failed += 1
                    errors.append(f"Record failed: {str(e)}")
            
            conn.commit()
            
            return {
                'batch_index': batch_index,
                'inserted': inserted,
                'updated': updated,
                'failed': failed,
                'errors': errors
            }
            
        finally:
            self._return_connection(conn)
    
    def _generate_sql(self, table_name: str, sample_record: Dict[str, Any],
                     conflict_columns: List[str], update_existing: bool) -> Tuple[str, Optional[str]]:
        """Generate SQL statements for insert and update operations."""
        columns = list(sample_record.keys())
        placeholders = ', '.join(['?' for _ in columns])
        
        # Insert SQL with conflict handling
        insert_sql = f"""
            INSERT OR {'REPLACE' if update_existing else 'IGNORE'} INTO {table_name} 
            ({', '.join(columns)}) 
            VALUES ({placeholders})
        """
        
        # Update SQL for explicit updates
        update_sql = None
        if update_existing and conflict_columns:
            set_clause = ', '.join([f"{col} = ?" for col in columns])
            where_clause = ' AND '.join([f"{col} = ?" for col in conflict_columns])
            
            update_sql = f"""
                UPDATE {table_name} 
                SET {set_clause}
                WHERE {where_clause}
            """
        
        return insert_sql, update_sql
    
    # Helper methods for data preparation
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to integer."""
        if pd.isna(value) or value is None or value == '':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(value) or value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_date(self, value) -> Optional[str]:
        """Parse date value to ISO format string."""
        if pd.isna(value) or value is None or value == '':
            return None
        try:
            if isinstance(value, str):
                # Try parsing various date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y']:
                    try:
                        parsed = datetime.strptime(value, fmt)
                        return parsed.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            return str(value)
        except:
            return None
    
    def _parse_datetime(self, value) -> str:
        """Parse datetime value to ISO format string."""
        if pd.isna(value) or value is None or value == '':
            return datetime.now().isoformat()
        try:
            if isinstance(value, str):
                parsed = pd.to_datetime(value)
                return parsed.isoformat()
            return str(value)
        except:
            return datetime.now().isoformat()
    
    def _calculate_popularity_score(self, row) -> float:
        """Calculate popularity score from available metrics."""
        vote_count = self._safe_float(row.get('Votes', row.get('No_of_Votes', 0))) or 0
        rating = self._safe_float(row.get('IMDBRating', row.get('IMDB_Rating', 0))) or 0
        
        # Simple popularity metric: vote_count * (rating / 10)
        return vote_count * (rating / 10.0) if rating > 0 else vote_count
    
    def _determine_activity_level(self, rating_count: int) -> str:
        """Determine user activity level based on rating count."""
        if rating_count >= 100:
            return 'very_active'
        elif rating_count >= 25:
            return 'active'
        elif rating_count >= 10:
            return 'moderate'
        elif rating_count >= 5:
            return 'casual'
        else:
            return 'minimal'
    
    def _determine_primary_genre(self, row) -> str:
        """Determine user's primary genre preference."""
        # Look for highest genre preference score
        genre_pref_cols = [col for col in row.index if col.startswith('genre_pref_')]
        
        if genre_pref_cols:
            max_col = max(genre_pref_cols, key=lambda x: row.get(x, 0))
            return max_col.replace('genre_pref_', '').title()
        
        return 'Unknown'
    
    def _extract_genre_vector(self, row) -> Optional[List[float]]:
        """Extract genre vector from row data."""
        genre_cols = [col for col in row.index if col.startswith('genre_onehot_')]
        
        if genre_cols:
            return [float(row.get(col, 0)) for col in genre_cols]
        
        return None
    
    def _determine_target_audience(self, row) -> str:
        """Determine target audience based on features."""
        rating = row.get('Certificate', '')
        
        if rating in ['G', 'PG']:
            return 'family'
        elif rating in ['PG-13']:
            return 'teen_adult'
        elif rating in ['R', 'NC-17']:
            return 'mature'
        else:
            return 'general'
    
    def _extract_genres_from_movies(self, movies_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract unique genres from movies data."""
        genres_set = set()
        
        for _, row in movies_df.iterrows():
            genres_str = row.get('Genres', '')
            if pd.notna(genres_str) and genres_str:
                genres = str(genres_str).split('|')
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        genres_set.add(genre)
        
        return [
            {
                'name': genre,
                'description': f'{genre} movies',
                'popularity_score': 0.0,
                'created_at': datetime.now().isoformat()
            }
            for genre in sorted(genres_set)
        ]
    
    def _prepare_movie_genre_relationships(self, movies_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare movie-genre relationship data."""
        relationships = []
        
        for _, row in movies_df.iterrows():
            movie_id = row.get('MovieID')
            genres_str = row.get('Genres', '')
            
            if pd.notna(genres_str) and genres_str and movie_id:
                genres = str(genres_str).split('|')
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        relationships.append({
                            'movie_id': int(movie_id),
                            'genre_id': f"SELECT id FROM genre WHERE name = '{genre}'",  # Will be resolved in SQL
                            'relevance_score': 1.0
                        })
        
        return relationships
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics."""
        total_attempted = sum(result.records_attempted for result in self.load_statistics.values())
        total_inserted = sum(result.records_inserted for result in self.load_statistics.values())
        total_updated = sum(result.records_updated for result in self.load_statistics.values())
        total_failed = sum(result.records_failed for result in self.load_statistics.values())
        total_time = sum(result.execution_time for result in self.load_statistics.values())
        
        return {
            'summary': {
                'total_attempted': total_attempted,
                'total_inserted': total_inserted,
                'total_updated': total_updated,
                'total_failed': total_failed,
                'success_rate': (total_inserted + total_updated) / total_attempted if total_attempted > 0 else 0,
                'total_execution_time': total_time,
                'average_records_per_second': total_attempted / total_time if total_time > 0 else 0
            },
            'by_table': {
                table: {
                    'records_attempted': result.records_attempted,
                    'records_inserted': result.records_inserted,
                    'records_updated': result.records_updated,
                    'records_failed': result.records_failed,
                    'execution_time': result.execution_time,
                    'records_per_second': result.records_attempted / result.execution_time if result.execution_time > 0 else 0,
                    'error_count': len(result.errors)
                }
                for table, result in self.load_statistics.items()
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        while not self.connection_pool.empty():
            try:
                conn = self.connection_pool.get_nowait()
                conn.close()
            except:
                pass