"""
Database Schema Creation and Management Module
Handles optimized table design with performance considerations for movie recommendation system.
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime


class DatabaseSchemaManager:
    """Optimized database schema creation and management for movie recommendation system."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.schema_version = "1.0.0"
        
    def create_complete_schema(self, drop_existing: bool = False) -> bool:
        """
        Create complete optimized database schema for the ETL pipeline.
        
        Args:
            drop_existing: Whether to drop existing tables
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys = ON")
                
                if drop_existing:
                    self._drop_all_tables(cursor)
                
                # Create tables in dependency order
                self._create_core_tables(cursor)
                self._create_relationship_tables(cursor)
                self._create_feature_tables(cursor)
                self._create_recommendation_tables(cursor)
                self._create_monitoring_tables(cursor)
                
                # Create indexes for performance
                self._create_performance_indexes(cursor)
                
                # Create views for common queries
                self._create_performance_views(cursor)
                
                # Insert metadata
                self._insert_schema_metadata(cursor)
                
                conn.commit()
                self.logger.info(f"Database schema created successfully at {self.db_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create database schema: {e}")
            return False
    
    def _drop_all_tables(self, cursor: sqlite3.Cursor):
        """Drop all existing tables."""
        tables_to_drop = [
            # Recommendation tables
            'user_recommendations', 'recommendation_cache', 'model_performance',
            # Feature tables  
            'user_profiles', 'movie_features', 'genre_features',
            # Relationship tables
            'user_rating', 'movie_genres', 'movie_countries', 'movie_languages', 
            'movie_cast', 'movie_crew',
            # Core tables
            'movie', 'genre', 'country', 'language', 'person', 'user',
            # Monitoring tables
            'etl_runs', 'data_quality_reports', 'schema_metadata'
        ]
        
        for table in tables_to_drop:
            cursor.execute(f'DROP TABLE IF EXISTS "{table}"')
    
    def _create_core_tables(self, cursor: sqlite3.Cursor):
        """Create core entity tables."""
        
        # Movies table - optimized for recommendation queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie" (
                "id" INTEGER PRIMARY KEY,
                "original_title" VARCHAR(500) NOT NULL,
                "english_title" VARCHAR(500) NOT NULL,
                "release_date" DATE,
                "runtime" INTEGER CHECK ("runtime" > 0),
                "budget" DECIMAL(15, 2) DEFAULT 0,
                "revenue" DECIMAL(15, 2) DEFAULT 0,
                "homepage" VARCHAR(500),
                "overview" TEXT,
                "poster_url" VARCHAR(500),
                "imdb_rating" DECIMAL(3, 1) CHECK ("imdb_rating" BETWEEN 1.0 AND 10.0),
                "vote_count" INTEGER DEFAULT 0,
                "popularity_score" DECIMAL(8, 4) DEFAULT 0.0,
                "certificate" VARCHAR(10),
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Computed columns for performance
                "release_year" INTEGER GENERATED ALWAYS AS (
                    CAST(strftime('%Y', "release_date") AS INTEGER)
                ) STORED,
                "budget_category" VARCHAR(10) GENERATED ALWAYS AS (
                    CASE
                        WHEN "budget" < 1000000 THEN 'low'
                        WHEN "budget" < 50000000 THEN 'medium'
                        WHEN "budget" < 200000000 THEN 'high'
                        ELSE 'blockbuster'
                    END
                ) STORED,
                "decade" VARCHAR(10) GENERATED ALWAYS AS (
                    CAST((CAST(strftime('%Y', "release_date") AS INTEGER) / 10) * 10 AS TEXT) || 's'
                ) STORED
            )
        ''')
        
        # Genres table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "genre" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "name" VARCHAR(100) UNIQUE NOT NULL,
                "description" TEXT,
                "popularity_score" DECIMAL(6, 4) DEFAULT 0.0,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Countries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "country" (
                "code" CHAR(2) PRIMARY KEY,
                "name" VARCHAR(200) UNIQUE NOT NULL,
                "region" VARCHAR(100),
                "movie_count" INTEGER DEFAULT 0,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Languages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "language" (
                "code" CHAR(2) PRIMARY KEY,
                "endonym" VARCHAR(200),
                "english_name" VARCHAR(200),
                "movie_count" INTEGER DEFAULT 0,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Persons table (actors, directors, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "person" (
                "id" VARCHAR(50) PRIMARY KEY,
                "name" VARCHAR(300) NOT NULL,
                "gender" INTEGER CHECK ("gender" BETWEEN 1 AND 3),
                "birth_date" DATE,
                "death_date" DATE,
                "biography" TEXT,
                "profile_image" VARCHAR(500),
                "popularity_score" DECIMAL(8, 4) DEFAULT 0.0,
                "primary_profession" VARCHAR(100),
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Users table (for user management)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "user" (
                "id" INTEGER PRIMARY KEY,
                "username" VARCHAR(100) UNIQUE,
                "email" VARCHAR(255) UNIQUE,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "last_active" TIMESTAMP,
                "rating_count" INTEGER DEFAULT 0,
                "avg_rating" DECIMAL(3, 2) DEFAULT 0.0,
                "preferences_updated" TIMESTAMP,
                "is_active" BOOLEAN DEFAULT TRUE
            )
        ''')
    
    def _create_relationship_tables(self, cursor: sqlite3.Cursor):
        """Create relationship/junction tables."""
        
        # User ratings - core table for recommendations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "user_rating" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "user_id" INTEGER NOT NULL,
                "movie_id" INTEGER NOT NULL,
                "rating" DECIMAL(3, 1) NOT NULL CHECK ("rating" BETWEEN 1.0 AND 10.0),
                "rating_date" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "source" VARCHAR(50) DEFAULT 'user_input',
                "confidence" DECIMAL(3, 2) DEFAULT 1.0,
                
                FOREIGN KEY ("user_id") REFERENCES "user" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                
                -- Ensure one rating per user-movie combination
                UNIQUE ("user_id", "movie_id")
            )
        ''')
        
        # Movie-Genre relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie_genres" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "movie_id" INTEGER NOT NULL,
                "genre_id" INTEGER NOT NULL,
                "relevance_score" DECIMAL(3, 2) DEFAULT 1.0,
                
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("genre_id") REFERENCES "genre" ("id") ON DELETE CASCADE,
                
                UNIQUE ("movie_id", "genre_id")
            )
        ''')
        
        # Movie-Country relationships (production countries)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie_countries" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "movie_id" INTEGER NOT NULL,
                "country_code" CHAR(2) NOT NULL,
                "role" VARCHAR(50) DEFAULT 'production',
                
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("country_code") REFERENCES "country" ("code"),
                
                UNIQUE ("movie_id", "country_code", "role")
            )
        ''')
        
        # Movie-Language relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie_languages" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "movie_id" INTEGER NOT NULL,
                "language_code" CHAR(2) NOT NULL,
                "role" VARCHAR(50) DEFAULT 'spoken',
                
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("language_code") REFERENCES "language" ("code"),
                
                UNIQUE ("movie_id", "language_code", "role")
            )
        ''')
        
        # Movie cast
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie_cast" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "movie_id" INTEGER NOT NULL,
                "person_id" VARCHAR(50) NOT NULL,
                "character_name" VARCHAR(500),
                "cast_order" INTEGER DEFAULT 999,
                "role_type" VARCHAR(50) DEFAULT 'actor',
                
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("person_id") REFERENCES "person" ("id"),
                
                UNIQUE ("movie_id", "person_id", "role_type")
            )
        ''')
        
        # Movie crew (directors, producers, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie_crew" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "movie_id" INTEGER NOT NULL,
                "person_id" VARCHAR(50) NOT NULL,
                "job_title" VARCHAR(200) NOT NULL,
                "department" VARCHAR(100),
                
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("person_id") REFERENCES "person" ("id"),
                
                UNIQUE ("movie_id", "person_id", "job_title")
            )
        ''')
    
    def _create_feature_tables(self, cursor: sqlite3.Cursor):
        """Create feature tables for ML/recommendation features."""
        
        # User profiles/features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "user_profiles" (
                "user_id" INTEGER PRIMARY KEY,
                "profile_data" JSON NOT NULL,
                "feature_version" VARCHAR(20) DEFAULT '1.0',
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Extracted key features for quick access
                "avg_rating" DECIMAL(3, 2),
                "rating_count" INTEGER,
                "genre_diversity" DECIMAL(3, 2),
                "activity_level" VARCHAR(20),
                "primary_genre" VARCHAR(100),
                
                FOREIGN KEY ("user_id") REFERENCES "user" ("id") ON DELETE CASCADE
            )
        ''')
        
        # Movie features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "movie_features" (
                "movie_id" INTEGER PRIMARY KEY,
                "feature_data" JSON NOT NULL,
                "feature_version" VARCHAR(20) DEFAULT '1.0',
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Extracted key features for quick access
                "popularity_score" DECIMAL(8, 4),
                "quality_score" DECIMAL(3, 2),
                "genre_vector" TEXT,  -- Serialized genre encoding
                "content_rating" VARCHAR(10),
                "target_audience" VARCHAR(50),
                
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE
            )
        ''')
        
        # Genre features and embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "genre_features" (
                "genre_id" INTEGER PRIMARY KEY,
                "embedding_vector" TEXT NOT NULL,  -- Serialized embedding
                "similarity_scores" JSON,
                "feature_version" VARCHAR(20) DEFAULT '1.0',
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY ("genre_id") REFERENCES "genre" ("id") ON DELETE CASCADE
            )
        ''')
    
    def _create_recommendation_tables(self, cursor: sqlite3.Cursor):
        """Create recommendation-specific tables."""
        
        # Pre-computed user recommendations cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "user_recommendations" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "user_id" INTEGER NOT NULL,
                "movie_id" INTEGER NOT NULL,
                "prediction_score" DECIMAL(4, 3) NOT NULL,
                "recommendation_type" VARCHAR(50) NOT NULL,
                "model_version" VARCHAR(20) DEFAULT '1.0',
                "rank_position" INTEGER,
                "explanation" TEXT,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "expires_at" TIMESTAMP,
                
                FOREIGN KEY ("user_id") REFERENCES "user" ("id") ON DELETE CASCADE,
                FOREIGN KEY ("movie_id") REFERENCES "movie" ("id") ON DELETE CASCADE,
                
                -- Unique combination per recommendation type
                UNIQUE ("user_id", "movie_id", "recommendation_type", "model_version")
            )
        ''')
        
        # Recommendation cache for popular/trending items
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "recommendation_cache" (
                "cache_key" VARCHAR(255) PRIMARY KEY,
                "recommendation_data" JSON NOT NULL,
                "cache_type" VARCHAR(50) NOT NULL,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "expires_at" TIMESTAMP NOT NULL,
                "hit_count" INTEGER DEFAULT 0,
                "last_accessed" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "model_performance" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "model_name" VARCHAR(100) NOT NULL,
                "model_version" VARCHAR(20) NOT NULL,
                "metric_name" VARCHAR(100) NOT NULL,
                "metric_value" DECIMAL(10, 6) NOT NULL,
                "evaluation_date" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "dataset_size" INTEGER,
                "notes" TEXT,
                
                UNIQUE ("model_name", "model_version", "metric_name", "evaluation_date")
            )
        ''')
    
    def _create_monitoring_tables(self, cursor: sqlite3.Cursor):
        """Create monitoring and logging tables."""
        
        # ETL pipeline runs tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "etl_runs" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "run_id" VARCHAR(100) UNIQUE NOT NULL,
                "pipeline_name" VARCHAR(100) NOT NULL,
                "start_time" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "end_time" TIMESTAMP,
                "status" VARCHAR(50) NOT NULL,
                "records_processed" INTEGER DEFAULT 0,
                "records_inserted" INTEGER DEFAULT 0,
                "records_updated" INTEGER DEFAULT 0,
                "records_failed" INTEGER DEFAULT 0,
                "error_message" TEXT,
                "execution_stats" JSON,
                "configuration" JSON
            )
        ''')
        
        # Data quality reports
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "data_quality_reports" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "report_id" VARCHAR(100) UNIQUE NOT NULL,
                "dataset_name" VARCHAR(100) NOT NULL,
                "check_type" VARCHAR(100) NOT NULL,
                "severity" VARCHAR(20) NOT NULL,
                "issue_count" INTEGER NOT NULL,
                "total_records" INTEGER NOT NULL,
                "quality_score" DECIMAL(5, 2),
                "report_data" JSON,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Schema metadata and versioning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "schema_metadata" (
                "id" INTEGER PRIMARY KEY AUTOINCREMENT,
                "schema_version" VARCHAR(20) NOT NULL,
                "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "description" TEXT,
                "migration_notes" TEXT
            )
        ''')
    
    def _create_performance_indexes(self, cursor: sqlite3.Cursor):
        """Create indexes for query performance optimization."""
        
        # Movie indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_movie_release_year ON movie (release_year)",
            "CREATE INDEX IF NOT EXISTS idx_movie_popularity ON movie (popularity_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_movie_rating ON movie (imdb_rating DESC)",
            "CREATE INDEX IF NOT EXISTS idx_movie_title ON movie (english_title)",
            "CREATE INDEX IF NOT EXISTS idx_movie_decade ON movie (decade)",
            
            # Rating indexes (critical for recommendation performance)
            "CREATE INDEX IF NOT EXISTS idx_rating_user ON user_rating (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_rating_movie ON user_rating (movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_rating_score ON user_rating (rating DESC)",
            "CREATE INDEX IF NOT EXISTS idx_rating_date ON user_rating (rating_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_rating_composite ON user_rating (user_id, movie_id, rating)",
            
            # Genre relationship indexes
            "CREATE INDEX IF NOT EXISTS idx_movie_genres_movie ON movie_genres (movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_movie_genres_genre ON movie_genres (genre_id)",
            
            # Cast and crew indexes
            "CREATE INDEX IF NOT EXISTS idx_cast_movie ON movie_cast (movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_cast_person ON movie_cast (person_id)",
            "CREATE INDEX IF NOT EXISTS idx_crew_movie ON movie_crew (movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_crew_person ON movie_crew (person_id)",
            
            # User profile indexes
            "CREATE INDEX IF NOT EXISTS idx_user_profiles_updated ON user_profiles (updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_user_activity ON user_profiles (activity_level)",
            
            # Recommendation indexes
            "CREATE INDEX IF NOT EXISTS idx_recommendations_user ON user_recommendations (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_movie ON user_recommendations (movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_score ON user_recommendations (prediction_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_type ON user_recommendations (recommendation_type)",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_expires ON user_recommendations (expires_at)",
            
            # Cache indexes
            "CREATE INDEX IF NOT EXISTS idx_cache_type ON recommendation_cache (cache_type)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON recommendation_cache (expires_at)",
            
            # Monitoring indexes
            "CREATE INDEX IF NOT EXISTS idx_etl_runs_status ON etl_runs (status)",
            "CREATE INDEX IF NOT EXISTS idx_etl_runs_time ON etl_runs (start_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_quality_reports_dataset ON data_quality_reports (dataset_name)",
            "CREATE INDEX IF NOT EXISTS idx_quality_reports_severity ON data_quality_reports (severity)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Failed to create index: {index_sql} - {e}")
    
    def _create_performance_views(self, cursor: sqlite3.Cursor):
        """Create views for common query patterns."""
        
        # Movie details with genres view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS movie_details AS
            SELECT 
                m.*,
                GROUP_CONCAT(g.name, '|') as genres,
                COUNT(DISTINCT ur.user_id) as rating_count,
                AVG(ur.rating) as avg_user_rating
            FROM movie m
            LEFT JOIN movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN genre g ON mg.genre_id = g.id
            LEFT JOIN user_rating ur ON m.id = ur.movie_id
            GROUP BY m.id
        ''')
        
        # User activity summary view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS user_activity AS
            SELECT 
                u.id,
                u.username,
                COUNT(ur.id) as total_ratings,
                AVG(ur.rating) as avg_rating,
                MIN(ur.rating_date) as first_rating,
                MAX(ur.rating_date) as last_rating,
                COUNT(DISTINCT DATE(ur.rating_date)) as active_days
            FROM user u
            LEFT JOIN user_rating ur ON u.id = ur.user_id
            GROUP BY u.id
        ''')
        
        # Popular movies view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS popular_movies AS
            SELECT 
                m.*,
                COUNT(ur.user_id) as rating_count,
                AVG(ur.rating) as avg_rating,
                (COUNT(ur.user_id) * AVG(ur.rating)) as popularity_metric
            FROM movie m
            LEFT JOIN user_rating ur ON m.id = ur.movie_id
            GROUP BY m.id
            HAVING COUNT(ur.user_id) >= 5
            ORDER BY popularity_metric DESC
        ''')
        
        # Recent recommendations view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS recent_recommendations AS
            SELECT 
                ur.*,
                m.english_title,
                m.release_year,
                u.username
            FROM user_recommendations ur
            JOIN movie m ON ur.movie_id = m.id
            JOIN user u ON ur.user_id = u.id
            WHERE ur.expires_at > CURRENT_TIMESTAMP
            ORDER BY ur.created_at DESC
        ''')
    
    def _insert_schema_metadata(self, cursor: sqlite3.Cursor):
        """Insert schema version metadata."""
        cursor.execute('''
            INSERT OR REPLACE INTO schema_metadata (schema_version, description)
            VALUES (?, ?)
        ''', (self.schema_version, 'Initial ETL pipeline schema with optimization for movie recommendations'))
    
    def verify_schema(self) -> Dict[str, Any]:
        """Verify schema integrity and return status report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check all tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                expected_tables = {
                    'movie', 'genre', 'country', 'language', 'person', 'user',
                    'user_rating', 'movie_genres', 'movie_countries', 'movie_languages',
                    'movie_cast', 'movie_crew', 'user_profiles', 'movie_features',
                    'genre_features', 'user_recommendations', 'recommendation_cache',
                    'model_performance', 'etl_runs', 'data_quality_reports',
                    'schema_metadata'
                }
                
                missing_tables = expected_tables - existing_tables
                extra_tables = existing_tables - expected_tables
                
                # Check indexes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                existing_indexes = {row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')}
                
                # Check views
                cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                existing_views = {row[0] for row in cursor.fetchall()}
                
                # Get schema version
                try:
                    cursor.execute("SELECT schema_version FROM schema_metadata ORDER BY id DESC LIMIT 1")
                    current_version = cursor.fetchone()[0]
                except:
                    current_version = "unknown"
                
                return {
                    'schema_valid': len(missing_tables) == 0,
                    'current_version': current_version,
                    'expected_version': self.schema_version,
                    'tables': {
                        'expected': len(expected_tables),
                        'existing': len(existing_tables),
                        'missing': list(missing_tables),
                        'extra': list(extra_tables)
                    },
                    'indexes': {
                        'count': len(existing_indexes),
                        'names': list(existing_indexes)
                    },
                    'views': {
                        'count': len(existing_views),
                        'names': list(existing_views)
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Schema verification failed: {e}")
            return {
                'schema_valid': False,
                'error': str(e)
            }
    
    def get_table_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tables."""
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        # Row count
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        row_count = cursor.fetchone()[0]
                        
                        # Column info
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = cursor.fetchall()
                        
                        stats[table] = {
                            'row_count': row_count,
                            'column_count': len(columns),
                            'columns': [
                                {
                                    'name': col[1],
                                    'type': col[2],
                                    'not_null': bool(col[3]),
                                    'default': col[4],
                                    'primary_key': bool(col[5])
                                }
                                for col in columns
                            ]
                        }
                        
                    except Exception as e:
                        stats[table] = {'error': str(e)}
                        
        except Exception as e:
            self.logger.error(f"Failed to get table statistics: {e}")
            
        return stats