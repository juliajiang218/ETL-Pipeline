"""
Main ETL Pipeline Orchestrator
Coordinates the complete Extract-Transform-Load pipeline for movie recommendation system.
"""

import logging
import sys
import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Import ETL components
from extract.data_sources.csv_extractor import CSVExtractor
from extract.data_sources.api_connector import APIConnector
from extract.data_sources.streaming_data import StreamingDataHandler
from extract.schemas.source_validation import DataQualityValidator

from transform.data_cleaning.normalization import DataNormalizer
from transform.data_cleaning.quality_validation import DataQualityEngine
from transform.data_cleaning.deduplication import DeduplicationEngine
from transform.feature_engineering.genre_encoding import GenreEncoder
from transform.feature_engineering.user_profiles import UserProfileBuilder

from load.database.schema_creation import DatabaseSchemaManager
from load.batch_processing.bulk_loader import BulkLoader


class ETLPipelineOrchestrator:
    """Main ETL pipeline orchestrator with comprehensive monitoring and error handling."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize pipeline components
        self.csv_extractor = CSVExtractor(config_path)
        self.api_connector = None
        self.streaming_handler = None
        self.data_validator = DataQualityValidator()
        
        self.data_normalizer = DataNormalizer()
        self.quality_engine = DataQualityEngine()
        self.dedup_engine = DeduplicationEngine()
        self.genre_encoder = GenreEncoder()
        self.profile_builder = UserProfileBuilder()
        
        self.schema_manager = DatabaseSchemaManager(self.config['database']['path'])
        self.bulk_loader = BulkLoader(
            db_path=self.config['database']['path'],
            batch_size=self.config.get('performance', {}).get('batch_size', 1000),
            max_workers=self.config.get('performance', {}).get('max_workers', 4)
        )
        
        # Pipeline state
        self.pipeline_id = f"etl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'status': 'initialized',
            'stages_completed': [],
            'stages_failed': [],
            'total_records_processed': 0,
            'errors': []
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration."""
        try:
            with open(config_path, 'r') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(file)
                else:
                    return json.load(file)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / f'etl_pipeline_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run_full_pipeline(self, stages: Optional[List[str]] = None) -> bool:
        """
        Run the complete ETL pipeline.
        
        Args:
            stages: Optional list of stages to run. If None, runs all stages.
            
        Returns:
            Success status
        """
        self.pipeline_stats['start_time'] = datetime.now()
        self.pipeline_stats['status'] = 'running'
        
        self.logger.info(f"Starting ETL pipeline {self.pipeline_id}")
        
        try:
            # Define pipeline stages
            pipeline_stages = [
                ('extract', self._run_extract_stage),
                ('validate', self._run_validation_stage),
                ('transform', self._run_transform_stage),
                ('load', self._run_load_stage),
                ('finalize', self._run_finalization_stage)
            ]
            
            # Filter stages if specified
            if stages:
                pipeline_stages = [(name, func) for name, func in pipeline_stages if name in stages]
            
            # Execute stages
            for stage_name, stage_func in pipeline_stages:
                self.logger.info(f"Starting stage: {stage_name}")
                
                try:
                    success = stage_func()
                    if success:
                        self.pipeline_stats['stages_completed'].append(stage_name)
                        self.logger.info(f"Stage {stage_name} completed successfully")
                    else:
                        self.pipeline_stats['stages_failed'].append(stage_name)
                        self.logger.error(f"Stage {stage_name} failed")
                        
                        if self.config.get('pipeline', {}).get('stop_on_error', True):
                            break
                            
                except Exception as e:
                    error_msg = f"Stage {stage_name} failed with exception: {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())
                    self.pipeline_stats['stages_failed'].append(stage_name)
                    self.pipeline_stats['errors'].append(error_msg)
                    
                    if self.config.get('pipeline', {}).get('stop_on_error', True):
                        break
            
            # Determine final status
            if self.pipeline_stats['stages_failed']:
                self.pipeline_stats['status'] = 'failed'
                success = False
            else:
                self.pipeline_stats['status'] = 'completed'
                success = True
                
        except Exception as e:
            self.pipeline_stats['status'] = 'error'
            self.pipeline_stats['errors'].append(str(e))
            self.logger.error(f"Pipeline failed with exception: {e}")
            self.logger.error(traceback.format_exc())
            success = False
        
        finally:
            self.pipeline_stats['end_time'] = datetime.now()
            self._log_pipeline_summary()
            self.bulk_loader.cleanup()
        
        return success
    
    def _run_extract_stage(self) -> bool:
        """Execute the extract stage."""
        try:
            self.logger.info("Starting data extraction")
            
            # Setup database schema
            if not self.schema_manager.create_complete_schema(
                drop_existing=self.config.get('database', {}).get('recreate_schema', False)
            ):
                self.logger.error("Failed to create database schema")
                return False
            
            # Extract from CSV sources
            csv_sources = self.config.get('data_sources', {}).get('csv_files', {})
            extracted_data = self.csv_extractor.extract_all_sources(csv_sources)
            
            if not extracted_data:
                self.logger.error("No data extracted from CSV sources")
                return False
            
            # Store extracted data for next stage
            self.extracted_data = extracted_data
            
            # Log extraction statistics
            extraction_stats = self.csv_extractor.get_extraction_stats(extracted_data)
            self.logger.info(f"Extraction completed: {json.dumps(extraction_stats, indent=2)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Extract stage failed: {e}")
            return False
    
    def _run_validation_stage(self) -> bool:
        """Execute the validation stage."""
        try:
            self.logger.info("Starting data validation")
            
            # Validate all extracted datasets
            validation_results = {}
            
            for source_name, df in self.extracted_data.items():
                self.logger.info(f"Validating {source_name}")
                
                # Determine dataset type for validation
                if 'movies' in source_name.lower():
                    results = self.data_validator.validate_movies_schema(df)
                elif 'ratings' in source_name.lower():
                    results = self.data_validator.validate_ratings_schema(df)
                elif 'persons' in source_name.lower():
                    results = self.data_validator.validate_persons_schema(df)
                elif 'imdb' in source_name.lower():
                    results = self.data_validator.validate_imdb_schema(df)
                else:
                    continue
                
                validation_results[source_name] = results
                
                # Check for critical issues
                critical_issues = [r for r in results if r.level.name == 'CRITICAL']
                if critical_issues and self.config.get('validation', {}).get('fail_on_critical', True):
                    self.logger.error(f"Critical validation issues found in {source_name}")
                    return False
            
            # Generate validation report
            total_issues = sum(len(results) for results in validation_results.values())
            self.logger.info(f"Validation completed with {total_issues} total issues found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation stage failed: {e}")
            return False
    
    def _run_transform_stage(self) -> bool:
        """Execute the transform stage."""
        try:
            self.logger.info("Starting data transformation")
            
            self.transformed_data = {}
            
            # Process each dataset
            for source_name, df in self.extracted_data.items():
                self.logger.info(f"Transforming {source_name}")
                
                # 1. Data Normalization
                if 'movies' in source_name.lower():
                    normalized_df = self.data_normalizer.normalize_movies_data(df)
                elif 'ratings' in source_name.lower():
                    normalized_df = self.data_normalizer.normalize_ratings_data(df)
                elif 'persons' in source_name.lower():
                    normalized_df = self.data_normalizer.normalize_persons_data(df)
                elif 'imdb' in source_name.lower():
                    normalized_df = self.data_normalizer.normalize_imdb_data(df)
                else:
                    normalized_df = df
                
                # 2. Data Quality Validation (with fixes)
                dataset_type = self._determine_dataset_type(source_name)
                quality_issues = self.quality_engine.validate_dataset(normalized_df, dataset_type)
                
                if quality_issues:
                    normalized_df = self.quality_engine.apply_fixes(
                        normalized_df, 
                        dataset_type,
                        fix_level=self.quality_engine.ValidationSeverity.HIGH
                    )
                
                # 3. Deduplication
                if 'movies' in source_name.lower():
                    deduplicated_df = self.dedup_engine.deduplicate_movies(normalized_df)
                elif 'ratings' in source_name.lower():
                    deduplicated_df = self.dedup_engine.deduplicate_ratings(normalized_df)
                elif 'persons' in source_name.lower():
                    deduplicated_df = self.dedup_engine.deduplicate_persons(normalized_df)
                else:
                    deduplicated_df = normalized_df
                
                self.transformed_data[source_name] = deduplicated_df
                
                self.logger.info(f"Transformed {source_name}: {len(df)} -> {len(deduplicated_df)} records")
            
            # 4. Feature Engineering
            if self._has_movies_and_ratings():
                self.logger.info("Starting feature engineering")
                
                # Genre encoding
                movies_df = self._get_movies_dataframe()
                movies_with_genres, genre_metadata = self.genre_encoder.fit_transform_movies(movies_df)
                self.transformed_data['movies_with_features'] = movies_with_genres
                
                # User profiles
                ratings_df = self._get_ratings_dataframe()
                if not ratings_df.empty and not movies_with_genres.empty:
                    user_profiles, profile_metadata = self.profile_builder.build_user_profiles(
                        ratings_df, movies_with_genres
                    )
                    self.transformed_data['user_profiles'] = user_profiles
                    
                    self.logger.info(f"Generated {len(user_profiles)} user profiles")
            
            # Update total records processed
            total_records = sum(len(df) for df in self.transformed_data.values())
            self.pipeline_stats['total_records_processed'] = total_records
            
            self.logger.info(f"Transform stage completed, processed {total_records} records")
            return True
            
        except Exception as e:
            self.logger.error(f"Transform stage failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_load_stage(self) -> bool:
        """Execute the load stage."""
        try:
            self.logger.info("Starting data loading")
            
            load_results = {}
            
            # Load core entities first
            if 'movies' in self.transformed_data or 'movies_with_features' in self.transformed_data:
                movies_df = self.transformed_data.get('movies_with_features', self.transformed_data.get('movies'))
                if not movies_df.empty:
                    result = self.bulk_loader.bulk_load_movies(movies_df, update_existing=True)
                    load_results['movies'] = result
                    self.logger.info(f"Loaded movies: {result.records_inserted} inserted, {result.records_failed} failed")
            
            # Load genres (extracted from movies)
            if 'movies' in self.transformed_data or 'movies_with_features' in self.transformed_data:
                movies_df = self.transformed_data.get('movies_with_features', self.transformed_data.get('movies'))
                genre_results = self.bulk_loader.bulk_load_genres(movies_df)
                load_results.update(genre_results)
            
            # Load persons and relationships
            persons_data = {}
            for key in ['persons', 'cast', 'crew']:
                if key in self.transformed_data:
                    persons_data[key] = self.transformed_data[key]
            
            if persons_data:
                person_results = self.bulk_loader.bulk_load_persons(**persons_data)
                load_results.update(person_results)
            
            # Load ratings
            if 'ratings' in self.transformed_data:
                ratings_df = self.transformed_data['ratings']
                if not ratings_df.empty:
                    result = self.bulk_loader.bulk_load_ratings(ratings_df, update_existing=False)
                    load_results['ratings'] = result
                    self.logger.info(f"Loaded ratings: {result.records_inserted} inserted, {result.records_failed} failed")
            
            # Load features
            feature_data = {}
            if 'user_profiles' in self.transformed_data:
                feature_data['user_profiles_df'] = self.transformed_data['user_profiles']
            if 'movies_with_features' in self.transformed_data:
                feature_data['movie_features_df'] = self.transformed_data['movies_with_features']
            
            if feature_data:
                feature_results = self.bulk_loader.bulk_load_features(**feature_data)
                load_results.update(feature_results)
            
            # Log loading statistics
            load_stats = self.bulk_loader.get_load_statistics()
            self.logger.info(f"Load stage completed: {json.dumps(load_stats['summary'], indent=2)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Load stage failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_finalization_stage(self) -> bool:
        """Execute finalization tasks."""
        try:
            self.logger.info("Running finalization tasks")
            
            # Verify database integrity
            schema_status = self.schema_manager.verify_schema()
            if not schema_status.get('schema_valid', False):
                self.logger.warning(f"Schema validation issues: {schema_status}")
            
            # Get table statistics
            table_stats = self.schema_manager.get_table_statistics()
            self.logger.info(f"Database statistics: {json.dumps({k: v.get('row_count', 0) for k, v in table_stats.items()}, indent=2)}")
            
            # Generate final reports
            self._generate_pipeline_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Finalization stage failed: {e}")
            return False
    
    def _determine_dataset_type(self, source_name: str) -> str:
        """Determine dataset type from source name."""
        source_lower = source_name.lower()
        if 'movies' in source_lower or 'imdb' in source_lower:
            return 'movies'
        elif 'ratings' in source_lower:
            return 'ratings'
        elif 'persons' in source_lower:
            return 'persons'
        else:
            return 'unknown'
    
    def _has_movies_and_ratings(self) -> bool:
        """Check if we have both movies and ratings data for feature engineering."""
        has_movies = any('movies' in key.lower() for key in self.transformed_data.keys())
        has_ratings = any('ratings' in key.lower() for key in self.transformed_data.keys())
        return has_movies and has_ratings
    
    def _get_movies_dataframe(self):
        """Get the movies DataFrame from transformed data."""
        for key, df in self.transformed_data.items():
            if 'movies' in key.lower():
                return df
        return None
    
    def _get_ratings_dataframe(self):
        """Get the ratings DataFrame from transformed data."""
        for key, df in self.transformed_data.items():
            if 'ratings' in key.lower():
                return df
        return None
    
    def _generate_pipeline_report(self):
        """Generate comprehensive pipeline execution report."""
        report = {
            'pipeline_id': self.pipeline_id,
            'execution_summary': self.pipeline_stats,
            'normalization_stats': self.data_normalizer.get_normalization_stats(),
            'deduplication_stats': self.dedup_engine.generate_deduplication_report(),
            'loading_stats': self.bulk_loader.get_load_statistics(),
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_path = Path('reports') / f'pipeline_report_{self.pipeline_id}.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline report saved to {report_path}")
    
    def _log_pipeline_summary(self):
        """Log comprehensive pipeline execution summary."""
        duration = None
        if self.pipeline_stats['start_time'] and self.pipeline_stats['end_time']:
            duration = (self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']).total_seconds()
        
        summary = f"""
        
        ========== ETL PIPELINE EXECUTION SUMMARY ==========
        Pipeline ID: {self.pipeline_id}
        Status: {self.pipeline_stats['status'].upper()}
        Duration: {duration:.2f} seconds
        
        Stages Completed: {', '.join(self.pipeline_stats['stages_completed'])}
        Stages Failed: {', '.join(self.pipeline_stats['stages_failed']) if self.pipeline_stats['stages_failed'] else 'None'}
        
        Total Records Processed: {self.pipeline_stats['total_records_processed']:,}
        
        Errors: {len(self.pipeline_stats['errors'])}
        {chr(10).join(f"  - {error}" for error in self.pipeline_stats['errors'][:5])}
        {'  ... and more' if len(self.pipeline_stats['errors']) > 5 else ''}
        
        ===================================================
        """
        
        if self.pipeline_stats['status'] == 'completed':
            self.logger.info(summary)
        else:
            self.logger.error(summary)


def main():
    """Main entry point for ETL pipeline."""
    parser = argparse.ArgumentParser(description='Movie Recommendation ETL Pipeline')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--stages', '-s', nargs='+', 
                       choices=['extract', 'validate', 'transform', 'load', 'finalize'],
                       help='Specific stages to run (default: all)')
    parser.add_argument('--log-level', '-l', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = ETLPipelineOrchestrator(args.config)
        
        # Override log level if specified
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        success = pipeline.run_full_pipeline(stages=args.stages)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Pipeline failed with exception: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()