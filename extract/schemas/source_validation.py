"""
Data Quality and Schema Validation for ETL Extract Layer
Validates extracted data against expected schemas and quality standards.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    level: ValidationLevel
    message: str
    column: Optional[str] = None
    row_count: Optional[int] = None
    failed_values: Optional[List[Any]] = None


class DataQualityValidator:
    """Comprehensive data quality validation and schema checking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []
    
    def validate_movies_schema(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate movie data schema and quality.
        
        Args:
            df: Movies DataFrame to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        # Required columns check
        required_cols = ['MovieID', 'OriginalTitle', 'EnglishTitle']
        results.extend(self._check_required_columns(df, required_cols, 'movies'))
        
        # Data type validations
        if 'MovieID' in df.columns:
            results.extend(self._validate_integer_column(df, 'MovieID', positive=True))
        
        if 'Runtime' in df.columns:
            results.extend(self._validate_numeric_range(df, 'Runtime', 1, 1000, 'minutes'))
        
        if 'Budget' in df.columns:
            results.extend(self._validate_numeric_range(df, 'Budget', 0, 1e9, 'USD'))
        
        if 'Revenue' in df.columns:
            results.extend(self._validate_numeric_range(df, 'Revenue', 0, 1e10, 'USD'))
        
        # String validations
        for col in ['OriginalTitle', 'EnglishTitle']:
            if col in df.columns:
                results.extend(self._validate_string_column(df, col, min_length=1, max_length=500))
        
        # Date validations
        if 'ReleaseDate' in df.columns:
            results.extend(self._validate_date_column(df, 'ReleaseDate'))
        
        # Genre format validation
        if 'Genres' in df.columns:
            results.extend(self._validate_genre_format(df, 'Genres'))
        
        # Duplicate check
        results.extend(self._check_duplicates(df, ['MovieID'], 'movies'))
        
        return results
    
    def validate_ratings_schema(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate ratings data schema and quality.
        
        Args:
            df: Ratings DataFrame to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        # Required columns check
        required_cols = ['UserID', 'MovieID', 'Rating']
        results.extend(self._check_required_columns(df, required_cols, 'ratings'))
        
        # Data type validations
        for col in ['UserID', 'MovieID']:
            if col in df.columns:
                results.extend(self._validate_integer_column(df, col, positive=True))
        
        # Rating value validation
        if 'Rating' in df.columns:
            results.extend(self._validate_rating_values(df, 'Rating'))
        
        # Date validation
        if 'Date' in df.columns:
            results.extend(self._validate_date_column(df, 'Date'))
        
        # Logical validations
        results.extend(self._validate_rating_distribution(df))
        results.extend(self._check_duplicates(df, ['UserID', 'MovieID'], 'ratings'))
        
        return results
    
    def validate_persons_schema(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate persons data schema and quality.
        
        Args:
            df: Persons DataFrame to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        # Required columns check
        required_cols = ['CastID', 'Name', 'MovieID']
        results.extend(self._check_required_columns(df, required_cols, 'persons'))
        
        # Data validations
        if 'MovieID' in df.columns:
            results.extend(self._validate_integer_column(df, 'MovieID', positive=True))
        
        if 'Name' in df.columns:
            results.extend(self._validate_string_column(df, 'Name', min_length=1, max_length=200))
        
        if 'Gender' in df.columns:
            results.extend(self._validate_gender_values(df, 'Gender'))
        
        # Duplicate check
        results.extend(self._check_duplicates(df, ['CastID', 'MovieID'], 'persons'))
        
        return results
    
    def validate_imdb_schema(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate IMDb data schema and quality.
        
        Args:
            df: IMDb DataFrame to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        # Required columns check
        required_cols = ['Series_Title', 'Released_Year']
        results.extend(self._check_required_columns(df, required_cols, 'imdb'))
        
        # Year validation
        if 'Released_Year' in df.columns:
            results.extend(self._validate_year_column(df, 'Released_Year'))
        
        # Rating validation
        if 'IMDB_Rating' in df.columns:
            results.extend(self._validate_numeric_range(df, 'IMDB_Rating', 1.0, 10.0, 'rating'))
        
        # Votes validation
        if 'No_of_Votes' in df.columns:
            results.extend(self._validate_integer_column(df, 'No_of_Votes', positive=True))
        
        return results
    
    def _check_required_columns(self, df: pd.DataFrame, required_cols: List[str], 
                               dataset_name: str) -> List[ValidationResult]:
        """Check if all required columns are present."""
        results = []
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Missing required columns in {dataset_name}: {missing_cols}"
            ))
        else:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                message=f"All required columns present in {dataset_name}"
            ))
        
        return results
    
    def _validate_integer_column(self, df: pd.DataFrame, column: str, 
                                positive: bool = False) -> List[ValidationResult]:
        """Validate integer column data types and constraints."""
        results = []
        
        if column not in df.columns:
            return results
        
        # Check for non-numeric values
        non_numeric = df[~pd.to_numeric(df[column], errors='coerce').notna()]
        if len(non_numeric) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Non-numeric values found in {column}",
                column=column,
                row_count=len(non_numeric)
            ))
        
        # Check for positive values if required
        if positive:
            numeric_df = pd.to_numeric(df[column], errors='coerce')
            negative_values = df[numeric_df <= 0]
            if len(negative_values) > 0:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Non-positive values found in {column}",
                    column=column,
                    row_count=len(negative_values)
                ))
        
        return results
    
    def _validate_numeric_range(self, df: pd.DataFrame, column: str, 
                               min_val: float, max_val: float, 
                               unit: str = "") -> List[ValidationResult]:
        """Validate numeric column falls within expected range."""
        results = []
        
        if column not in df.columns:
            return results
        
        numeric_col = pd.to_numeric(df[column], errors='coerce')
        out_of_range = df[(numeric_col < min_val) | (numeric_col > max_val)]
        
        if len(out_of_range) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Values outside expected range ({min_val}-{max_val} {unit}) in {column}",
                column=column,
                row_count=len(out_of_range)
            ))
        
        return results
    
    def _validate_string_column(self, df: pd.DataFrame, column: str, 
                               min_length: int = 1, max_length: int = 1000) -> List[ValidationResult]:
        """Validate string column constraints."""
        results = []
        
        if column not in df.columns:
            return results
        
        # Check for null/empty strings
        null_empty = df[(df[column].isna()) | (df[column].str.strip() == '')]
        if len(null_empty) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Null or empty values found in {column}",
                column=column,
                row_count=len(null_empty)
            ))
        
        # Check string lengths
        valid_strings = df[df[column].notna() & (df[column].str.strip() != '')]
        if len(valid_strings) > 0:
            too_short = valid_strings[valid_strings[column].str.len() < min_length]
            too_long = valid_strings[valid_strings[column].str.len() > max_length]
            
            if len(too_short) > 0:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Strings shorter than {min_length} characters in {column}",
                    column=column,
                    row_count=len(too_short)
                ))
            
            if len(too_long) > 0:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Strings longer than {max_length} characters in {column}",
                    column=column,
                    row_count=len(too_long)
                ))
        
        return results
    
    def _validate_date_column(self, df: pd.DataFrame, column: str) -> List[ValidationResult]:
        """Validate date column format and values."""
        results = []
        
        if column not in df.columns:
            return results
        
        # Try to parse dates
        try:
            pd.to_datetime(df[column], errors='raise')
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                message=f"Date format valid in {column}",
                column=column
            ))
        except:
            # Check how many dates are invalid
            valid_dates = pd.to_datetime(df[column], errors='coerce')
            invalid_dates = df[valid_dates.isna() & df[column].notna()]
            
            if len(invalid_dates) > 0:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Invalid date format in {column}",
                    column=column,
                    row_count=len(invalid_dates)
                ))
        
        return results
    
    def _validate_rating_values(self, df: pd.DataFrame, column: str) -> List[ValidationResult]:
        """Validate rating values are within expected range."""
        results = []
        
        if column not in df.columns:
            return results
        
        numeric_ratings = pd.to_numeric(df[column], errors='coerce')
        
        # Check for values outside 0.5-5.0 range (original scale)
        out_of_range = df[(numeric_ratings < 0.5) | (numeric_ratings > 10.0)]
        if len(out_of_range) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Rating values outside valid range (0.5-10.0) in {column}",
                column=column,
                row_count=len(out_of_range)
            ))
        
        return results
    
    def _validate_genre_format(self, df: pd.DataFrame, column: str) -> List[ValidationResult]:
        """Validate genre column format (pipe-separated values)."""
        results = []
        
        if column not in df.columns:
            return results
        
        # Check for valid genre format (pipe-separated)
        genre_pattern = re.compile(r'^[A-Za-z\s]+(\|[A-Za-z\s]+)*$')
        valid_genres = df[df[column].notna()]
        
        if len(valid_genres) > 0:
            invalid_format = valid_genres[~valid_genres[column].str.match(genre_pattern, na=False)]
            
            if len(invalid_format) > 0:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Invalid genre format in {column} (expected pipe-separated)",
                    column=column,
                    row_count=len(invalid_format)
                ))
        
        return results
    
    def _validate_gender_values(self, df: pd.DataFrame, column: str) -> List[ValidationResult]:
        """Validate gender column values."""
        results = []
        
        if column not in df.columns:
            return results
        
        valid_genders = {1, 2, 3}  # Assuming 1=Female, 2=Male, 3=Other
        numeric_gender = pd.to_numeric(df[column], errors='coerce')
        invalid_genders = df[~numeric_gender.isin(valid_genders) & df[column].notna()]
        
        if len(invalid_genders) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Invalid gender values in {column} (expected 1, 2, or 3)",
                column=column,
                row_count=len(invalid_genders)
            ))
        
        return results
    
    def _validate_year_column(self, df: pd.DataFrame, column: str) -> List[ValidationResult]:
        """Validate year column values."""
        results = []
        
        if column not in df.columns:
            return results
        
        current_year = datetime.now().year
        numeric_year = pd.to_numeric(df[column], errors='coerce')
        
        # Check for reasonable year range
        invalid_years = df[(numeric_year < 1888) | (numeric_year > current_year + 5)]
        
        if len(invalid_years) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Unrealistic year values in {column}",
                column=column,
                row_count=len(invalid_years)
            ))
        
        return results
    
    def _validate_rating_distribution(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate rating distribution looks reasonable."""
        results = []
        
        if 'Rating' not in df.columns:
            return results
        
        numeric_ratings = pd.to_numeric(df['Rating'], errors='coerce')
        rating_std = numeric_ratings.std()
        
        # Check if all ratings are the same (suspicious)
        if rating_std == 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="All ratings have the same value - suspicious distribution",
                column='Rating'
            ))
        
        # Check for very skewed distributions
        rating_mean = numeric_ratings.mean()
        if rating_mean > 4.5:  # On 1-10 scale, mean > 9
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="Rating distribution heavily skewed towards high values",
                column='Rating'
            ))
        
        return results
    
    def _check_duplicates(self, df: pd.DataFrame, key_columns: List[str], 
                         dataset_name: str) -> List[ValidationResult]:
        """Check for duplicate records based on key columns."""
        results = []
        
        # Check if all key columns exist
        missing_cols = [col for col in key_columns if col not in df.columns]
        if missing_cols:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Cannot check duplicates - missing columns: {missing_cols}",
                column=str(key_columns)
            ))
            return results
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=key_columns, keep=False)]
        
        if len(duplicates) > 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Duplicate records found in {dataset_name}",
                column=str(key_columns),
                row_count=len(duplicates)
            ))
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': len(results),
            'summary': {
                'errors': len([r for r in results if r.level == ValidationLevel.ERROR]),
                'warnings': len([r for r in results if r.level == ValidationLevel.WARNING]),
                'info': len([r for r in results if r.level == ValidationLevel.INFO])
            },
            'details': []
        }
        
        for result in results:
            detail = {
                'level': result.level.value,
                'message': result.message,
                'column': result.column,
                'affected_rows': result.row_count
            }
            report['details'].append(detail)
        
        return report
    
    def validate_all_sources(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Validate all data sources and return comprehensive results.
        
        Args:
            dataframes: Dictionary mapping source names to DataFrames
            
        Returns:
            Dictionary of validation reports for each source
        """
        all_reports = {}
        
        for source_name, df in dataframes.items():
            self.logger.info(f"Validating {source_name} with {len(df)} rows")
            
            if 'movies' in source_name.lower():
                results = self.validate_movies_schema(df)
            elif 'ratings' in source_name.lower():
                results = self.validate_ratings_schema(df)
            elif 'persons' in source_name.lower():
                results = self.validate_persons_schema(df)
            elif 'imdb' in source_name.lower():
                results = self.validate_imdb_schema(df)
            else:
                # Generic validation for unknown sources
                results = [ValidationResult(
                    level=ValidationLevel.INFO,
                    message=f"Generic validation for {source_name}: {len(df)} rows, {len(df.columns)} columns"
                )]
            
            all_reports[source_name] = self.generate_validation_report(results)
        
        return all_reports