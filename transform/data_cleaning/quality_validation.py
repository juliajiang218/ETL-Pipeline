"""
Advanced Data Quality Validation Module
Implements comprehensive data quality rules and business logic validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re


class ValidationSeverity(Enum):
    """Data quality issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QualityRule:
    """Data quality rule definition."""
    name: str
    description: str
    severity: ValidationSeverity
    columns: List[str]
    condition: str
    action: str = "flag"  # flag, fix, remove


@dataclass
class QualityIssue:
    """Data quality issue found during validation."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    affected_rows: int
    affected_columns: List[str]
    sample_values: List[Any]
    recommendation: str


class DataQualityEngine:
    """Advanced data quality validation with business rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_rules = self._initialize_quality_rules()
        self.validation_results = {}
    
    def _initialize_quality_rules(self) -> Dict[str, List[QualityRule]]:
        """Initialize comprehensive quality rules for each data type."""
        rules = {
            'movies': [
                QualityRule(
                    name="required_movie_fields",
                    description="Critical movie fields must not be null",
                    severity=ValidationSeverity.CRITICAL,
                    columns=["MovieID", "OriginalTitle", "EnglishTitle"],
                    condition="not_null",
                    action="remove"
                ),
                QualityRule(
                    name="valid_release_dates",
                    description="Release dates should be realistic (1888-2030)",
                    severity=ValidationSeverity.HIGH,
                    columns=["ReleaseDate"],
                    condition="date_range",
                    action="flag"
                ),
                QualityRule(
                    name="reasonable_runtime",
                    description="Movie runtime should be between 1-1000 minutes",
                    severity=ValidationSeverity.MEDIUM,
                    columns=["Runtime"],
                    condition="numeric_range",
                    action="flag"
                ),
                QualityRule(
                    name="budget_revenue_consistency",
                    description="Revenue should not be significantly less than budget for successful movies",
                    severity=ValidationSeverity.LOW,
                    columns=["Budget", "Revenue"],
                    condition="business_logic",
                    action="flag"
                ),
                QualityRule(
                    name="duplicate_movies",
                    description="Movies should not have duplicate IDs",
                    severity=ValidationSeverity.CRITICAL,
                    columns=["MovieID"],
                    condition="unique",
                    action="remove"
                ),
                QualityRule(
                    name="title_consistency",
                    description="Original and English titles should be reasonably similar",
                    severity=ValidationSeverity.MEDIUM,
                    columns=["OriginalTitle", "EnglishTitle"],
                    condition="text_similarity",
                    action="flag"
                )
            ],
            'ratings': [
                QualityRule(
                    name="required_rating_fields",
                    description="Critical rating fields must not be null",
                    severity=ValidationSeverity.CRITICAL,
                    columns=["UserID", "MovieID", "Rating"],
                    condition="not_null",
                    action="remove"
                ),
                QualityRule(
                    name="valid_rating_range",
                    description="Ratings must be within valid range (1-10)",
                    severity=ValidationSeverity.CRITICAL,
                    columns=["Rating"],
                    condition="numeric_range",
                    action="remove"
                ),
                QualityRule(
                    name="duplicate_user_movie_ratings",
                    description="Users should not have duplicate ratings for the same movie",
                    severity=ValidationSeverity.HIGH,
                    columns=["UserID", "MovieID"],
                    condition="unique_combination",
                    action="fix"
                ),
                QualityRule(
                    name="rating_temporal_consistency",
                    description="Rating dates should be after movie release dates",
                    severity=ValidationSeverity.MEDIUM,
                    columns=["Date"],
                    condition="temporal_logic",
                    action="flag"
                ),
                QualityRule(
                    name="user_rating_patterns",
                    description="Users with suspicious rating patterns (all same rating)",
                    severity=ValidationSeverity.LOW,
                    columns=["UserID", "Rating"],
                    condition="pattern_analysis",
                    action="flag"
                )
            ],
            'persons': [
                QualityRule(
                    name="required_person_fields",
                    description="Critical person fields must not be null",
                    severity=ValidationSeverity.CRITICAL,
                    columns=["CastID", "Name", "MovieID"],
                    condition="not_null",
                    action="remove"
                ),
                QualityRule(
                    name="valid_gender_values",
                    description="Gender values must be valid (1, 2, or 3)",
                    severity=ValidationSeverity.MEDIUM,
                    columns=["Gender"],
                    condition="categorical_values",
                    action="flag"
                ),
                QualityRule(
                    name="reasonable_character_names",
                    description="Character names should be reasonable length and format",
                    severity=ValidationSeverity.LOW,
                    columns=["Character"],
                    condition="text_format",
                    action="flag"
                ),
                QualityRule(
                    name="duplicate_cast_roles",
                    description="Same actor should not have duplicate roles in same movie",
                    severity=ValidationSeverity.MEDIUM,
                    columns=["CastID", "MovieID"],
                    condition="unique_combination",
                    action="flag"
                )
            ]
        }
        return rules
    
    def validate_dataset(self, df: pd.DataFrame, dataset_type: str) -> List[QualityIssue]:
        """
        Validate dataset against quality rules.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset ('movies', 'ratings', 'persons')
            
        Returns:
            List of quality issues found
        """
        issues = []
        rules = self.quality_rules.get(dataset_type, [])
        
        self.logger.info(f"Validating {dataset_type} dataset with {len(rules)} rules")
        
        for rule in rules:
            try:
                rule_issues = self._apply_quality_rule(df, rule)
                issues.extend(rule_issues)
            except Exception as e:
                self.logger.error(f"Error applying rule {rule.name}: {e}")
        
        # Store results
        self.validation_results[dataset_type] = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]),
            'high_issues': len([i for i in issues if i.severity == ValidationSeverity.HIGH]),
            'issues': issues
        }
        
        return issues
    
    def _apply_quality_rule(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Apply a specific quality rule to the DataFrame."""
        issues = []
        
        if rule.condition == "not_null":
            issues.extend(self._check_not_null(df, rule))
        elif rule.condition == "numeric_range":
            issues.extend(self._check_numeric_range(df, rule))
        elif rule.condition == "date_range":
            issues.extend(self._check_date_range(df, rule))
        elif rule.condition == "unique":
            issues.extend(self._check_unique(df, rule))
        elif rule.condition == "unique_combination":
            issues.extend(self._check_unique_combination(df, rule))
        elif rule.condition == "categorical_values":
            issues.extend(self._check_categorical_values(df, rule))
        elif rule.condition == "text_format":
            issues.extend(self._check_text_format(df, rule))
        elif rule.condition == "text_similarity":
            issues.extend(self._check_text_similarity(df, rule))
        elif rule.condition == "business_logic":
            issues.extend(self._check_business_logic(df, rule))
        elif rule.condition == "temporal_logic":
            issues.extend(self._check_temporal_logic(df, rule))
        elif rule.condition == "pattern_analysis":
            issues.extend(self._check_pattern_analysis(df, rule))
        
        return issues
    
    def _check_not_null(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check for null values in required columns."""
        issues = []
        
        for column in rule.columns:
            if column in df.columns:
                null_mask = df[column].isna()
                null_count = null_mask.sum()
                
                if null_count > 0:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {null_count} null values in required column '{column}'",
                        affected_rows=null_count,
                        affected_columns=[column],
                        sample_values=[],
                        recommendation=f"Remove rows with null {column} or provide default values"
                    ))
        
        return issues
    
    def _check_numeric_range(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check numeric values are within expected ranges."""
        issues = []
        
        # Define expected ranges for different columns
        ranges = {
            'Runtime': (1, 1000),
            'Budget': (0, 1e10),
            'Revenue': (0, 1e11),
            'Rating': (1.0, 10.0),
            'Votes': (0, 1e7)
        }
        
        for column in rule.columns:
            if column in df.columns and column in ranges:
                min_val, max_val = ranges[column]
                numeric_col = pd.to_numeric(df[column], errors='coerce')
                out_of_range = (numeric_col < min_val) | (numeric_col > max_val)
                out_of_range_count = out_of_range.sum()
                
                if out_of_range_count > 0:
                    sample_values = df[out_of_range][column].dropna().head(5).tolist()
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {out_of_range_count} values outside expected range ({min_val}-{max_val}) in '{column}'",
                        affected_rows=out_of_range_count,
                        affected_columns=[column],
                        sample_values=sample_values,
                        recommendation=f"Review and validate {column} values outside normal range"
                    ))
        
        return issues
    
    def _check_date_range(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check date values are within reasonable ranges."""
        issues = []
        
        for column in rule.columns:
            if column in df.columns:
                try:
                    date_col = pd.to_datetime(df[column], errors='coerce')
                    
                    # Define reasonable date range
                    min_date = pd.to_datetime('1888-01-01')  # First motion picture
                    max_date = pd.to_datetime('2030-12-31')   # Near future
                    
                    invalid_dates = (date_col < min_date) | (date_col > max_date)
                    invalid_count = invalid_dates.sum()
                    
                    if invalid_count > 0:
                        sample_values = df[invalid_dates][column].dropna().head(5).tolist()
                        issues.append(QualityIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"Found {invalid_count} dates outside reasonable range in '{column}'",
                            affected_rows=invalid_count,
                            affected_columns=[column],
                            sample_values=sample_values,
                            recommendation=f"Review and correct unrealistic dates in {column}"
                        ))
                        
                except Exception as e:
                    self.logger.warning(f"Could not validate dates in {column}: {e}")
        
        return issues
    
    def _check_unique(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check for duplicate values in columns that should be unique."""
        issues = []
        
        for column in rule.columns:
            if column in df.columns:
                duplicates = df[df[column].duplicated(keep=False)]
                duplicate_count = len(duplicates)
                
                if duplicate_count > 0:
                    sample_values = duplicates[column].unique()[:5].tolist()
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {duplicate_count} duplicate values in '{column}'",
                        affected_rows=duplicate_count,
                        affected_columns=[column],
                        sample_values=sample_values,
                        recommendation=f"Remove or consolidate duplicate {column} values"
                    ))
        
        return issues
    
    def _check_unique_combination(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check for duplicate combinations of columns."""
        issues = []
        
        # Check if all columns exist
        existing_columns = [col for col in rule.columns if col in df.columns]
        if len(existing_columns) == len(rule.columns):
            duplicates = df[df.duplicated(subset=existing_columns, keep=False)]
            duplicate_count = len(duplicates)
            
            if duplicate_count > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"Found {duplicate_count} duplicate combinations of {existing_columns}",
                    affected_rows=duplicate_count,
                    affected_columns=existing_columns,
                    sample_values=[],
                    recommendation=f"Remove or consolidate duplicate combinations of {existing_columns}"
                ))
        
        return issues
    
    def _check_categorical_values(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check categorical values are within expected sets."""
        issues = []
        
        # Define valid values for categorical columns
        valid_values = {
            'Gender': {1, 2, 3},
            'Certificate': {'G', 'PG', 'PG-13', 'R', 'NC-17', 'Unrated', ''},
        }
        
        for column in rule.columns:
            if column in df.columns and column in valid_values:
                valid_set = valid_values[column]
                invalid_mask = ~df[column].isin(valid_set) & df[column].notna()
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    sample_values = df[invalid_mask][column].unique()[:5].tolist()
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {invalid_count} invalid values in categorical column '{column}'",
                        affected_rows=invalid_count,
                        affected_columns=[column],
                        sample_values=sample_values,
                        recommendation=f"Map invalid {column} values to valid categories: {valid_set}"
                    ))
        
        return issues
    
    def _check_text_format(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check text fields have reasonable format and length."""
        issues = []
        
        for column in rule.columns:
            if column in df.columns:
                text_col = df[column].astype(str)
                
                # Check for very short or very long text
                too_short = (text_col.str.len() < 1) & df[column].notna()
                too_long = text_col.str.len() > 1000
                
                # Check for suspicious patterns
                all_caps = text_col.str.isupper() & (text_col.str.len() > 10)
                all_numbers = text_col.str.isdigit() & (text_col.str.len() > 3)
                
                total_issues = too_short.sum() + too_long.sum() + all_caps.sum() + all_numbers.sum()
                
                if total_issues > 0:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {total_issues} text format issues in '{column}'",
                        affected_rows=total_issues,
                        affected_columns=[column],
                        sample_values=[],
                        recommendation=f"Review and standardize text format in {column}"
                    ))
        
        return issues
    
    def _check_text_similarity(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check similarity between related text fields."""
        issues = []
        
        if len(rule.columns) >= 2:
            col1, col2 = rule.columns[0], rule.columns[1]
            if col1 in df.columns and col2 in df.columns:
                # Simple similarity check based on length difference
                valid_rows = df[col1].notna() & df[col2].notna()
                if valid_rows.any():
                    len_diff = abs(df[valid_rows][col1].str.len() - df[valid_rows][col2].str.len())
                    very_different = len_diff > 50  # Threshold for "very different"
                    
                    different_count = very_different.sum()
                    if different_count > 0:
                        issues.append(QualityIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"Found {different_count} cases where {col1} and {col2} are very different",
                            affected_rows=different_count,
                            affected_columns=[col1, col2],
                            sample_values=[],
                            recommendation=f"Review cases where {col1} and {col2} differ significantly"
                        ))
        
        return issues
    
    def _check_business_logic(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check business logic rules."""
        issues = []
        
        if rule.name == "budget_revenue_consistency":
            if 'Budget' in df.columns and 'Revenue' in df.columns:
                budget = pd.to_numeric(df['Budget'], errors='coerce')
                revenue = pd.to_numeric(df['Revenue'], errors='coerce')
                
                # Flag cases where revenue < 10% of budget (for movies with significant budget)
                significant_budget = budget > 1000000  # $1M+
                low_revenue = (revenue < budget * 0.1) & significant_budget
                
                issue_count = low_revenue.sum()
                if issue_count > 0:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {issue_count} movies with revenue < 10% of budget",
                        affected_rows=issue_count,
                        affected_columns=['Budget', 'Revenue'],
                        sample_values=[],
                        recommendation="Review movies with unusually low revenue compared to budget"
                    ))
        
        return issues
    
    def _check_temporal_logic(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check temporal consistency between related dates."""
        issues = []
        # Implementation would require cross-referencing with movie data
        # For now, just check if rating dates are reasonable
        if 'Date' in df.columns:
            try:
                date_col = pd.to_datetime(df['Date'], errors='coerce')
                future_dates = date_col > pd.Timestamp.now()
                future_count = future_dates.sum()
                
                if future_count > 0:
                    issues.append(QualityIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {future_count} ratings with future dates",
                        affected_rows=future_count,
                        affected_columns=['Date'],
                        sample_values=[],
                        recommendation="Review and correct future rating dates"
                    ))
            except Exception as e:
                self.logger.warning(f"Could not check temporal logic: {e}")
        
        return issues
    
    def _check_pattern_analysis(self, df: pd.DataFrame, rule: QualityRule) -> List[QualityIssue]:
        """Check for suspicious patterns in user behavior."""
        issues = []
        
        if 'UserID' in df.columns and 'Rating' in df.columns:
            # Find users who always give the same rating
            user_rating_std = df.groupby('UserID')['Rating'].agg(['std', 'count'])
            users_same_rating = user_rating_std[(user_rating_std['std'] == 0) & (user_rating_std['count'] > 5)]
            
            if len(users_same_rating) > 0:
                issues.append(QualityIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"Found {len(users_same_rating)} users with identical ratings (5+ movies)",
                    affected_rows=len(users_same_rating),
                    affected_columns=['UserID', 'Rating'],
                    sample_values=users_same_rating.index.tolist()[:5],
                    recommendation="Review users with suspicious rating patterns"
                ))
        
        return issues
    
    def apply_fixes(self, df: pd.DataFrame, dataset_type: str, 
                   fix_level: ValidationSeverity = ValidationSeverity.HIGH) -> pd.DataFrame:
        """
        Apply automatic fixes for quality issues up to specified severity level.
        
        Args:
            df: DataFrame to fix
            dataset_type: Type of dataset
            fix_level: Maximum severity level to auto-fix
            
        Returns:
            Fixed DataFrame
        """
        df_fixed = df.copy()
        issues = self.validation_results.get(dataset_type, {}).get('issues', [])
        
        severity_order = [ValidationSeverity.LOW, ValidationSeverity.MEDIUM, 
                         ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]
        
        for issue in issues:
            if severity_order.index(issue.severity) >= severity_order.index(fix_level):
                continue  # Skip issues above fix level
            
            # Apply fixes based on rule name
            if issue.rule_name == "required_movie_fields" or issue.rule_name == "required_rating_fields":
                # Remove rows with missing required fields
                for column in issue.affected_columns:
                    if column in df_fixed.columns:
                        df_fixed = df_fixed.dropna(subset=[column])
            
            elif issue.rule_name == "valid_rating_range":
                # Remove invalid ratings
                if 'Rating' in df_fixed.columns:
                    df_fixed = df_fixed[
                        (df_fixed['Rating'] >= 1.0) & 
                        (df_fixed['Rating'] <= 10.0) & 
                        (df_fixed['Rating'].notna())
                    ]
            
            elif issue.rule_name == "duplicate_user_movie_ratings":
                # Keep the most recent rating for duplicate user-movie combinations
                if all(col in df_fixed.columns for col in ['UserID', 'MovieID', 'Date']):
                    df_fixed = df_fixed.sort_values('Date', ascending=False)
                    df_fixed = df_fixed.drop_duplicates(subset=['UserID', 'MovieID'], keep='first')
        
        fixed_rows = len(df) - len(df_fixed)
        if fixed_rows > 0:
            self.logger.info(f"Applied automatic fixes, removed {fixed_rows} problematic rows")
        
        return df_fixed
    
    def generate_quality_report(self, dataset_type: str) -> Dict[str, Any]:
        """Generate comprehensive quality report for a dataset."""
        if dataset_type not in self.validation_results:
            return {'error': 'No validation results available'}
        
        results = self.validation_results[dataset_type]
        issues = results.get('issues', [])
        
        # Aggregate issues by severity
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = len([i for i in issues if i.severity == severity])
        
        # Calculate quality score (0-100)
        total_rows = results.get('total_rows', 0)
        if total_rows > 0:
            critical_weight = 10
            high_weight = 5
            medium_weight = 2
            low_weight = 1
            
            weighted_issues = (
                severity_counts.get('critical', 0) * critical_weight +
                severity_counts.get('high', 0) * high_weight +
                severity_counts.get('medium', 0) * medium_weight +
                severity_counts.get('low', 0) * low_weight
            )
            
            quality_score = max(0, 100 - (weighted_issues / total_rows * 100))
        else:
            quality_score = 0
        
        return {
            'dataset_type': dataset_type,
            'timestamp': results.get('timestamp'),
            'total_rows': total_rows,
            'quality_score': round(quality_score, 2),
            'severity_counts': severity_counts,
            'total_issues': results.get('total_issues', 0),
            'issues_by_rule': {issue.rule_name: issue.message for issue in issues},
            'recommendations': list(set([issue.recommendation for issue in issues]))
        }