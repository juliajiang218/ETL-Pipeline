"""
Advanced Deduplication Module
Handles duplicate detection and resolution with configurable strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
from difflib import SequenceMatcher
import re


class DeduplicationStrategy(Enum):
    """Strategies for handling duplicates."""
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    KEEP_BEST_QUALITY = "keep_best_quality"
    MERGE_RECORDS = "merge_records"
    MANUAL_REVIEW = "manual_review"


class MatchType(Enum):
    """Types of duplicate matches."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


@dataclass
class DuplicateGroup:
    """Group of duplicate records."""
    group_id: str
    records: List[int]  # Row indices
    match_type: MatchType
    confidence: float
    primary_columns: List[str]
    resolution_strategy: DeduplicationStrategy


class DeduplicationEngine:
    """Advanced duplicate detection and resolution engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.duplicate_groups: Dict[str, List[DuplicateGroup]] = {}
        self.deduplication_stats: Dict[str, Any] = {}
    
    def deduplicate_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate movie records with intelligent matching.
        
        Args:
            df: Movies DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        self.logger.info(f"Starting movie deduplication on {len(df)} records")
        
        # Define deduplication strategies for movies
        strategies = [
            {
                'columns': ['MovieID'],
                'strategy': DeduplicationStrategy.KEEP_FIRST,
                'match_type': MatchType.EXACT,
                'priority': 1
            },
            {
                'columns': ['OriginalTitle', 'ReleaseDate'],
                'strategy': DeduplicationStrategy.KEEP_BEST_QUALITY,
                'match_type': MatchType.EXACT,
                'priority': 2
            },
            {
                'columns': ['OriginalTitle', 'EnglishTitle'],
                'strategy': DeduplicationStrategy.MERGE_RECORDS,
                'match_type': MatchType.FUZZY,
                'threshold': 0.9,
                'priority': 3
            }
        ]
        
        df_deduplicated = self._apply_deduplication_strategies(df, strategies, 'movies')
        
        original_count = len(df)
        final_count = len(df_deduplicated)
        removed_count = original_count - final_count
        
        self.deduplication_stats['movies'] = {
            'original_records': original_count,
            'final_records': final_count,
            'duplicates_removed': removed_count,
            'deduplication_rate': removed_count / original_count if original_count > 0 else 0
        }
        
        self.logger.info(f"Movie deduplication completed: {original_count} -> {final_count} ({removed_count} duplicates removed)")
        return df_deduplicated
    
    def deduplicate_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate rating records with user-movie combination logic.
        
        Args:
            df: Ratings DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        self.logger.info(f"Starting rating deduplication on {len(df)} records")
        
        strategies = [
            {
                'columns': ['UserID', 'MovieID', 'Date'],
                'strategy': DeduplicationStrategy.KEEP_FIRST,
                'match_type': MatchType.EXACT,
                'priority': 1
            },
            {
                'columns': ['UserID', 'MovieID'],
                'strategy': DeduplicationStrategy.KEEP_LAST,  # Keep most recent rating
                'match_type': MatchType.EXACT,
                'priority': 2
            }
        ]
        
        df_deduplicated = self._apply_deduplication_strategies(df, strategies, 'ratings')
        
        original_count = len(df)
        final_count = len(df_deduplicated)
        removed_count = original_count - final_count
        
        self.deduplication_stats['ratings'] = {
            'original_records': original_count,
            'final_records': final_count,
            'duplicates_removed': removed_count,
            'deduplication_rate': removed_count / original_count if original_count > 0 else 0
        }
        
        self.logger.info(f"Rating deduplication completed: {original_count} -> {final_count} ({removed_count} duplicates removed)")
        return df_deduplicated
    
    def deduplicate_persons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate person records with name and role matching.
        
        Args:
            df: Persons DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        self.logger.info(f"Starting person deduplication on {len(df)} records")
        
        strategies = [
            {
                'columns': ['CastID', 'MovieID'],
                'strategy': DeduplicationStrategy.KEEP_FIRST,
                'match_type': MatchType.EXACT,
                'priority': 1
            },
            {
                'columns': ['Name', 'MovieID', 'Character'],
                'strategy': DeduplicationStrategy.MERGE_RECORDS,
                'match_type': MatchType.FUZZY,
                'threshold': 0.95,
                'priority': 2
            }
        ]
        
        df_deduplicated = self._apply_deduplication_strategies(df, strategies, 'persons')
        
        original_count = len(df)
        final_count = len(df_deduplicated)
        removed_count = original_count - final_count
        
        self.deduplication_stats['persons'] = {
            'original_records': original_count,
            'final_records': final_count,
            'duplicates_removed': removed_count,
            'deduplication_rate': removed_count / original_count if original_count > 0 else 0
        }
        
        self.logger.info(f"Person deduplication completed: {original_count} -> {final_count} ({removed_count} duplicates removed)")
        return df_deduplicated
    
    def _apply_deduplication_strategies(self, df: pd.DataFrame, strategies: List[Dict], 
                                      dataset_type: str) -> pd.DataFrame:
        """Apply multiple deduplication strategies in priority order."""
        df_result = df.copy()
        all_duplicate_groups = []
        
        for strategy_config in sorted(strategies, key=lambda x: x['priority']):
            # Find duplicates using this strategy
            duplicate_groups = self._find_duplicates(
                df_result, 
                strategy_config['columns'],
                strategy_config['match_type'],
                strategy_config.get('threshold', 1.0)
            )
            
            if duplicate_groups:
                # Apply resolution strategy
                df_result = self._resolve_duplicates(
                    df_result,
                    duplicate_groups,
                    strategy_config['strategy']
                )
                all_duplicate_groups.extend(duplicate_groups)
        
        # Store duplicate groups for analysis
        self.duplicate_groups[dataset_type] = all_duplicate_groups
        
        return df_result
    
    def _find_duplicates(self, df: pd.DataFrame, columns: List[str], 
                        match_type: MatchType, threshold: float = 1.0) -> List[DuplicateGroup]:
        """Find duplicate groups based on specified columns and match type."""
        duplicate_groups = []
        
        # Check if all columns exist
        existing_columns = [col for col in columns if col in df.columns]
        if len(existing_columns) != len(columns):
            self.logger.warning(f"Some columns missing for duplicate detection: {columns}")
            return duplicate_groups
        
        if match_type == MatchType.EXACT:
            duplicate_groups = self._find_exact_duplicates(df, existing_columns)
        elif match_type == MatchType.FUZZY:
            duplicate_groups = self._find_fuzzy_duplicates(df, existing_columns, threshold)
        elif match_type == MatchType.SEMANTIC:
            duplicate_groups = self._find_semantic_duplicates(df, existing_columns, threshold)
        
        return duplicate_groups
    
    def _find_exact_duplicates(self, df: pd.DataFrame, columns: List[str]) -> List[DuplicateGroup]:
        """Find exact duplicate matches."""
        duplicate_groups = []
        
        # Group by the specified columns
        grouped = df.groupby(columns)
        
        group_id = 0
        for name, group in grouped:
            if len(group) > 1:  # More than one record in group
                duplicate_groups.append(DuplicateGroup(
                    group_id=f"exact_{group_id}",
                    records=group.index.tolist(),
                    match_type=MatchType.EXACT,
                    confidence=1.0,
                    primary_columns=columns,
                    resolution_strategy=DeduplicationStrategy.KEEP_FIRST
                ))
                group_id += 1
        
        return duplicate_groups
    
    def _find_fuzzy_duplicates(self, df: pd.DataFrame, columns: List[str], 
                              threshold: float) -> List[DuplicateGroup]:
        """Find fuzzy duplicate matches based on string similarity."""
        duplicate_groups = []
        
        if not columns:
            return duplicate_groups
        
        # Focus on text columns for fuzzy matching
        text_columns = [col for col in columns if df[col].dtype == 'object']
        if not text_columns:
            return duplicate_groups
        
        # Create combined text for comparison
        df_text = df[text_columns].fillna('').astype(str)
        combined_text = df_text.agg(' '.join, axis=1)
        
        # Find fuzzy matches
        processed_indices = set()
        group_id = 0
        
        for i, text1 in combined_text.items():
            if i in processed_indices:
                continue
                
            matches = [i]
            processed_indices.add(i)
            
            for j, text2 in combined_text.items():
                if j <= i or j in processed_indices:
                    continue
                
                similarity = self._calculate_text_similarity(text1, text2)
                if similarity >= threshold:
                    matches.append(j)
                    processed_indices.add(j)
            
            if len(matches) > 1:
                duplicate_groups.append(DuplicateGroup(
                    group_id=f"fuzzy_{group_id}",
                    records=matches,
                    match_type=MatchType.FUZZY,
                    confidence=threshold,
                    primary_columns=text_columns,
                    resolution_strategy=DeduplicationStrategy.KEEP_BEST_QUALITY
                ))
                group_id += 1
        
        return duplicate_groups
    
    def _find_semantic_duplicates(self, df: pd.DataFrame, columns: List[str], 
                                 threshold: float) -> List[DuplicateGroup]:
        """Find semantic duplicates (placeholder for more advanced NLP matching)."""
        # For now, fall back to fuzzy matching
        # In a production system, this would use word embeddings or other semantic similarity
        return self._find_fuzzy_duplicates(df, columns, threshold)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize text
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower().strip())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower().strip())
        
        # Use sequence matcher for similarity
        return SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    def _resolve_duplicates(self, df: pd.DataFrame, duplicate_groups: List[DuplicateGroup],
                           strategy: DeduplicationStrategy) -> pd.DataFrame:
        """Resolve duplicates using the specified strategy."""
        indices_to_remove = set()
        
        for group in duplicate_groups:
            if strategy == DeduplicationStrategy.KEEP_FIRST:
                # Keep the first record, remove the rest
                indices_to_remove.update(group.records[1:])
                
            elif strategy == DeduplicationStrategy.KEEP_LAST:
                # Keep the last record, remove the rest
                indices_to_remove.update(group.records[:-1])
                
            elif strategy == DeduplicationStrategy.KEEP_BEST_QUALITY:
                # Keep the record with the best quality score
                best_index = self._find_best_quality_record(df, group.records)
                indices_to_remove.update([idx for idx in group.records if idx != best_index])
                
            elif strategy == DeduplicationStrategy.MERGE_RECORDS:
                # Merge records and keep one (placeholder implementation)
                merged_record = self._merge_records(df, group.records)
                if merged_record is not None:
                    # Update the first record with merged data
                    first_idx = group.records[0]
                    for col, value in merged_record.items():
                        if col in df.columns:
                            df.loc[first_idx, col] = value
                    # Remove other records
                    indices_to_remove.update(group.records[1:])
                else:
                    # Fall back to keeping first
                    indices_to_remove.update(group.records[1:])
        
        # Remove duplicate records
        if indices_to_remove:
            df_result = df.drop(indices_to_remove)
            self.logger.info(f"Removed {len(indices_to_remove)} duplicate records")
        else:
            df_result = df
        
        return df_result
    
    def _find_best_quality_record(self, df: pd.DataFrame, record_indices: List[int]) -> int:
        """Find the record with the best quality among duplicates."""
        subset = df.loc[record_indices]
        
        # Quality scoring based on completeness and data richness
        quality_scores = {}
        
        for idx in record_indices:
            record = subset.loc[idx]
            
            # Count non-null values
            completeness_score = record.notna().sum()
            
            # Prefer records with more text content
            text_length_score = 0
            for col in subset.columns:
                if subset[col].dtype == 'object' and pd.notna(record[col]):
                    text_length_score += len(str(record[col]))
            
            # Prefer records with more recent dates
            recency_score = 0
            for col in subset.columns:
                if 'date' in col.lower() and pd.notna(record[col]):
                    try:
                        date_val = pd.to_datetime(record[col])
                        days_from_now = (datetime.now() - date_val).days
                        recency_score = max(0, 1000 - days_from_now)  # More recent = higher score
                    except:
                        pass
            
            # Combine scores
            quality_scores[idx] = completeness_score + (text_length_score / 100) + (recency_score / 100)
        
        # Return index with highest quality score
        return max(quality_scores.items(), key=lambda x: x[1])[0]
    
    def _merge_records(self, df: pd.DataFrame, record_indices: List[int]) -> Optional[Dict[str, Any]]:
        """Merge multiple records into one comprehensive record."""
        subset = df.loc[record_indices]
        merged_record = {}
        
        for col in subset.columns:
            col_values = subset[col].dropna()
            
            if len(col_values) == 0:
                merged_record[col] = np.nan
            elif len(col_values) == 1:
                merged_record[col] = col_values.iloc[0]
            else:
                # Multiple non-null values - use merge strategy based on column type
                if col_values.dtype == 'object':
                    # For text, prefer the longest value
                    merged_record[col] = max(col_values, key=len)
                elif np.issubdtype(col_values.dtype, np.number):
                    # For numbers, prefer the maximum (assuming higher is better)
                    merged_record[col] = col_values.max()
                elif pd.api.types.is_datetime64_any_dtype(col_values):
                    # For dates, prefer the most recent
                    merged_record[col] = col_values.max()
                else:
                    # Default to first value
                    merged_record[col] = col_values.iloc[0]
        
        return merged_record
    
    def detect_potential_duplicates(self, df: pd.DataFrame, columns: List[str],
                                  match_type: MatchType = MatchType.FUZZY,
                                  threshold: float = 0.8) -> pd.DataFrame:
        """
        Detect potential duplicates for manual review.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to use for matching
            match_type: Type of matching to perform
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            DataFrame with potential duplicate pairs
        """
        duplicate_groups = self._find_duplicates(df, columns, match_type, threshold)
        
        duplicate_pairs = []
        for group in duplicate_groups:
            records = df.loc[group.records]
            for i, idx1 in enumerate(group.records):
                for idx2 in group.records[i+1:]:
                    duplicate_pairs.append({
                        'record1_index': idx1,
                        'record2_index': idx2,
                        'match_type': group.match_type.value,
                        'confidence': group.confidence,
                        'columns': ', '.join(group.primary_columns),
                        'record1_preview': str(df.loc[idx1, columns[0]])[:50] if columns else '',
                        'record2_preview': str(df.loc[idx2, columns[0]])[:50] if columns else ''
                    })
        
        return pd.DataFrame(duplicate_pairs)
    
    def generate_deduplication_report(self) -> Dict[str, Any]:
        """Generate comprehensive deduplication report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets_processed': list(self.deduplication_stats.keys()),
            'summary': {},
            'details': self.deduplication_stats,
            'duplicate_groups': {}
        }
        
        # Calculate overall statistics
        total_original = sum(stats.get('original_records', 0) for stats in self.deduplication_stats.values())
        total_final = sum(stats.get('final_records', 0) for stats in self.deduplication_stats.values())
        total_removed = sum(stats.get('duplicates_removed', 0) for stats in self.deduplication_stats.values())
        
        report['summary'] = {
            'total_original_records': total_original,
            'total_final_records': total_final,
            'total_duplicates_removed': total_removed,
            'overall_deduplication_rate': total_removed / total_original if total_original > 0 else 0
        }
        
        # Add duplicate group information
        for dataset_type, groups in self.duplicate_groups.items():
            report['duplicate_groups'][dataset_type] = {
                'total_groups': len(groups),
                'exact_matches': len([g for g in groups if g.match_type == MatchType.EXACT]),
                'fuzzy_matches': len([g for g in groups if g.match_type == MatchType.FUZZY]),
                'semantic_matches': len([g for g in groups if g.match_type == MatchType.SEMANTIC])
            }
        
        return report
    
    def get_duplicate_statistics(self, dataset_type: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific dataset's deduplication."""
        if dataset_type not in self.deduplication_stats:
            return {'error': f'No deduplication stats available for {dataset_type}'}
        
        stats = self.deduplication_stats[dataset_type].copy()
        
        if dataset_type in self.duplicate_groups:
            groups = self.duplicate_groups[dataset_type]
            stats['duplicate_group_analysis'] = {
                'total_groups_found': len(groups),
                'average_group_size': np.mean([len(g.records) for g in groups]) if groups else 0,
                'largest_group_size': max([len(g.records) for g in groups]) if groups else 0,
                'match_type_distribution': {
                    match_type.value: len([g for g in groups if g.match_type == match_type])
                    for match_type in MatchType
                }
            }
        
        return stats