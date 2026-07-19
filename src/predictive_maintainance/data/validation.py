"""
Data Validation Module

Validates data quality, constraints, and business rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from ..utils.logger import get_logger
from ..utils.exceptions import DataValidationError

logger = get_logger(__name__)

class DataValidator:
    """Validator for data quality and business rules."""
    
    def __init__(self, expectations_path: Optional[str] = None):
        """
        Initialize validator with expectations.
        
        Args:
            expectations_path: Path to expectations JSON file
        """
        self.expectations = {}
        if expectations_path:
            self.load_expectations(expectations_path)
            
    def load_expectations(self, path: str):
        """Load expectations from JSON file."""
        try:
            with open(path, 'r') as f:
                self.expectations = json.load(f)
            logger.info(f"Loaded expectations from {path}")
        except Exception as e:
            logger.error(f"Failed to load expectations: {str(e)}")
            raise DataValidationError(f"Failed to load expectations: {str(e)}")
            
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        expectations: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate DataFrame against expectations.
        
        Args:
            df: DataFrame to validate
            expectations: Optional expectations to use instead of loaded
            
        Returns:
            Validation results dictionary
        """
        if expectations is None:
            expectations = self.expectations
            
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Validate structure
        structure_results = self._validate_structure(df, expectations.get('structure', {}))
        results['valid'] &= structure_results['valid']
        results['errors'].extend(structure_results['errors'])
        results['warnings'].extend(structure_results['warnings'])
        
        # Validate constraints
        constraint_results = self._validate_constraints(df, expectations.get('constraints', {}))
        results['valid'] &= constraint_results['valid']
        results['errors'].extend(constraint_results['errors'])
        results['warnings'].extend(constraint_results['warnings'])
        
        # Validate relationships
        relationship_results = self._validate_relationships(df, expectations.get('relationships', {}))
        results['valid'] &= relationship_results['valid']
        results['errors'].extend(relationship_results['errors'])
        results['warnings'].extend(relationship_results['warnings'])
        
        # Calculate metrics
        results['metrics'] = self._calculate_metrics(df)
        
        return results
        
    def _validate_structure(
        self,
        df: pd.DataFrame,
        structure_expectations: Dict
    ) -> Dict[str, Any]:
        """Validate DataFrame structure."""
        results = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check required columns
        required_cols = structure_expectations.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
            
        # Check column types
        column_types = structure_expectations.get('column_types', {})
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    results['warnings'].append(
                        f"Column {col} expected type {expected_type}, got {actual_type}"
                    )
                    
        # Check row count
        min_rows = structure_expectations.get('min_rows')
        max_rows = structure_expectations.get('max_rows')
        if min_rows is not None and len(df) < min_rows:
            results['valid'] = False
            results['errors'].append(f"Row count {len(df)} below minimum {min_rows}")
        if max_rows is not None and len(df) > max_rows:
            results['warnings'].append(f"Row count {len(df)} exceeds maximum {max_rows}")
            
        return results
        
    def _validate_constraints(
        self,
        df: pd.DataFrame,
        constraints: Dict
    ) -> Dict[str, Any]:
        """Validate column constraints."""
        results = {'valid': True, 'errors': [], 'warnings': []}
        
        for col, constraint in constraints.items():
            if col not in df.columns:
                continue
                
            # Check null constraints
            if 'not_null' in constraint and constraint['not_null']:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    results['valid'] = False
                    results['errors'].append(f"Column {col} contains {null_count} null values")
                    
            # Check unique constraint
            if 'unique' in constraint and constraint['unique']:
                duplicate_count = df[col].duplicated().sum()
                if duplicate_count > 0:
                    results['warnings'].append(
                        f"Column {col} contains {duplicate_count} duplicate values"
                    )
                    
            # Check value range
            if col in df.select_dtypes(include=[np.number]).columns:
                min_val = constraint.get('min')
                max_val = constraint.get('max')
                
                if min_val is not None:
                    below_min = (df[col] < min_val).sum()
                    if below_min > 0:
                        results['warnings'].append(
                            f"Column {col} has {below_min} values below {min_val}"
                        )
                        
                if max_val is not None:
                    above_max = (df[col] > max_val).sum()
                    if above_max > 0:
                        results['warnings'].append(
                            f"Column {col} has {above_max} values above {max_val}"
                        )
                        
            # Check allowed values
            if 'allowed_values' in constraint:
                allowed = set(constraint['allowed_values'])
                invalid = ~df[col].isin(allowed)
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    results['warnings'].append(
                        f"Column {col} has {invalid_count} invalid values"
                    )
                    
        return results
        
    def _validate_relationships(
        self,
        df: pd.DataFrame,
        relationships: Dict
    ) -> Dict[str, Any]:
        """Validate relationships between columns."""
        results = {'valid': True, 'errors': [], 'warnings': []}
        
        for rel in relationships:
            col1 = rel.get('column1')
            col2 = rel.get('column2')
            rel_type = rel.get('type')
            
            if col1 not in df.columns or col2 not in df.columns:
                continue
                
            if rel_type == 'mutually_exclusive':
                invalid = (df[col1].notna() & df[col2].notna()).sum()
                if invalid > 0:
                    results['warnings'].append(
                        f"Mutually exclusive violation: {invalid} rows have both {col1} and {col2}"
                    )
                    
            elif rel_type == 'conditional':
                when_col = rel.get('when_column')
                when_value = rel.get('when_value')
                then_col = rel.get('then_column')
                then_value = rel.get('then_value')
                
                if when_col in df.columns and then_col in df.columns:
                    mask = df[when_col] == when_value
                    invalid = (mask & (df[then_col] != then_value)).sum()
                    if invalid > 0:
                        results['warnings'].append(
                            f"Conditional violation: {invalid} rows violate {when_col}={when_value} -> {then_col}={then_value}"
                        )
                        
        return results
        
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        metrics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'missing_values': {},
            'unique_values': {},
            'null_percentage': {},
            'dtypes': df.dtypes.to_dict()
        }
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            metrics['missing_values'][col] = null_count
            metrics['null_percentage'][col] = (null_count / len(df)) * 100
            metrics['unique_values'][col] = df[col].nunique()
            
        return metrics
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate validation report."""
        report = []
        report.append("=" * 80)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Valid: {'✓' if results['valid'] else '✗'}")
        report.append(f"Errors: {len(results['errors'])}")
        report.append(f"Warnings: {len(results['warnings'])}")
        report.append("")
        
        if results['errors']:
            report.append("ERRORS:")
            for error in results['errors']:
                report.append(f"  - {error}")
            report.append("")
            
        if results['warnings']:
            report.append("WARNINGS:")
            for warning in results['warnings']:
                report.append(f"  - {warning}")
            report.append("")
            
        report.append("METRICS:")
        metrics = results['metrics']
        report.append(f"  Row Count: {metrics['row_count']:,}")
        report.append(f"  Column Count: {metrics['column_count']}")
        report.append(f"  Duplicate Rows: {metrics['duplicate_rows']:,}")
        report.append("")
        
        report.append("MISSING VALUES:")
        for col, null_count in metrics['missing_values'].items():
            pct = metrics['null_percentage'][col]
            report.append(f"  {col}: {null_count:,} ({pct:.2f}%)")
        report.append("")
        
        report.append("UNIQUE VALUES:")
        for col, unique_count in metrics['unique_values'].items():
            report.append(f"  {col}: {unique_count:,}")
            
        report.append("=" * 80)
        return "\n".join(report)