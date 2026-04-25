"""
GNN Utilities Package

Based on voltage_variation_pretraining_utils.py structure.
Provides data preprocessing utilities for GNN pretraining.
"""

from .gnn_data_preprocessing_utils import (
    # Node feature normalization
    normalize_node_features,
    normalize_node_features_safe,

    # Output filtering and normalization (combined)
    filter_and_normalize_task_outputs,

    # Output normalization only (for already filtered data)
    normalize_task_outputs,

    # Statistics calculation
    calculate_norm_stats_from_minimal_data_safe,

    # Data validation
    validate_gnn_data,

    # Complete preprocessing pipeline
    preprocess_gnn_minimal_data
)

__all__ = [
    # Node feature normalization
    'normalize_node_features',
    'normalize_node_features_safe',

    # Output filtering and normalization
    'filter_and_normalize_task_outputs',
    'normalize_task_outputs',

    # Statistics calculation
    'calculate_norm_stats_from_minimal_data_safe',

    # Data validation
    'validate_gnn_data',

    # Complete pipeline
    'preprocess_gnn_minimal_data'
]
