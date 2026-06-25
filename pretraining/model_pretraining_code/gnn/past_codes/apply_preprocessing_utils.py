#!/usr/bin/env python
"""
Script to apply preprocessing utils to all GNN training files.
This adds the safe data preprocessing pipeline to avoid NaN issues during training.
"""

import os
import re

# Files to update
FILES_TO_UPDATE = [
    'baseline_gnn_training_cached.py',
    'baseline_gnn_training_cached_global_norm.py',
    'maml_gnn_training_cached_global_norm.py',
]

UTILS_IMPORT = """sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from gnn_maml import (
    MAML_GNN_Model,
    create_maml_gcn_model
)
from gnn_data_preprocessing_utils import (
    preprocess_gnn_minimal_data,
    normalize_node_features_safe,
    normalize_task_outputs
)"""

OLD_IMPORT = """from gnn_maml import (
    MAML_GNN_Model,
    create_maml_gcn_model
)"""

INSTRUCTIONS = """
📋 Instructions to apply preprocessing utils to GNN training files:

1. Add utils import to the imports section (after sys.path.insert lines):

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from gnn_data_preprocessing_utils import (
    preprocess_gnn_minimal_data,
    normalize_node_features_safe,
    normalize_task_outputs
)
```

2. Replace the normalize_node_features method with:

```python
def normalize_node_features(self, node_features):
    \"\"\"Normalize node features using saved statistics (with NaN/Inf protection)\"\"\"
    if self.norm_stats is None:
        return node_features

    # Use safe normalization from utils
    normalized, _ = normalize_node_features_safe(
        node_features,
        norm_stats=self.norm_stats['node_features']
    )

    return normalized
```

3. Replace the normalize_all_task_outputs method with:

```python
def normalize_all_task_outputs(self, stacked_outputs):
    \"\"\"Pre-normalize all task outputs before training (with NaN/Inf protection)\"\"\"
    if stacked_outputs is None:
        return None

    # Use safe normalization from utils
    normalized, task_norm_stats = normalize_task_outputs(
        stacked_outputs,
        min_std_threshold=1e-8
    )

    # Update task_norm_stats
    self.task_norm_stats = task_norm_stats

    return normalized
```

4. In the data loading function (load_cached_gnn_data_*), replace the normalization section with:

```python
# Apply preprocessing pipeline (filtering + normalization with NaN/Inf detection)
print(f"\\n🔧 Applying data preprocessing pipeline...")
preprocessed_data, norm_stats, preprocessing_stats = preprocess_gnn_minimal_data(
    all_minimal_data_per_file,
    min_std_threshold=1e-6,
    enable_filtering=True,
    verbose=True
)

print(f"\\n📊 Preprocessing Summary:")
print(f"   Valid tasks after filtering: {preprocessing_stats['filtering']['valid_tasks']}")
print(f"   Filter ratio: {preprocessing_stats['filtering']['filter_ratio']:.1f}%")

return preprocessed_data, topology_cache, norm_stats
```

5. Remove the old calculate_norm_stats_from_minimal_data function if it exists.

═══════════════════════════════════════════════════════════════════

Files to update:
"""

def main():
    print(INSTRUCTIONS)

    base_path = os.path.dirname(os.path.abspath(__file__))

    for filename in FILES_TO_UPDATE:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            print(f"  ⚠️  {filename} (not found)")

    print("\n" + "="*80)
    print("✅ maml_gnn_training_cached.py has already been updated!")
    print("\nTo update the remaining files, apply the instructions above to each file.")
    print("\nKey changes:")
    print("  1. Import utils module")
    print("  2. Use normalize_node_features_safe for node feature normalization")
    print("  3. Use normalize_task_outputs for output normalization")
    print("  4. Use preprocess_gnn_minimal_data for data filtering and validation")
    print("\nThese changes add NaN/Inf detection and filtering to prevent training loss = NaN")
    print("="*80)

if __name__ == "__main__":
    main()
