#!/usr/bin/env python
# coding: utf-8

"""
Test Dataset Configuration Module
Manages different test dataset configurations for MAML/MLP testing
"""

# Test dataset configuration mapping
TEST_CONFIGS = {
    0: {
        'name': 'ASAP7 Intra Topology',
        'tech': 'asap7',
        'topology_type': 'intra',

        # Default settings
        'default_cells': ['NAND3x2', 'OR2x6', 'NOR2xp67', 'AND2x6'],
        'default_gpu': '7',
        'default_meta': 64,
        'default_data_type': 'transition',
        'default_num_iterations': 300000,

        # Training data paths (for normalization)
        'train_data_paths': lambda data_type: [
            (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded/{data_type}_intratopology_train_input.pth",
             f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded/{data_type}_intratopology_train_output.pth"),
            (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_input_{data_type}.pth",
             f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_output_{data_type}.pth")
        ],

        # Test data directory pattern
        'test_data_dir': "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded",
        'test_input_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/{data_type}_{cell}_test_input.pth",
        'test_output_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/{data_type}_{cell}_test_output.pth",

        # Model paths
        'maml_model_path': lambda data_type, innerdiv, meta, inner, num_iterations, layer_length=40: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/checkpoints/taskdivide_all_checkpoints/{data_type}_innerdiv{innerdiv}_meta{meta}_intra_topology_519traintask_full1DMAML_weights_3hidden_({layer_length})_{num_iterations}_inner{inner}_upgraded.pth",
        'mlp_model_path': lambda data_type, model_type, num_iterations: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/MLP_pretrained_model/training_loss_pretrained_asap7_intra_topology_{data_type}_{model_type}_{num_iterations}.pth",
    },

    1: {
        'name': 'ASAP7 Topology Agnostic',
        'tech': 'asap7',
        'topology_type': 'agnostic',

        # Default settings
        'default_cells': ["A2O1A1O1Ixp25","AO21x1","AO32x1","O2A1O1Ixp5","OAI22x1","FAx1"],
        #'default_cells': ["MAJIxp5", "MAJx2", "MAJx3", "HAxp5", "FAx1", "XOR2xp5", "XOR2x2", "XOR2x1" ,"XNOR2xp5", "XNOR2x2", "XNOR2x1","A2O1A1O1Ixp25","AO21x1","AO32x1","O2A1O1Ixp5","OAI22x1"],
        'default_gpu': '0',
        'default_meta': 32,
        'default_data_type': 'transition',
        'default_num_iterations': 300000,

        # Training data paths
        'train_data_paths': lambda data_type: [
            (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_input_{data_type}.pth",
             f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_output_{data_type}.pth"),
            (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/topology_agnostic_data_upgraded/{data_type}_topology_agnostic_train_input.pth",
             f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/topology_agnostic_data_upgraded/{data_type}_topology_agnostic_train_output.pth")
        ],

        # Test data directory pattern
        'test_data_dir': "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/test_topology_agnostic",
        'test_input_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/{data_type}_{cell}_test_input.pth",
        'test_output_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/{data_type}_{cell}_test_output.pth",

        # Model paths
        'maml_model_path': lambda data_type, innerdiv, meta, inner, num_iterations, layer_length=40: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/checkpoints/taskdivide_all_checkpoints/{data_type}_innerdiv{innerdiv}_meta{meta}_topology_agnostic_519traintask_full1DMAML_weights_3hidden_({layer_length})_{num_iterations}_inner{inner}_upgraded.pth",
        'mlp_model_path': lambda data_type, model_type, num_iterations: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/MLP_pretrained_model/training_loss_pretrained_asap7_topology_agnostic_{data_type}_{model_type}_{num_iterations}.pth",
    },

    2: {
        'name': 'TSMC Intra Topology',
        'tech': 'tsmc',
        'topology_type': 'intra',

        # Default settings
        'default_cells': ['AN4D0BWP30P140', 'NR3D1BWP30P140', 'OR4D0BWP30P140', 'ND3D0BWP30P140',
                           'XOR3D1BWP30P140', 'XNR3D1BWP30P140'],
        'default_gpu': '3',
        'default_meta': 32,
        'default_data_type': 'transition',
        'default_num_iterations': 300000,

        # Training data paths
        'train_data_paths': lambda data_type: [
            (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC/intra_topology_data/tsmc_intra_topology_train_input_{data_type}.pth",
             f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC/intra_topology_data/tsmc_intra_topology_train_output_{data_type}.pth")
        ],

        # Test data directory pattern
        'test_data_dir': "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC/intra_topology_data/",
        'test_input_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/tsmc_merged_test_input_{data_type}.pth",
        'test_output_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/tsmc_merged_test_output_{data_type}.pth",

        # Model paths
        'maml_model_path': lambda data_type, innerdiv, meta, inner, num_iterations, layer_length=40: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/taskdivide_all/{data_type}_innerdiv{innerdiv}_meta{meta}_intra_topology_519traintask_full1DMAML_weights_3hidden_({layer_length})_{num_iterations}_inner{inner}_upgraded_tsmc.pth",
        'mlp_model_path': lambda data_type, model_type, num_iterations: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/MLP_pretrained_model/training_loss_pretrained_tsmc_intra_topology_{data_type}_{model_type}_{num_iterations}.pth",
    },

    3: {
        'name': 'TSMC Topology Agnostic',
        'tech': 'tsmc',
        'topology_type': 'agnostic',

        # Default settings
        'default_cells': ['HA1D0BWP30P140', 'FA1D0BWP30P140', 'IOA21D0BWP30P140', 'IOA21D1BWP30P140',
                          'OA21D0BWP30P140', 'OA21D1BWP30P140', 'OA211D0BWP30P140', 'OA211D1BWP30P140',
                          'IAO21D0BWP30P140', 'IAO21D1BWP30P140', 'AO21D0BWP30P140', 'AO21D1BWP30P140',
                          'AO211D0BWP30P140', 'AO211D1BWP30P140', 'SDFSNQD0BWP30P140', 'DFCNQD1BWP30P140'],
        'default_gpu': '1',
        'default_meta': 32,
        'default_data_type': 'transition',
        'default_num_iterations': 300000,

        # Training data paths
        'train_data_paths': lambda data_type: [
            (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC/topology_agnostic_data/tsmc_topology_agnostic_train_input_{data_type}.pth",
             f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC/topology_agnostic_data/tsmc_topology_agnostic_train_output_{data_type}.pth")
        ],

        # Test data directory pattern
        'test_data_dir': "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC/topology_agnostic_data",
        'test_input_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/tsmc_merged_test_input_{data_type}.pth",
        'test_output_pattern': lambda cell, data_type: f"{{test_dir}}/{cell}/tsmc_merged_test_output_{data_type}.pth",

        # Model paths
        'maml_model_path': lambda data_type, innerdiv, meta, inner, num_iterations, layer_length=40: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/taskdivide_all/{data_type}_innerdiv{innerdiv}_meta{meta}_topology_agnostic_519traintask_full1DMAML_weights_3hidden_({layer_length})_{num_iterations}_inner{inner}_upgraded_tsmc.pth",
        'mlp_model_path': lambda data_type, model_type, num_iterations: f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/MLP_pretrained_model/training_loss_pretrained_tsmc_topology_agnostic_{data_type}_{model_type}_{num_iterations}.pth",
    }
}


def get_test_config(config_id):
    """
    Get test configuration by ID

    Args:
        config_id (int): Configuration ID (0-3)

    Returns:
        dict: Configuration dictionary

    Raises:
        ValueError: If config_id is not in valid range (0-3)
    """
    if config_id not in TEST_CONFIGS:
        raise ValueError(f"Invalid config ID: {config_id}. Must be 0-3.")

    return TEST_CONFIGS[config_id]


def get_train_data_paths(config_id, data_type='transition'):
    """
    Get training data paths for normalization

    Args:
        config_id (int): Configuration ID (0-3)
        data_type (str): 'cell' or 'transition'

    Returns:
        list: List of (input_path, output_path) tuples
    """
    config = get_test_config(config_id)
    return config['train_data_paths'](data_type)


def get_test_data_paths(config_id, cell, data_type='transition'):
    """
    Get test data paths for a specific cell

    Args:
        config_id (int): Configuration ID (0-3)
        cell (str): Cell name
        data_type (str): 'cell' or 'transition'

    Returns:
        tuple: (input_path, output_path)
    """
    config = get_test_config(config_id)
    test_dir = config['test_data_dir']

    input_path = config['test_input_pattern'](cell, data_type).format(test_dir=test_dir)
    output_path = config['test_output_pattern'](cell, data_type).format(test_dir=test_dir)

    return input_path, output_path


def get_maml_model_path(config_id, data_type='transition', innerdiv=100, meta=None, inner=1, num_iterations=None, layer_length=40, custom_path=None):
    """
    Get MAML model path

    Args:
        config_id (int): Configuration ID (0-3)
        data_type (str): 'cell' or 'transition'
        innerdiv (int): Inner learning rate divisor
        meta (int): Meta batch size (if None, uses default)
        inner (int): Inner loop steps
        num_iterations (int): Number of training iterations (if None, uses default)
        layer_length (int): Hidden layer size (default: 40)
        custom_path (str): Custom model path (overrides auto-detection)

    Returns:
        str: Model path
    """
    if custom_path:
        return custom_path

    config = get_test_config(config_id)

    if meta is None:
        meta = config['default_meta']

    if num_iterations is None:
        num_iterations = config['default_num_iterations']

    return config['maml_model_path'](data_type, innerdiv, meta, inner, num_iterations, layer_length)


def get_mlp_model_path(config_id, data_type='cell', model_type='aadam', num_iterations=300000, custom_path=None):
    """
    Get MLP model path

    Args:
        config_id (int): Configuration ID (0-3)
        data_type (str): 'cell' or 'transition'
        model_type (str): 'aadam' or 'mlp'
        num_iterations (int): Number of training iterations
        custom_path (str): Custom model path (overrides auto-detection)

    Returns:
        str: Model path
    """
    if custom_path:
        return custom_path

    config = get_test_config(config_id)
    return config['mlp_model_path'](data_type, model_type, num_iterations)


def print_available_configs():
    """
    Print all available test configurations
    """
    print("\n📋 Available test configurations:")
    print("="*80)
    for config_id, config in TEST_CONFIGS.items():
        print(f"\n  [{config_id}] {config['name']}")
        print(f"      Tech: {config['tech']}, Topology: {config['topology_type']}")
        print(f"      Default cells: {', '.join(config['default_cells'][:3])}{'...' if len(config['default_cells']) > 3 else ''}")
        print(f"      GPU: {config['default_gpu']}, Meta: {config['default_meta']}, Data type: {config['default_data_type']}")


if __name__ == "__main__":
    # Example usage
    print_available_configs()

    print("\n\n📝 Example usage:")
    print("-"*80)

    # Example 1: Get train data paths
    config_id = 0
    data_type = 'transition'
    train_paths = get_train_data_paths(config_id, data_type)
    print(f"\nConfig {config_id} - Training data paths:")
    for i, (inp, out) in enumerate(train_paths, 1):
        print(f"  {i}. Input:  {inp}")
        print(f"     Output: {out}")

    # Example 2: Get test data paths
    cell = 'NAND3x2'
    test_input, test_output = get_test_data_paths(config_id, cell, data_type)
    print(f"\nConfig {config_id} - Test data for cell '{cell}':")
    print(f"  Input:  {test_input}")
    print(f"  Output: {test_output}")

    # Example 3: Get model path
    model_path = get_maml_model_path(config_id, data_type, innerdiv=100, meta=64, inner=1)
    print(f"\nConfig {config_id} - MAML model path:")
    print(f"  {model_path}")
