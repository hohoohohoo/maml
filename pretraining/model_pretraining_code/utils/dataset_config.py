#!/usr/bin/env python
# coding: utf-8

"""
Dataset Configuration Module
Manages different dataset configurations for MAML pretraining
"""

import torch

# Dataset configuration mapping
DATASET_CONFIGS = {
    0: {
        'name': 'ASAP7_intra_topology',
        'tech': 'asap7',
        'topology_type': 'intra',
        'description': 'ASAP7 intra-topology dataset (intra_topology_data_upgraded + unified_invbuf)'
    },
    1: {
        'name': 'ASAP7_topology_agnostic',
        'tech': 'asap7',
        'topology_type': 'agnostic',
        'description': 'ASAP7 topology-agnostic dataset (unified_invbuf + topology_agnostic_data_upgraded)'
    },
    2: {
        'name': 'TSMC_intra_topology',
        'tech': 'tsmc',
        'topology_type': 'intra',
        'description': 'TSMC intra-topology dataset (dataset_tsmc_processed/intra_topology_data)'
    },
    3: {
        'name': 'TSMC_topology_agnostic',
        'tech': 'tsmc',
        'topology_type': 'agnostic',
        'description': 'TSMC topology-agnostic dataset (dataset_tsmc_processed/topology_agnostic_data)'
    }
}


def get_dataset_config(config_id):
    """
    Get dataset configuration by ID

    Args:
        config_id (int): Dataset configuration ID (0-3)

    Returns:
        dict: Configuration dictionary containing name, tech, topology_type, and description

    Raises:
        ValueError: If config_id is not in valid range (0-3)
    """
    if config_id not in DATASET_CONFIGS:
        raise ValueError(f"Invalid dataset config ID: {config_id}. Must be 0-3.")

    return DATASET_CONFIGS[config_id]


def print_available_datasets():
    """
    Print all available dataset configurations
    """
    print("\n📋 Available dataset configurations:")
    for config_id, config in DATASET_CONFIGS.items():
        print(f"   [{config_id}] {config['name']}")
        print(f"       Tech: {config['tech']}, Topology: {config['topology_type']}")
        print(f"       {config['description']}")


def load_dataset_by_config(config_id, data_type='cell'):
    """
    Load dataset based on configuration ID

    Args:
        config_id (int): Dataset configuration ID (0-3)
        data_type (str): Data type - 'cell' or 'transition'

    Returns:
        tuple: (input_tensor, output_tensor)
            - input_tensor: torch.Tensor of shape [total_samples, 61, 9]
            - output_tensor: torch.Tensor of shape [total_samples, 61, 1]

    Raises:
        ValueError: If config_id is invalid
        FileNotFoundError: If dataset files cannot be found
    """
    config = get_dataset_config(config_id)
    tech = config['tech']
    topology_type = config['topology_type']

    print(f"\n📂 Loading dataset: {config['name']} ({data_type})")

    if config_id == 0:
        # ASAP7 intra-topology: intra_topology_data_upgraded + unified_invbuf
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded"
        test_data_input = torch.load(f"{data_dir}/{data_type}_intratopology_train_input.pth")
        test_data_output_1 = torch.load(f"{data_dir}/{data_type}_intratopology_train_output.pth")

        # Add unified_invbuf data
        test_data_input2 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_input_{data_type}.pth")
        test_data_output_2 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_output_{data_type}.pth")

        test_data_input = torch.cat([test_data_input, test_data_input2], dim=0)
        test_data_output_1 = torch.cat([test_data_output_1, test_data_output_2], dim=0)

    elif config_id == 1:
        # ASAP7 topology-agnostic: unified_invbuf + topology_agnostic_data_upgraded
        test_data_input = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_input_{data_type}.pth")
        test_data_output_1 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_output_{data_type}.pth")

        data_dir2 = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/topology_agnostic_data_upgraded"
        test_data_input2 = torch.load(f"{data_dir2}/{data_type}_topology_agnostic_train_input.pth")
        test_data_output_2 = torch.load(f"{data_dir2}/{data_type}_topology_agnostic_train_output.pth")

        test_data_input = torch.cat([test_data_input, test_data_input2], dim=0)
        test_data_output_1 = torch.cat([test_data_output_1, test_data_output_2], dim=0)

    elif config_id == 2:
        # TSMC intra-topology
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_tsmc_processed/intra_topology_data/"
        test_data_input = torch.load(f"{data_dir}/tsmc_intra_topology_train_input_{data_type}.pth")
        test_data_output_1 = torch.load(f"{data_dir}/tsmc_intra_topology_train_output_{data_type}.pth")

    elif config_id == 3:
        # TSMC topology-agnostic
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_tsmc_processed/topology_agnostic_data"
        test_data_input = torch.load(f"{data_dir}/tsmc_topology_agnostic_train_input_{data_type}.pth")
        test_data_output_1 = torch.load(f"{data_dir}/tsmc_topology_agnostic_train_output_{data_type}.pth")

    print(f"   ✅ Loaded: Input {test_data_input.shape}, Output {test_data_output_1.shape}")

    return test_data_input, test_data_output_1
