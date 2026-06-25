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
    # 1-D MLP TSMC dataset aligned with the 1-D / 2-D GNN convention:
    # train pool = 44 combinational train cells × 6 train temps
    #              {-25, 12.5, 37.5, 62.5, 87.5, 125}
    # test pool  = (44 train cells + 20 OOD cells) × 5 test temps
    #              {0, 25, 50, 75, 100}
    # Configs 2 and 3 share the SAME train data; topology_type only affects
    # the downstream model_path / result filename topology label so the
    # existing analyzers (analyze_gcn_single_arch / FG_vs_SA / etc)
    # automatically separate intra vs agnostic metrics by filename.
    # model_path_suffix '_combined' is retained for physical ckpt
    # filename compatibility with the existing on-disk checkpoints.
    2: {
        'name': 'TSMC_intra_topology',
        'tech': 'tsmc',
        'topology_type': 'intra',
        'model_path_suffix': '_combined',
        'description': 'TSMC 1-D MLP dataset (44 train cells × 6 train temps); '
                       'intra topology evaluation. Train data shared with config 3.'
    },
    3: {
        'name': 'TSMC_topology_agnostic',
        'tech': 'tsmc',
        'topology_type': 'agnostic',
        'model_path_suffix': '_combined',
        'description': 'TSMC 1-D MLP dataset (44 train cells × 6 train temps); '
                       'agnostic topology evaluation. Train data shared with config 2.'
    },
    4: {
        'name': 'TSMC_intra_topology_2d',
        'tech': 'tsmc',
        'topology_type': 'intra',
        'is_2d': True,
        'description': 'TSMC intra-topology 2-D V×T dataset (61V × 6T grid per task)'
    },
    5: {
        'name': 'TSMC_topology_agnostic_2d',
        'tech': 'tsmc',
        'topology_type': 'agnostic',
        'is_2d': True,
        'description': 'TSMC topology-agnostic 2-D V×T dataset (61V × 6T grid per task)'
    },
}


def get_dataset_config(config_id):
    """
    Get dataset configuration by ID

    Args:
        config_id (int): Dataset configuration ID (0-5)

    Returns:
        dict: Configuration dictionary containing name, tech, topology_type, and description

    Raises:
        ValueError: If config_id is not in valid range (0-5)
    """
    if config_id not in DATASET_CONFIGS:
        raise ValueError(
            f"Invalid dataset config ID: {config_id}. "
            f"Must be one of {sorted(DATASET_CONFIGS.keys())}."
        )

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
        config_id (int): Dataset configuration ID (0-5)
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

        # Handle dimension mismatch: squeeze all 3D tensors with last dim=1 to 2D
        if test_data_output_1.dim() == 3 and test_data_output_1.shape[-1] == 1:
            test_data_output_1 = test_data_output_1.squeeze(-1)
        if test_data_output_2.dim() == 3 and test_data_output_2.shape[-1] == 1:
            test_data_output_2 = test_data_output_2.squeeze(-1)
        if test_data_input.dim() == 3 and test_data_input.shape[-1] == 1:
            test_data_input = test_data_input.squeeze(-1)
        if test_data_input2.dim() == 3 and test_data_input2.shape[-1] == 1:
            test_data_input2 = test_data_input2.squeeze(-1)

        test_data_input = torch.cat([test_data_input, test_data_input2], dim=0)
        test_data_output_1 = torch.cat([test_data_output_1, test_data_output_2], dim=0)

    elif config_id == 1:
        # ASAP7 topology-agnostic: unified_invbuf + topology_agnostic_data_upgraded
        test_data_input = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_input_{data_type}.pth")
        test_data_output_1 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_output_{data_type}.pth")

        data_dir2 = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/topology_agnostic_data_upgraded"
        test_data_input2 = torch.load(f"{data_dir2}/{data_type}_topology_agnostic_train_input.pth")
        test_data_output_2 = torch.load(f"{data_dir2}/{data_type}_topology_agnostic_train_output.pth")

        # Handle dimension mismatch: squeeze all 3D tensors with last dim=1 to 2D
        if test_data_output_1.dim() == 3 and test_data_output_1.shape[-1] == 1:
            test_data_output_1 = test_data_output_1.squeeze(-1)
        if test_data_output_2.dim() == 3 and test_data_output_2.shape[-1] == 1:
            test_data_output_2 = test_data_output_2.squeeze(-1)
        if test_data_input.dim() == 3 and test_data_input.shape[-1] == 1:
            test_data_input = test_data_input.squeeze(-1)
        if test_data_input2.dim() == 3 and test_data_input2.shape[-1] == 1:
            test_data_input2 = test_data_input2.squeeze(-1)

        test_data_input = torch.cat([test_data_input, test_data_input2], dim=0)
        test_data_output_1 = torch.cat([test_data_output_1, test_data_output_2], dim=0)

    elif config_id in (2, 3):
        # TSMC 1-D MLP: configs 2 and 3 load identical train data.
        # File naming is hard-coded to 'topology_agnostic' by the builder,
        # but that is just a file-on-disk convention — both intra (config 2)
        # and agnostic (config 3) downstream evaluation pull from the same
        # train tensor here.
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/combined_data"
        test_data_input = torch.load(
            f"{data_dir}/tsmc_topology_agnostic_train_input_{data_type}.pth"
        )
        test_data_output_1 = torch.load(
            f"{data_dir}/tsmc_topology_agnostic_train_output_{data_type}.pth"
        )

    elif config_id == 4:
        # TSMC intra-topology 2-D V×T
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_2d/intra_topology_data"
        test_data_input = torch.load(
            f"{data_dir}/tsmc_intra_topology_train_input_{data_type}_2d.pth",
            weights_only=False, map_location='cpu',
        )
        test_data_output_1 = torch.load(
            f"{data_dir}/tsmc_intra_topology_train_output_{data_type}_2d.pth",
            weights_only=False, map_location='cpu',
        )

    elif config_id == 5:
        # TSMC topology-agnostic 2-D V×T
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_2d/topology_agnostic_data"
        test_data_input = torch.load(
            f"{data_dir}/tsmc_topology_agnostic_train_input_{data_type}_2d.pth",
            weights_only=False, map_location='cpu',
        )
        test_data_output_1 = torch.load(
            f"{data_dir}/tsmc_topology_agnostic_train_output_{data_type}_2d.pth",
            weights_only=False, map_location='cpu',
        )

    print(f"   ✅ Loaded: Input {test_data_input.shape}, Output {test_data_output_1.shape}")

    return test_data_input, test_data_output_1
