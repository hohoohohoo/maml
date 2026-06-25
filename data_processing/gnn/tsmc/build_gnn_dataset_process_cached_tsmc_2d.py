#!/usr/bin/env python
"""
Build TSMC GNN dataset with 2-D V×T tensor format for MAML adaptation.

Each task = (corner, cell, delay_arc, slew_idx, load_idx), containing samples across
the full (61 voltage × N temperature) plane.

Train: 5 corners × (61 V × 6 T) → [61, 6, total_nodes, num_features]
Test:  per-cell file with [61, 5_test_temps, total_nodes, num_features]

See: docs/superpowers/specs/2026-05-27-gnn-dataset-2d-VT-design.md
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import argparse
import gc
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MLP', 'utils'))

# Reuse helpers and constants from the 1-D builder
from build_gnn_dataset_process_cached_tsmc import (
    FIXED_PARAM_A, FIXED_PARAM_B, FIXED_PARAM_C,
    TRAIN_TEMPERATURES, TEST_TEMPERATURES, TRAIN_CORNERS,
    INTRA_TOPOLOGY_CELLS,
    parse_tsmc_folder_name,
    get_abc_parameters,
    apply_topology_with_process_tsmc,
    temp_to_folder_str,
    get_expected_train_folders,
    get_test_folders,
    process_lib_file_for_unified,
    process_directory_for_unified,
)


def build_unified_datasets_2d(
    cache_path: str,
    cache_type: str,
    lib_base_path: str,
    output_dir: str,
    data_type: str = "cell",
    skip_train: bool = False,
    include_parasitic_cap: bool = False,
    voltage_mode: str = 'all_nodes',
    temperature_mode: str = 'mos_only',
    include_zeros_in_norm: bool = False,
    topology_suffix: str = "",
    slew_mode: str = 'all',
):
    """
    Build train and test datasets with 4-D V×T tensor format.

    Train: [61_V, 6_T, total_nodes, num_features]
    Test:  per-cell [61_V, 5_T, total_nodes_per_cell, num_features]
    """
    num_features = 12 if include_parasitic_cap else 11
    num_train_temps = len(TRAIN_TEMPERATURES)  # 6
    num_test_temps = len(TEST_TEMPERATURES)    # 5

    print("=" * 80)
    print("BUILDING TSMC GNN DATASET - 2-D V×T FORMAT")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Lib base path: {lib_base_path}")
    print(f"Output dir: {output_dir}")
    print(f"Data type: {data_type}")
    print(f"Include parasitic cap: {include_parasitic_cap}")
    print(f"Voltage mode: {voltage_mode}")
    print(f"Temperature mode: {temperature_mode}")
    print(f"Slew mode: {slew_mode}")
    print(f"Include zeros in norm: {include_zeros_in_norm}")
    print(f"Topology suffix: '{topology_suffix}'")
    print(f"Node features: {num_features}D")
    print(f"\nTrain temperatures: {TRAIN_TEMPERATURES}")
    print(f"Test temperatures: {TEST_TEMPERATURES}")
    print(f"Train corners: {TRAIN_CORNERS}")
    print(f"\n⚠️  Excluding INTRA_TOPOLOGY_CELLS from train data:")
    for cell in INTRA_TOPOLOGY_CELLS:
        print(f"   - {cell}")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading topology cache...")
    topology_cache = torch.load(cache_path, weights_only=False)
    print(f"   ✓ Loaded {len(topology_cache)} cells")

    lib_base_path = Path(lib_base_path)
    output_dir = Path(output_dir)
    if include_parasitic_cap:
        output_dir = output_dir / "with_parasitic_cap"
        print(f"   Using subdirectory for 12D features: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode suffix matches 1-D builder convention
    voltage_suffix = f"_{voltage_mode}" if voltage_mode != 'all_nodes' else ""
    temperature_suffix = f"_{temperature_mode}" if temperature_mode != 'mos_only' else ""
    slew_suffix = "_relpin" if slew_mode == 'related_pin_only' else ""
    mode_suffix = f"{topology_suffix}{voltage_suffix}{temperature_suffix}{slew_suffix}"

    # Validate train folders
    print(f"\n🔍 Checking expected train folders (5 corners × 6 temps = 30)...")
    _train_folders_flat, missing_train = get_expected_train_folders(lib_base_path)
    if missing_train:
        print(f"\n❌ ERROR: Missing {len(missing_train)} train folders!")
        for fname in missing_train:
            print(f"      - {fname}")
        raise FileNotFoundError(f"Missing {len(missing_train)} required train folders")
    print(f"   ✅ All 30 train folders found.")

    test_folders = get_test_folders(lib_base_path)
    print(f"   Found {len(test_folders)} test folders")

    # ==================== PROCESS TRAIN DATA ====================
    train_path = None
    num_voltages_observed = None

    if not skip_train:
        print(f"\n{'='*80}")
        print("PROCESSING TRAIN DATA (2-D V×T)")
        print(f"{'='*80}")

        all_train_tasks = []   # each entry: {'corner': str, 'samples_by_TV': {(v,t): sample}}

        for corner_idx, corner in enumerate(TRAIN_CORNERS):
            print(f"\n[Corner {corner_idx+1}/{len(TRAIN_CORNERS)}] {corner}")

            samples_2d_per_temp = []   # samples_2d_per_temp[temp_idx] = list[per_lib][task_idx]
            num_tasks_per_temp = []

            for temp_idx, temp in enumerate(TRAIN_TEMPERATURES):
                temp_str = temp_to_folder_str(temp)
                folder = lib_base_path / f"TSMC_{corner}_{temp_str}"
                process_params = get_abc_parameters(corner, temp)

                print(f"   [temp {temp_idx+1}/{num_train_temps}] {folder.name}")

                lib_files = sorted(folder.glob("*.lib"))
                if num_voltages_observed is None:
                    num_voltages_observed = len(lib_files)
                    print(f"      Number of voltage lib files: {num_voltages_observed}")
                elif len(lib_files) != num_voltages_observed:
                    raise RuntimeError(
                        f"Voltage axis mismatch: expected {num_voltages_observed} lib files, "
                        f"found {len(lib_files)} in {folder.name}"
                    )

                all_samples_per_lib, num_tasks_dir = process_directory_for_unified(
                    folder, topology_cache, cache_type, process_params, data_type,
                    include_parasitic_cap=include_parasitic_cap,
                    voltage_mode=voltage_mode,
                    temperature_mode=temperature_mode,
                    slew_mode=slew_mode,
                )
                print(f"      ✓ {num_tasks_dir} tasks from {len(lib_files)} lib files")
                samples_2d_per_temp.append(all_samples_per_lib)
                num_tasks_per_temp.append(num_tasks_dir)

            # Alignment check across 6 temps within this corner
            num_tasks_per_corner = min(num_tasks_per_temp) if num_tasks_per_temp else 0
            if len(set(num_tasks_per_temp)) > 1:
                print(f"   ⚠️ num_tasks varies across temps: {num_tasks_per_temp}, taking min={num_tasks_per_corner}")

            corner_tasks_added = 0
            excluded_count = 0
            skipped_mismatch = 0

            for task_idx in range(num_tasks_per_corner):
                # Confirm cell name alignment across all 6 temps (using lib_idx=0)
                cell_names_at_idx = set()
                for temp_idx in range(num_train_temps):
                    per_lib_lists = samples_2d_per_temp[temp_idx]
                    if 0 < len(per_lib_lists) and task_idx < len(per_lib_lists[0]):
                        cell_names_at_idx.add(per_lib_lists[0][task_idx]['cell_name'])
                if len(cell_names_at_idx) != 1:
                    skipped_mismatch += 1
                    continue
                cell_name = cell_names_at_idx.pop()
                if cell_name in INTRA_TOPOLOGY_CELLS:
                    excluded_count += 1
                    continue

                # Build the (V, T) plane for this task
                samples_by_TV = {}
                valid = True
                for temp_idx in range(num_train_temps):
                    per_lib_lists = samples_2d_per_temp[temp_idx]
                    for volt_idx in range(num_voltages_observed):
                        if volt_idx >= len(per_lib_lists) or task_idx >= len(per_lib_lists[volt_idx]):
                            valid = False
                            break
                        samples_by_TV[(volt_idx, temp_idx)] = per_lib_lists[volt_idx][task_idx]
                    if not valid:
                        break
                if not valid:
                    skipped_mismatch += 1
                    continue

                all_train_tasks.append({
                    'corner': corner,
                    'samples_by_TV': samples_by_TV,
                })
                corner_tasks_added += 1

            print(f"   ✓ {corner_tasks_added} tasks added; "
                  f"{excluded_count} INTRA_TOPOLOGY excluded; "
                  f"{skipped_mismatch} skipped (mismatch)")

            del samples_2d_per_temp
            gc.collect()

        print(f"\n📊 Total train tasks collected: {len(all_train_tasks)}")
        print(f"   num_voltages: {num_voltages_observed}, num_temps: {num_train_temps}")

        if not all_train_tasks:
            print("❌ No valid train tasks!")
            return

        # ----- Allocate 4-D tensors -----
        task_node_counts = []
        cell_names = []
        delay_types = []
        output_names = []
        task_corners = []
        for t in all_train_tasks:
            sample0 = t['samples_by_TV'][(0, 0)]
            task_node_counts.append(sample0['num_nodes'])
            cell_names.append(sample0['cell_name'])
            delay_types.append(sample0['delay_type'])
            output_names.append(sample0['output_name'])
            task_corners.append(t['corner'])

        num_tasks = len(all_train_tasks)
        total_nodes = sum(task_node_counts)

        print(f"\n🔧 Train task metadata: {num_tasks} tasks, {total_nodes} total nodes")
        print(f"\n💾 Allocating 4-D tensors:")
        print(f"   node_features: [{num_voltages_observed}, {num_train_temps}, {total_nodes}, {num_features}]")
        print(f"   outputs:       [{num_voltages_observed}, {num_train_temps}, {num_tasks}]")

        node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
        node_slices[1:] = np.cumsum(task_node_counts)

        all_node_features = np.zeros(
            (num_voltages_observed, num_train_temps, total_nodes, num_features),
            dtype=np.float32,
        )
        all_outputs = np.zeros(
            (num_voltages_observed, num_train_temps, num_tasks),
            dtype=np.float32,
        )

        # ----- Fill -----
        print(f"\n📝 Filling tensors...")
        for task_idx, t in enumerate(all_train_tasks):
            if task_idx % 10000 == 0:
                print(f"   Processing task {task_idx}/{num_tasks}...")
            node_start = node_slices[task_idx]
            node_end = node_slices[task_idx + 1]
            for (v, temp_idx), sample in t['samples_by_TV'].items():
                nf = sample['node_features']
                if isinstance(nf, torch.Tensor):
                    nf = nf.cpu().numpy()
                all_node_features[v, temp_idx, node_start:node_end, :] = nf
                out = sample['output']
                if isinstance(out, torch.Tensor):
                    out = out.item()
                all_outputs[v, temp_idx, task_idx] = out

        # ----- Extract voltage / temperature axis coordinates -----
        # Voltage value = the (already-applied) voltage feature (index 4) for any non-zero node at (v, t=0)
        voltages = np.zeros(num_voltages_observed, dtype=np.float32)
        for v in range(num_voltages_observed):
            v_slice = all_node_features[v, 0, :, 4]
            nz = v_slice[v_slice != 0]
            voltages[v] = float(nz[0]) if len(nz) > 0 else 0.0
        temperatures = np.array(TRAIN_TEMPERATURES, dtype=np.float32)

        # ----- Normalization stats (global, per feature index) -----
        norm_method = "all values" if include_zeros_in_norm else "non-zero values only"
        print(f"\n📊 Calculating normalization statistics ({norm_method})...")
        if include_parasitic_cap:
            NORMALIZE_INDICES = [4, 5, 6, 10, 11]
            NORMALIZE_NAMES = ['voltage', 'input_slew', 'output_load', 'temperature', 'parasitic_cap']
        else:
            NORMALIZE_INDICES = [4, 5, 6, 10]
            NORMALIZE_NAMES = ['voltage', 'input_slew', 'output_load', 'temperature']

        norm_stats = {}
        for idx, name in zip(NORMALIZE_INDICES, NORMALIZE_NAMES):
            feature_data = all_node_features[:, :, :, idx].flatten()
            if include_zeros_in_norm:
                mean = float(np.mean(feature_data))
                std = float(np.std(feature_data))
                if std == 0:
                    std = 1.0
                nz = feature_data[feature_data != 0]
                nz_ratio = len(nz) / len(feature_data) * 100
            else:
                nz = feature_data[feature_data != 0]
                if len(nz) > 0:
                    mean = float(np.mean(nz))
                    std = float(np.std(nz))
                else:
                    mean = 0.0
                    std = 1.0
                nz_ratio = len(nz) / len(feature_data) * 100
            norm_stats[idx] = {'name': name, 'mean': mean, 'std': std}
            print(f"   {name}: mean={mean:.6f}, std={std:.6f} (non-zero: {nz_ratio:.1f}%)")

        # ----- Save train -----
        train_path = output_dir / f"train_{data_type}_{cache_type}{mode_suffix}_2d.pth"
        print(f"\n💾 Saving train dataset: {train_path}")

        train_data = {
            'node_features': torch.from_numpy(all_node_features),
            'outputs': torch.from_numpy(all_outputs),
            'node_slices': torch.from_numpy(node_slices),
            'voltages': torch.from_numpy(voltages),
            'temperatures': torch.from_numpy(temperatures),
            'cell_names': cell_names,
            'delay_types': delay_types,
            'output_names': output_names,
            'task_corners': task_corners,
            'node_counts': task_node_counts,
            'num_tasks': num_tasks,
            'num_voltages': num_voltages_observed,
            'num_temps': num_train_temps,
            'num_features': num_features,
            'total_nodes': total_nodes,
            'format': 'unified_4d_VT',
            'process_node': 'TSMC',
            'data_type': data_type,
            'graph_mode': cache_type,
            'cache_path': cache_path,
            'train_corners': TRAIN_CORNERS,
            'train_temperatures': TRAIN_TEMPERATURES,
            'include_parasitic_cap': include_parasitic_cap,
            'voltage_mode': voltage_mode,
            'slew_mode': slew_mode,
            'temperature_mode': temperature_mode,
            'topology_suffix': topology_suffix,
            'excluded_cells': INTRA_TOPOLOGY_CELLS,
            'norm_stats': {
                'node_features': {
                    stats['name']: {'mean': stats['mean'], 'std': stats['std']}
                    for stats in norm_stats.values()
                }
            },
            'normalize_indices': NORMALIZE_INDICES,
            'normalize_names': NORMALIZE_NAMES,
            'normalize_nonzero_only': not include_zeros_in_norm,
            'include_zeros_in_norm': include_zeros_in_norm,
        }
        torch.save(train_data, train_path)
        print(f"   ✅ Saved: {train_path}")
        print(f"   node_features: {train_data['node_features'].shape}")
        print(f"   outputs: {train_data['outputs'].shape}")

        del all_node_features, all_outputs, all_train_tasks
        gc.collect()
    else:
        print(f"\n{'='*80}")
        print("SKIPPING TRAIN DATA PROCESSING (--skip_train)")
        print(f"{'='*80}")

    # ==================== PROCESS TEST DATA ====================
    print(f"\n{'='*80}")
    print("PROCESSING TEST DATA (2-D V×T)")
    print(f"{'='*80}")

    test_output_dir = output_dir / f"test_by_{data_type}_{cache_type}{mode_suffix}_2d"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = test_output_dir / ".temp_partials"
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_cell_names = set()

    print(f"\n📂 Step 1: Processing test folders and saving partials...")
    for dir_idx, folder in enumerate(test_folders):
        corner, temperature, is_variant = parse_tsmc_folder_name(folder.name)
        if corner is None or temperature is None:
            print(f"   [{dir_idx+1}] Skipping {folder.name} (unparseable)")
            continue

        process_params = get_abc_parameters(corner, temperature)
        tag = " [variant]" if is_variant else ""
        print(f"   [{dir_idx+1}/{len(test_folders)}] {folder.name}{tag}  "
              f"(corner={corner}, temp={temperature})")

        lib_files = sorted(folder.glob("*.lib"))
        if not lib_files:
            print(f"      ⚠️ No .lib files, skipping")
            continue

        all_samples_per_lib, num_tasks_dir = process_directory_for_unified(
            folder, topology_cache, cache_type, process_params, data_type,
            include_parasitic_cap=include_parasitic_cap,
            voltage_mode=voltage_mode,
            temperature_mode=temperature_mode,
            slew_mode=slew_mode,
        )
        print(f"      ✓ {num_tasks_dir} tasks from {len(lib_files)} lib files")

        folder_samples_by_cell = defaultdict(list)
        for task_idx in range(num_tasks_dir):
            samples_by_lib = {}
            cell_name = None
            valid = True
            for lib_idx in range(len(lib_files)):
                if lib_idx < len(all_samples_per_lib) and task_idx < len(all_samples_per_lib[lib_idx]):
                    s = all_samples_per_lib[lib_idx][task_idx]
                    samples_by_lib[lib_idx] = s
                    if cell_name is None:
                        cell_name = s['cell_name']
                else:
                    valid = False
                    break
            if valid and cell_name and len(samples_by_lib) == len(lib_files):
                folder_samples_by_cell[cell_name].append({
                    'corner': corner,
                    'temperature': temperature,
                    'samples_by_lib': samples_by_lib,
                })
                all_cell_names.add(cell_name)

        for cell_name, tasks in folder_samples_by_cell.items():
            partial_path = temp_dir / f"{cell_name}_partial_{dir_idx:04d}.pth"
            torch.save(tasks, partial_path)

        del all_samples_per_lib, folder_samples_by_cell
        gc.collect()

    print(f"\n   ✅ Processed {len(test_folders)} folders, found {len(all_cell_names)} unique cells")

    print(f"\n📦 Step 2: Merging partials into per-cell 2-D files...")
    print(f"   Output directory: {test_output_dir}")
    saved_count = 0

    for cell_idx, cell_name in enumerate(sorted(all_cell_names)):
        partial_files = sorted(temp_dir.glob(f"{cell_name}_partial_*.pth"))
        if not partial_files:
            continue

        # Load all partials for this cell — each partial is a list of task records
        # (each record: {'corner', 'temperature', 'samples_by_lib'})
        flat_records = []
        for pf in partial_files:
            recs = torch.load(pf, weights_only=False)
            flat_records.extend(recs)
            pf.unlink()
        if not flat_records:
            continue

        corners_seen = sorted({r['corner'] for r in flat_records})
        temps_seen = sorted({r['temperature'] for r in flat_records})

        # Group records by (corner, temp) → list (preserving folder-internal task_idx order)
        ct_groups: Dict[Tuple[str, float], List[dict]] = defaultdict(list)
        for r in flat_records:
            ct_groups[(r['corner'], r['temperature'])].append(r)

        # Require every (corner, temp) in the cell's grid to be populated
        counts = []
        missing_ct = []
        for c in corners_seen:
            for t in temps_seen:
                cnt = len(ct_groups.get((c, t), []))
                counts.append(cnt)
                if cnt == 0:
                    missing_ct.append((c, t))
        if missing_ct:
            print(f"   [{cell_idx+1}/{len(all_cell_names)}] {cell_name}: "
                  f"missing (corner, temp) groups {missing_ct} — skipping cell")
            continue

        tasks_per_corner = min(counts)
        if len(set(counts)) > 1:
            print(f"   [{cell_idx+1}/{len(all_cell_names)}] {cell_name}: "
                  f"task-count mismatch {sorted(set(counts))}, using min={tasks_per_corner}")

        num_corners = len(corners_seen)
        num_temps = len(temps_seen)
        num_libs_cell = len(flat_records[0]['samples_by_lib'])
        num_tasks_cell = num_corners * tasks_per_corner

        # Build task list: outer corner, inner cell-local task index
        task_index_list = [(corner, ti) for corner in corners_seen for ti in range(tasks_per_corner)]

        task_node_counts = []
        delay_types = []
        output_names = []
        task_corners_out = []
        for (corner, ti) in task_index_list:
            sample0 = ct_groups[(corner, temps_seen[0])][ti]['samples_by_lib'][0]
            task_node_counts.append(sample0['num_nodes'])
            delay_types.append(sample0['delay_type'])
            output_names.append(sample0['output_name'])
            task_corners_out.append(corner)

        total_nodes_cell = sum(task_node_counts)
        node_slices = np.zeros(num_tasks_cell + 1, dtype=np.int64)
        node_slices[1:] = np.cumsum(task_node_counts)

        node_features = np.zeros(
            (num_libs_cell, num_temps, total_nodes_cell, num_features),
            dtype=np.float32,
        )
        outputs = np.zeros((num_libs_cell, num_temps, num_tasks_cell), dtype=np.float32)

        for task_out_idx, (corner, ti) in enumerate(task_index_list):
            node_start = node_slices[task_out_idx]
            node_end = node_slices[task_out_idx + 1]
            for temp_idx, temp in enumerate(temps_seen):
                record = ct_groups[(corner, temp)][ti]
                samples_by_lib = record['samples_by_lib']
                for lib_idx, sample in samples_by_lib.items():
                    nf = sample['node_features']
                    if isinstance(nf, torch.Tensor):
                        nf = nf.cpu().numpy()
                    node_features[lib_idx, temp_idx, node_start:node_end, :] = nf
                    out = sample['output']
                    if isinstance(out, torch.Tensor):
                        out = out.item()
                    outputs[lib_idx, temp_idx, task_out_idx] = out

        # Voltages from any non-zero (v, t=0) feature index 4
        voltages = np.zeros(num_libs_cell, dtype=np.float32)
        for v in range(num_libs_cell):
            v_slice = node_features[v, 0, :, 4]
            nz = v_slice[v_slice != 0]
            voltages[v] = float(nz[0]) if len(nz) > 0 else 0.0
        temperatures_arr = np.array(temps_seen, dtype=np.float32)

        cell_path = test_output_dir / f"{cell_name}.pth"
        cell_data = {
            'node_features': torch.from_numpy(node_features),
            'outputs': torch.from_numpy(outputs),
            'node_slices': torch.from_numpy(node_slices),
            'voltages': torch.from_numpy(voltages),
            'temperatures': torch.from_numpy(temperatures_arr),
            'delay_types': delay_types,
            'output_names': output_names,
            'task_corners': task_corners_out,
            'cell_name': cell_name,
            'num_tasks': num_tasks_cell,
            'num_voltages': num_libs_cell,
            'num_temps': num_temps,
            'num_features': num_features,
            'total_nodes': total_nodes_cell,
            'format': 'unified_4d_VT',
            'include_parasitic_cap': include_parasitic_cap,
            'voltage_mode': voltage_mode,
            'slew_mode': slew_mode,
            'topology_suffix': topology_suffix,
            'normalize_nonzero_only': not include_zeros_in_norm,
            'include_zeros_in_norm': include_zeros_in_norm,
        }
        torch.save(cell_data, cell_path)
        saved_count += 1

        del node_features, outputs, flat_records, ct_groups
        gc.collect()

        if (cell_idx + 1) % 10 == 0:
            print(f"   Processed {cell_idx+1}/{len(all_cell_names)} cells")

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print(f"   ✅ Saved {saved_count} cell files")

    # ----- Summary -----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if not skip_train and train_path is not None:
        print(f"Train: {train_path}")
    print(f"Test:  {test_output_dir}")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TSMC GNN dataset (2-D V×T format)")
    parser.add_argument("--cache_path", type=str, required=True,
                        help="Path to topology cache")
    parser.add_argument("--cache_type", type=str, required=True,
                        choices=['full_graph', 'stage_aware'],
                        help="Cache type")
    parser.add_argument("--lib_base_path", type=str, required=True,
                        help="Base path to TSMC library files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--data_type", type=str, default="cell",
                        choices=['cell', 'transition'],
                        help="Data type: cell (delay) or transition (slew)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip train data processing")
    parser.add_argument("--include_parasitic_cap", action="store_true",
                        help="Include parasitic cap as 12th feature")
    parser.add_argument("--voltage_mode", type=str, default="all_nodes",
                        choices=['all_nodes', 'vdd_only', 'vdd_mos'])
    parser.add_argument("--temperature_mode", type=str, default="mos_only",
                        choices=["mos_only", "temp_all"])
    parser.add_argument("--include_zeros_in_norm", action="store_true",
                        help="Include zeros in normalization stats (default: non-zero only)")
    parser.add_argument("--topology_suffix", type=str, default="",
                        help="Topology option suffix (e.g., '_weighted_inputport')")
    parser.add_argument("--slew_mode", type=str, default="all",
                        choices=['all', 'related_pin_only'])

    args = parser.parse_args()
    build_unified_datasets_2d(
        cache_path=args.cache_path,
        cache_type=args.cache_type,
        lib_base_path=args.lib_base_path,
        output_dir=args.output_dir,
        data_type=args.data_type,
        skip_train=args.skip_train,
        include_parasitic_cap=args.include_parasitic_cap,
        voltage_mode=args.voltage_mode,
        temperature_mode=args.temperature_mode,
        include_zeros_in_norm=args.include_zeros_in_norm,
        topology_suffix=args.topology_suffix,
        slew_mode=args.slew_mode,
    )
