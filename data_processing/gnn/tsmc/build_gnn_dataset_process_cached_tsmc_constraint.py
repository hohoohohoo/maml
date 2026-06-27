"""
Parallel dataset builder for *constraint* LUTs (setup / hold / recovery /
removal / non_seq_setup / non_seq_hold).

Sibling of `build_gnn_dataset_process_cached_tsmc.py`.  Does **not** modify the
existing cell / transition pipeline — imports the shared topology / process /
folder utilities, swaps the extractor for `libdata_extract_MAML_constraint`,
and writes one PTH per timing category.

Mapping decisions for constraint LUTs (3×3 in lib → per-task 61-V curve):
  * One *task* = (cell × related_pin × constrained_pin × when × delay_type
                  × related_slew × constrained_slew × P-T) at 61 voltage points.
  * Graph topology key  = first available `output_topologies` of the cell
                           (Q / QN for FF cells — encodes the C2MOS storage path).
  * `delay_type` routing = rise_constraint → pull_up, fall_constraint → pull_down
                           (same convention as cell_rise → pull_up in the delay
                           pipeline; the inner-loop MAML adapts the mapping).
  * Slew placement:
      - related_slew    → `input_slew` arg + `slew_mode='related_pin_only'`
                           so it lands on the related_pin node (e.g., CP).
      - constrained_slew → `output_load` arg (placed at the output-port slot of
                            the graph — semantic approximation; the model
                            consumes it as a per-task numeric conditioning value).
  * Mandatory categories: 'setup', 'hold' (always built unless explicitly removed).
  * Optional categories : 'recovery', 'removal', 'non_seq_setup', 'non_seq_hold'
                           (opt-in via `--include_optional ...`).
  * One PTH per category — train_{cat}_<cache>.pth and
    test_by_{cat}_<cache>/<cell>.pth.

Output paths mirror the existing `train_cell_stage_aware.pth` /
`test_by_cell_stage_aware/<cell>.pth` convention, just with the category name.
"""

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Reuse infrastructure from the cell / transition pipeline.  This file lives in
# the same directory; we add `../MLP/utils` for the constraint extractor below.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / 'MLP' / 'utils'))

from build_gnn_dataset_process_cached_tsmc import (
    parse_tsmc_folder_name,
    get_abc_parameters,
    apply_topology_with_process_tsmc,
    get_expected_train_folders,
    get_test_folders,
    INTRA_TOPOLOGY_CELLS,
    TRAIN_CORNERS,
    TRAIN_TEMPERATURES,
    TEST_TEMPERATURES,
)
from libdata_extract_MAML_constraint import (
    parse_liberty_pin_blocks,
    flatten_pin_data,
    CONSTRAINT_TIMING_CATEGORIES,
    DEFAULT_CATEGORIES,
    OPTIONAL_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Per-lib processing — constraint variant
# ---------------------------------------------------------------------------
def process_lib_file_for_constraint(lib_file_path, topology_cache, cache_type, process_params,
                                    categories: Iterable[str],
                                    include_parasitic_cap: bool = False,
                                    voltage_mode: str = 'all_nodes',
                                    temperature_mode: str = 'mos_only',
                                    slew_mode: str = 'related_pin_only',
                                    rescale_constrained_to_load: float = 1.0) -> Dict[str, list]:
    """Process one .lib file and return per-category sample lists (LUT order).

    Returns a dict keyed by category name. Each value is a list of minimal
    samples in *the same order across lib files for the same folder*, so the
    existing alignment-by-index logic in the train/test loops still works.
    """
    categories = tuple(categories)

    with open(lib_file_path, "r") as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    by_cat, _ = flatten_pin_data(pin_data, include_categories=categories)

    out: Dict[str, list] = {cat: [] for cat in categories}

    for category in categories:
        rows = by_cat.get(category, [])
        for row in rows:
            cell_name = row['cell']
            if cell_name not in topology_cache:
                continue
            cell_cache = topology_cache[cell_name]

            # Use the FF cell's primary output (Q for almost every FF in TSMC)
            # as the graph topology key.  No constraint LUT has a meaningful
            # "output" of its own — the storage path through the master latch
            # is what physically determines setup/hold.
            #
            # stage_aware cache stores per-(output, direction) subgraphs under
            # 'output_topologies'.  full_graph cache stores a single per-cell
            # adjacency_matrix and only carries the cell's output pin list in
            # 'output_nodes' — so we branch on cache_type to pick the right
            # source for output_pin.
            if cache_type == 'stage_aware':
                output_topologies = cell_cache.get('output_topologies', {})
                if not output_topologies:
                    continue
                output_pin = next(iter(output_topologies.keys()))
            else:  # full_graph
                output_nodes = cell_cache.get('output_nodes', [])
                if not output_nodes:
                    continue
                output_pin = output_nodes[0]

            related_pin = row.get('related_pin', '') or None
            external_inputs = cell_cache.get('external_inputs', [])

            # Constraint LUT axes: index_1 = related_pin_transition (related slew),
            #                      index_2 = constrained_pin_transition (constrained slew).
            related_slews     = row.get('index_1', [])
            constrained_slews = row.get('index_2', [])
            values            = row.get('values', [[]])
            constraint_dt     = row['delay_type']             # 'rise_constraint' / 'fall_constraint'
            timing_type       = row.get('timing_type', '')
            constrained_pin   = row.get('pin_name', '')
            when_clause       = row.get('when', '')

            # Route to the same pull_up/pull_down keys the cell/transition
            # pipeline uses; the inner-loop MAML adaptation can absorb the
            # semantic mismatch.
            stage_delay_type = 'rise_transition' if 'rise' in constraint_dt else 'fall_transition'

            actual_rows = len(values) if isinstance(values, list) else 0
            actual_cols = len(values[0]) if actual_rows > 0 and isinstance(values[0], list) else 0
            n_rows = min(len(related_slews), actual_rows) if actual_rows > 0 else len(related_slews)
            n_cols = min(len(constrained_slews), actual_cols) if actual_cols > 0 else len(constrained_slews)

            for ri in range(n_rows):
                for ci in range(n_cols):
                    related_slew     = float(related_slews[ri])
                    constrained_slew = float(constrained_slews[ci])
                    output_value     = float(values[ri][ci])
                    # Rescale the constrained-slew axis (ns) into the output_load (pF) scale that the
                    # pretrained model expects.  TSMC lib templates have slew_max ≈ 0.6113 ns and
                    # load_max ≈ 0.1189 pF; the recommended ratio is 0.1189 / 0.6113 ≈ 0.1945.  Passing
                    # 1.0 disables the rescale (use only when the downstream model has been retrained
                    # on the raw slew distribution).
                    constrained_slew_scaled = constrained_slew * rescale_constrained_to_load
                    try:
                        graph_sample = apply_topology_with_process_tsmc(
                            topology_cache, cache_type, cell_name, output_pin, stage_delay_type,
                            row['Voltage'], related_slew, constrained_slew_scaled, external_inputs, process_params,
                            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
                            temperature_mode=temperature_mode, slew_mode=slew_mode,
                            related_pin=related_pin,
                        )
                        expected_features = 12 if include_parasitic_cap else 11
                        if graph_sample['node_features'].shape[1] != expected_features:
                            continue
                        out[category].append({
                            'node_features':     graph_sample['node_features'],
                            'output':            output_value,
                            'cell_name':         cell_name,
                            'delay_type':        stage_delay_type,
                            'output_name':       output_pin,
                            'num_nodes':         graph_sample['node_features'].shape[0],
                            # constraint-specific metadata (kept for downstream loaders)
                            'category':          category,
                            'timing_type':       timing_type,
                            'related_pin':       related_pin or '',
                            'constrained_pin':   constrained_pin,
                            'when':              when_clause,
                            'related_slew':      related_slew,
                            'constrained_slew':  constrained_slew,
                            'constrained_slew_scaled': constrained_slew_scaled,
                        })
                    except Exception:
                        # No alt-output fallback for constraint (Q/QN guaranteed by FF lib).
                        continue

    return out


def process_directory_for_constraint(folder_path, topology_cache, cache_type, process_params,
                                     categories: Iterable[str],
                                     include_parasitic_cap: bool = False,
                                     voltage_mode: str = 'all_nodes',
                                     temperature_mode: str = 'mos_only',
                                     slew_mode: str = 'related_pin_only',
                                     rescale_constrained_to_load: float = 1.0,
                                     ) -> Tuple[Dict[str, List[list]], Dict[str, int]]:
    """Process every .lib in `folder_path` for the requested categories.

    Returns
    -------
    samples_per_lib : dict[category, list[list[sample]]]
        Per-category, one entry per lib file (61 V points = 61 libs).
    num_tasks : dict[category, int]
        Number of constraint tasks per category in the first lib.
    """
    lib_files = sorted(folder_path.glob("*.lib"))
    if not lib_files:
        return {cat: [] for cat in categories}, {cat: 0 for cat in categories}

    categories = tuple(categories)
    samples_per_lib: Dict[str, List[list]] = {cat: [] for cat in categories}
    for lib in lib_files:
        per_cat = process_lib_file_for_constraint(
            str(lib), topology_cache, cache_type, process_params, categories,
            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
            temperature_mode=temperature_mode, slew_mode=slew_mode,
            rescale_constrained_to_load=rescale_constrained_to_load,
        )
        for cat in categories:
            samples_per_lib[cat].append(per_cat[cat])

    num_tasks = {cat: (len(samples_per_lib[cat][0]) if samples_per_lib[cat] else 0)
                 for cat in categories}
    # Surface alignment problems early — they break the per-task lib-index join.
    for cat in categories:
        counts = [len(s) for s in samples_per_lib[cat]]
        if counts and len(set(counts)) > 1:
            print(f"   ⚠️ {cat}: sample-count mismatch across libs (unique counts={sorted(set(counts))})")
    return samples_per_lib, num_tasks


# ---------------------------------------------------------------------------
# Train tensor packing helpers
# ---------------------------------------------------------------------------
def _build_train_tensor(all_train_tasks: List[dict], num_libs: int, num_features: int):
    """Common 3D-tensor packing for one category's train tasks."""
    task_node_counts, cell_names, delay_types, output_names = [], [], [], []
    for ti in all_train_tasks:
        s = ti['samples_by_lib'][0]
        task_node_counts.append(s['num_nodes'])
        cell_names.append(s['cell_name'])
        delay_types.append(s['delay_type'])
        output_names.append(s['output_name'])
    num_tasks = len(all_train_tasks)
    total_nodes = sum(task_node_counts)
    node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(task_node_counts)

    all_nf = np.zeros((num_libs, total_nodes, num_features), dtype=np.float32)
    all_y  = np.zeros((num_libs, num_tasks), dtype=np.float32)
    for ti_idx, ti in enumerate(all_train_tasks):
        ns, ne = node_slices[ti_idx], node_slices[ti_idx + 1]
        for li, s in ti['samples_by_lib'].items():
            nf = s['node_features']
            if isinstance(nf, torch.Tensor):
                nf = nf.cpu().numpy()
            all_nf[li, ns:ne, :] = nf
            y = s['output']
            if isinstance(y, torch.Tensor):
                y = y.item()
            all_y[li, ti_idx] = y
    return all_nf, all_y, node_slices, cell_names, delay_types, output_names, total_nodes


def _compute_norm_stats(all_nf: np.ndarray, include_parasitic_cap: bool, include_zeros_in_norm: bool):
    if include_parasitic_cap:
        idxs, names = [4, 5, 6, 10, 11], ['voltage', 'input_slew', 'output_load', 'temperature', 'parasitic_cap']
    else:
        idxs, names = [4, 5, 6, 10], ['voltage', 'input_slew', 'output_load', 'temperature']
    stats = {}
    for idx, name in zip(idxs, names):
        data = all_nf[:, :, idx].flatten()
        if include_zeros_in_norm:
            mean = float(np.mean(data))
            std  = float(np.std(data))
        else:
            nz = data[data != 0]
            if len(nz) > 0:
                mean = float(np.mean(nz))
                std  = float(np.std(nz))
            else:
                mean, std = 0.0, 1.0
        if std == 0:
            std = 1.0
        stats[idx] = {'name': name, 'mean': mean, 'std': std}
    return stats, idxs, names


# ---------------------------------------------------------------------------
# Orchestrator helpers — setup / collect / save / extract / merge
# ---------------------------------------------------------------------------

def _print_config_banner(categories, cache_path, cache_type, lib_base_path, output_dir,
                        include_parasitic_cap, voltage_mode, temperature_mode,
                        slew_mode, topology_suffix) -> None:
    print("=" * 80)
    print("BUILDING TSMC GNN DATASET - CONSTRAINT LUTs")
    print("=" * 80)
    print(f"Categories          : {categories}")
    print(f"Cache path          : {cache_path}")
    print(f"Cache type          : {cache_type}")
    print(f"Lib base path       : {lib_base_path}")
    print(f"Output dir          : {output_dir}")
    print(f"Include parasitic   : {include_parasitic_cap}")
    print(f"Voltage / Temp mode : {voltage_mode} / {temperature_mode}")
    print(f"Slew mode           : {slew_mode}   (related_pin_only recommended for constraint)")
    print(f"Topology suffix     : '{topology_suffix}'")
    print(f"\nTrain corners×temps  : {TRAIN_CORNERS} × {TRAIN_TEMPERATURES}")
    print(f"Excluded from train  : {INTRA_TOPOLOGY_CELLS}")
    print("=" * 80)


def _setup_paths_and_suffix(output_dir, include_parasitic_cap, voltage_mode,
                            temperature_mode, slew_mode, topology_suffix,
                            rescale_constrained_to_load) -> Tuple[Path, str]:
    output_dir = Path(output_dir)
    if include_parasitic_cap:
        output_dir = output_dir / "with_parasitic_cap"
    output_dir.mkdir(parents=True, exist_ok=True)

    voltage_suffix     = f"_{voltage_mode}"     if voltage_mode     != 'all_nodes' else ""
    temperature_suffix = f"_{temperature_mode}" if temperature_mode != 'mos_only'  else ""
    slew_suffix        = "_relpin" if slew_mode == 'related_pin_only' else ""
    # `_scaledload` marks PTHs whose constrained-slew axis was rescaled into the output_load (pF)
    # scale so the pretrained cell-delay model can consume it without distribution shift.
    scale_suffix       = "_scaledload" if rescale_constrained_to_load != 1.0 else ""
    mode_suffix        = f"{topology_suffix}{voltage_suffix}{temperature_suffix}{slew_suffix}{scale_suffix}"
    return output_dir, mode_suffix


def _collect_train_tasks_per_category(
    train_folders, topology_cache, cache_type, categories,
    include_parasitic_cap, voltage_mode, temperature_mode, slew_mode,
    rescale_constrained_to_load,
) -> Tuple[Dict[str, list], Optional[int]]:
    """Run process_directory_for_constraint over all train folders and fold the
    per-lib results into per-category aligned task lists. Excludes
    INTRA_TOPOLOGY_CELLS. Returns (all_train_tasks_per_cat, num_libs).
    """
    all_train_tasks_per_cat: Dict[str, list] = {cat: [] for cat in categories}
    num_libs: Optional[int] = None

    for d_idx, folder in enumerate(train_folders):
        corner, temperature, _ = parse_tsmc_folder_name(folder.name)
        process_params = get_abc_parameters(corner, temperature)
        print(f"\n[{d_idx+1}/{len(train_folders)}] {folder.name} (corner={corner}, temp={temperature})")

        lib_files = sorted(folder.glob("*.lib"))
        if not lib_files:
            print(f"   ⚠️  no .lib — skipping")
            continue
        if num_libs is None:
            num_libs = len(lib_files)
            print(f"   num libs (voltage points): {num_libs}")

        samples_per_lib, num_tasks_dir = process_directory_for_constraint(
            folder, topology_cache, cache_type, process_params, categories,
            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
            temperature_mode=temperature_mode, slew_mode=slew_mode,
            rescale_constrained_to_load=rescale_constrained_to_load,
        )

        for cat in categories:
            added = excluded = 0
            for ti in range(num_tasks_dir[cat]):
                by_lib = {}
                ok = True
                for li in range(num_libs):
                    if li < len(samples_per_lib[cat]) and ti < len(samples_per_lib[cat][li]):
                        by_lib[li] = samples_per_lib[cat][li][ti]
                    else:
                        ok = False
                        break
                if not (ok and len(by_lib) == num_libs):
                    continue
                cell_name = by_lib[0]['cell_name']
                if cell_name in INTRA_TOPOLOGY_CELLS:
                    excluded += 1
                    continue
                all_train_tasks_per_cat[cat].append({
                    'dir_name': folder.name, 'corner': corner,
                    'temperature': temperature, 'samples_by_lib': by_lib,
                })
                added += 1
            if added or excluded:
                print(f"   [{cat:14s}] +{added:4d} tasks  (excluded {excluded})")
        gc.collect()
    return all_train_tasks_per_cat, num_libs


def _save_train_for_category(
    category, tasks, num_libs, num_features, output_dir: Path,
    cache_type, mode_suffix, cache_path,
    include_parasitic_cap, include_zeros_in_norm,
    voltage_mode, slew_mode, topology_suffix, num_conditions,
) -> Optional[Path]:
    """Build train tensor + norm stats + save for a single constraint category."""
    if not tasks:
        print(f"\n⚠️  {category}: no train tasks — skipping save")
        return None
    print(f"\n📊 [{category}] {len(tasks)} train tasks")

    all_nf, all_y, node_slices, cells, dtypes, onames, total_nodes = _build_train_tensor(
        tasks, num_libs, num_features,
    )
    stats, idxs, names = _compute_norm_stats(all_nf, include_parasitic_cap, include_zeros_in_norm)

    train_path = output_dir / f"train_{category}_{cache_type}{mode_suffix}.pth"
    print(f"💾 saving {train_path}")
    torch.save({
        'node_features':  torch.from_numpy(all_nf),
        'outputs':        torch.from_numpy(all_y),
        'node_slices':    torch.from_numpy(node_slices),
        'cell_names':     cells,
        'delay_types':    dtypes,
        'output_names':   onames,
        'node_counts':    [s['num_nodes'] for ti in tasks for s in [ti['samples_by_lib'][0]]],
        'num_tasks':      len(tasks),
        'num_libs':       num_libs,
        'num_features':   num_features,
        'total_nodes':    total_nodes,
        'format':         'unified_3d',
        'process_node':   'TSMC',
        'data_type':      category,                       # category as data_type
        'data_family':    'constraint',                   # group tag
        'graph_mode':     cache_type,
        'cache_path':     cache_path,
        'train_corners':  TRAIN_CORNERS,
        'train_temperatures': TRAIN_TEMPERATURES,
        'num_conditions': num_conditions,
        'include_parasitic_cap': include_parasitic_cap,
        'voltage_mode':   voltage_mode,
        'slew_mode':      slew_mode,
        'topology_suffix': topology_suffix,
        'excluded_cells': INTRA_TOPOLOGY_CELLS,
        'norm_stats': {
            'node_features': {s['name']: {'mean': s['mean'], 'std': s['std']}
                              for _, s in stats.items()}
        },
        'normalize_indices':     idxs,
        'normalize_names':       names,
        'normalize_nonzero_only': not include_zeros_in_norm,
        'include_zeros_in_norm':  include_zeros_in_norm,
        # Documentation breadcrumb for downstream consumers:
        'constraint_axis_semantics': {
            'feature_slot_5_input_slew':  'related_pin_transition (clock slew etc.)',
            'feature_slot_6_output_load': 'constrained_pin_transition (data slew etc.)',
        },
    }, train_path)
    print(f"   ✅ node_features {all_nf.shape}, outputs {all_y.shape}")

    del all_nf, all_y
    gc.collect()
    return train_path


def _extract_test_partials_constraint(
    test_folders, topology_cache, cache_type, categories,
    temp_dirs: Dict[str, Path],
    include_parasitic_cap, voltage_mode, temperature_mode, slew_mode,
    rescale_constrained_to_load,
) -> Dict[str, set]:
    """Step 1: per test folder, dump per-cat per-cell partial .pth files into
    `temp_dirs[cat]`. Returns the set of cells seen per category."""
    seen_cells_per_cat: Dict[str, set] = {cat: set() for cat in categories}

    for d_idx, folder in enumerate(test_folders):
        corner, temperature, is_variant = parse_tsmc_folder_name(folder.name)
        process_params = get_abc_parameters(corner, temperature)
        variant_tag = " [variant]" if is_variant else ""
        print(f"\n[{d_idx+1}/{len(test_folders)}] {folder.name}{variant_tag}")
        lib_files = sorted(folder.glob("*.lib"))
        if not lib_files:
            print(f"   ⚠️  no .lib — skipping")
            continue

        samples_per_lib, num_tasks_dir = process_directory_for_constraint(
            folder, topology_cache, cache_type, process_params, categories,
            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
            temperature_mode=temperature_mode, slew_mode=slew_mode,
            rescale_constrained_to_load=rescale_constrained_to_load,
        )
        n_libs_dir = len(lib_files)

        for cat in categories:
            by_cell: Dict[str, list] = defaultdict(list)
            for ti in range(num_tasks_dir[cat]):
                by_lib = {}
                ok = True
                cell_name = None
                for li in range(n_libs_dir):
                    if li < len(samples_per_lib[cat]) and ti < len(samples_per_lib[cat][li]):
                        s = samples_per_lib[cat][li][ti]
                        by_lib[li] = s
                        if cell_name is None:
                            cell_name = s['cell_name']
                    else:
                        ok = False
                        break
                if ok and len(by_lib) == n_libs_dir and cell_name:
                    by_cell[cell_name].append({'dir_name': folder.name, 'samples_by_lib': by_lib})
                    seen_cells_per_cat[cat].add(cell_name)
            for cell_name, tlist in by_cell.items():
                torch.save(tlist, temp_dirs[cat] / f"{cell_name}_partial_{d_idx:04d}.pth")
        del samples_per_lib
        gc.collect()

    return seen_cells_per_cat


def _merge_one_cell_for_category(
    cell_name: str, temp_dir: Path, test_dir: Path,
    category, num_features, include_parasitic_cap, voltage_mode,
    slew_mode, topology_suffix, include_zeros_in_norm,
) -> bool:
    """Load + merge all partials of `cell_name` into the per-cell .pth.
    Returns True if saved."""
    partials = sorted(temp_dir.glob(f"{cell_name}_partial_*.pth"))
    if not partials:
        return False

    all_tasks = []
    for pf in partials:
        all_tasks.extend(torch.load(pf, weights_only=False))
        pf.unlink()
    if not all_tasks:
        return False

    num_libs_cell = len(all_tasks[0]['samples_by_lib'])
    num_tasks = len(all_tasks)
    tn = [t['samples_by_lib'][0]['num_nodes'] for t in all_tasks]
    dt = [t['samples_by_lib'][0].get('delay_type', 'rise_transition') for t in all_tasks]
    on = [t['samples_by_lib'][0].get('output_name', '') for t in all_tasks]
    total_nodes = sum(tn)
    ns = np.zeros(num_tasks + 1, dtype=np.int64)
    ns[1:] = np.cumsum(tn)

    nf = np.zeros((num_libs_cell, total_nodes, num_features), dtype=np.float32)
    yy = np.zeros((num_libs_cell, num_tasks), dtype=np.float32)
    for ti, t in enumerate(all_tasks):
        a, b = ns[ti], ns[ti + 1]
        for li, s in t['samples_by_lib'].items():
            x = s['node_features']
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            nf[li, a:b, :] = x
            yi = s['output']
            if isinstance(yi, torch.Tensor):
                yi = yi.item()
            yy[li, ti] = yi

    cell_data = {
        'node_features':  torch.from_numpy(nf),
        'outputs':        torch.from_numpy(yy),
        'node_slices':    torch.from_numpy(ns),
        'delay_types':    dt,
        'output_names':   on,
        'num_tasks':      num_tasks,
        'num_libs':       num_libs_cell,
        'num_features':   num_features,
        'total_nodes':    total_nodes,
        'cell_name':      cell_name,
        'format':         'unified_3d',
        'data_type':      category,
        'data_family':    'constraint',
        'include_parasitic_cap': include_parasitic_cap,
        'voltage_mode':   voltage_mode,
        'slew_mode':      slew_mode,
        'topology_suffix': topology_suffix,
        'normalize_nonzero_only': not include_zeros_in_norm,
        'include_zeros_in_norm':  include_zeros_in_norm,
    }
    torch.save(cell_data, test_dir / f"{cell_name}.pth")
    del all_tasks, nf, yy, cell_data
    gc.collect()
    return True


def _merge_all_test_partials(
    categories, seen_cells_per_cat, temp_dirs, test_dirs,
    num_features, include_parasitic_cap, voltage_mode,
    slew_mode, topology_suffix, include_zeros_in_norm,
) -> None:
    print(f"\n📦 Merging partials...")
    for cat in categories:
        saved = 0
        cells_sorted = sorted(seen_cells_per_cat[cat])
        for cn in cells_sorted:
            ok = _merge_one_cell_for_category(
                cn, temp_dirs[cat], test_dirs[cat], cat,
                num_features, include_parasitic_cap, voltage_mode,
                slew_mode, topology_suffix, include_zeros_in_norm,
            )
            if ok:
                saved += 1
        try:
            temp_dirs[cat].rmdir()
        except OSError:
            pass
        print(f"   [{cat:14s}] saved {saved} per-cell .pth files in {test_dirs[cat]}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_constraint_datasets(
    cache_path: str,
    cache_type: str,
    lib_base_path: str,
    output_dir: str,
    categories: Iterable[str],
    skip_train: bool = False,
    include_parasitic_cap: bool = False,
    voltage_mode: str = 'all_nodes',
    temperature_mode: str = 'mos_only',
    include_zeros_in_norm: bool = False,
    topology_suffix: str = "",
    slew_mode: str = 'related_pin_only',
    rescale_constrained_to_load: float = 1.0,
):
    """Build constraint datasets (per-category train PTH + test PTH-per-cell).

    Mirrors `build_unified_datasets` from the cell/transition pipeline.  One
    pass over the lib corpus extracts every requested category, so the parser
    cost is shared.
    """
    categories = tuple(categories)
    num_features = 12 if include_parasitic_cap else 11

    _print_config_banner(
        categories, cache_path, cache_type, lib_base_path, output_dir,
        include_parasitic_cap, voltage_mode, temperature_mode, slew_mode, topology_suffix,
    )

    print(f"\n📦 Loading topology cache...")
    topology_cache = torch.load(cache_path, weights_only=False)
    print(f"   ✓ {len(topology_cache)} cells in cache")

    lib_base_path = Path(lib_base_path)
    output_dir, mode_suffix = _setup_paths_and_suffix(
        output_dir, include_parasitic_cap, voltage_mode, temperature_mode,
        slew_mode, topology_suffix, rescale_constrained_to_load,
    )

    if not skip_train:
        print(f"\n🔍 Checking train folders...")
        train_folders, missing = get_expected_train_folders(lib_base_path)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} required train folders: {missing}")
        print(f"   ✅ {len(train_folders)} train folders")
    else:
        train_folders = []
        print("\n(skip_train=True — train-folder discovery skipped)")
    test_folders = get_test_folders(lib_base_path)
    print(f"   Found {len(test_folders)} test folders")

    # ---- Train ----
    num_libs: Optional[int] = None
    if not skip_train:
        print(f"\n{'='*80}\nPROCESSING TRAIN DATA — per category\n{'='*80}")
        all_train_tasks_per_cat, num_libs = _collect_train_tasks_per_category(
            train_folders, topology_cache, cache_type, categories,
            include_parasitic_cap, voltage_mode, temperature_mode, slew_mode,
            rescale_constrained_to_load,
        )
        for cat in categories:
            _save_train_for_category(
                cat, all_train_tasks_per_cat[cat], num_libs, num_features, output_dir,
                cache_type, mode_suffix, cache_path,
                include_parasitic_cap, include_zeros_in_norm,
                voltage_mode, slew_mode, topology_suffix, len(train_folders),
            )
        del all_train_tasks_per_cat
        gc.collect()
    else:
        print("\nskip_train=True — train datasets unchanged")
        if num_libs is None:
            num_libs = 61

    # ---- Test ----
    print(f"\n{'='*80}\nPROCESSING TEST DATA — per cell, per category\n{'='*80}")

    test_dirs = {cat: output_dir / f"test_by_{cat}_{cache_type}{mode_suffix}" for cat in categories}
    temp_dirs = {cat: d / ".temp_partials" for cat, d in test_dirs.items()}
    for d in list(test_dirs.values()) + list(temp_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    seen_cells_per_cat = _extract_test_partials_constraint(
        test_folders, topology_cache, cache_type, categories,
        temp_dirs, include_parasitic_cap, voltage_mode, temperature_mode,
        slew_mode, rescale_constrained_to_load,
    )
    _merge_all_test_partials(
        categories, seen_cells_per_cat, temp_dirs, test_dirs,
        num_features, include_parasitic_cap, voltage_mode,
        slew_mode, topology_suffix, include_zeros_in_norm,
    )

    # ---- Summary ----
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    if not skip_train:
        for cat in categories:
            p = output_dir / f"train_{cat}_{cache_type}{mode_suffix}.pth"
            print(f"  Train : {p}")
    for cat in categories:
        print(f"  Test  : {test_dirs[cat]}")
    print("\n✅ Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_categories(args) -> List[str]:
    cats = list(DEFAULT_CATEGORIES)                       # setup, hold (always)
    for opt in args.include_optional or []:
        if opt not in OPTIONAL_CATEGORIES:
            raise SystemExit(f"--include_optional value '{opt}' not in {OPTIONAL_CATEGORIES}")
        if opt not in cats:
            cats.append(opt)
    return cats


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build TSMC GNN constraint-LUT datasets (parallel to the cell/transition pipeline)."
    )
    p.add_argument("--cache_path",    type=str, required=True)
    p.add_argument("--cache_type",    type=str, required=True, choices=['full_graph', 'stage_aware'])
    p.add_argument("--lib_base_path", type=str, required=True)
    p.add_argument("--output_dir",    type=str, required=True)
    p.add_argument("--include_optional", type=str, nargs="*", default=[],
                   choices=list(OPTIONAL_CATEGORIES),
                   help="Additional constraint categories to build alongside the mandatory setup/hold. "
                        f"Choices: {OPTIONAL_CATEGORIES}.")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--include_parasitic_cap", action="store_true")
    p.add_argument("--voltage_mode",     type=str, default="all_nodes",
                   choices=['all_nodes', 'vdd_only', 'vdd_mos'])
    p.add_argument("--temperature_mode", type=str, default="mos_only",
                   choices=['mos_only', 'temp_all'])
    p.add_argument("--include_zeros_in_norm", action="store_true")
    p.add_argument("--topology_suffix", type=str, default="")
    p.add_argument("--slew_mode", type=str, default="related_pin_only",
                   choices=['all', 'related_pin_only'],
                   help="Where to place the index_1 (related slew) feature.  "
                        "'related_pin_only' (default for constraint) puts it on the related_pin node, "
                        "matching how clock-slew conditioning actually flows.")
    p.add_argument("--rescale_constrained_to_load", type=float, default=1.0,
                   help="Multiplier applied to the index_2 (constrained_pin_transition) slew before "
                        "writing it into the output_load (pF) slot of the graph feature vector.  "
                        "The pretrained cell-delay model was trained with output_load in pF "
                        "(slot 6 range ~0.0002-0.12 pF) but constraint_template_3x3 axis 2 is a slew "
                        "in ns (range ~0.0017-0.61 ns).  Default 1.0 leaves the raw slew (≈ +20 sigma "
                        "OOD); recommended 0.1945 = 0.1189 pF / 0.6113 ns to map the axis extrema.")
    args = p.parse_args()

    categories = _parse_categories(args)
    build_constraint_datasets(
        cache_path=args.cache_path,
        cache_type=args.cache_type,
        lib_base_path=args.lib_base_path,
        output_dir=args.output_dir,
        categories=categories,
        skip_train=args.skip_train,
        include_parasitic_cap=args.include_parasitic_cap,
        voltage_mode=args.voltage_mode,
        temperature_mode=args.temperature_mode,
        include_zeros_in_norm=args.include_zeros_in_norm,
        topology_suffix=args.topology_suffix,
        slew_mode=args.slew_mode,
        rescale_constrained_to_load=args.rescale_constrained_to_load,
    )
