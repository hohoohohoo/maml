"""Re-extract topology cache for the 3 DFF cells with loop-closing enabled,
combine with the existing combinational-cell entries from the baseline cache,
and write to a new `_loopclose.pth` cache file.

Usage:
    python3 regen_dff_cache_loopclose.py \
        --base_cache topology_cache/stage_aware_topology_cache_tsmc_tcbn28hpcplusbwp30p140_110a_lpe_typical.pth \
        --spi_path  /home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/tcbn28hpcplusbwp30p140_110a_lpe_typical.spi \
        --output    topology_cache/stage_aware_topology_cache_tsmc_tcbn28hpcplusbwp30p140_110a_lpe_typical_loopclose.pth
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))

from precompute_stage_aware_topology import precompute_stage_aware_topology_tsmc


DFF_CELLS = ['DFCNQD1BWP30P140', 'SDFSNQD0BWP30P140', 'SDFCSNQD1BWP30P140']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_cache', required=True, help='existing baseline cache to start from')
    ap.add_argument('--spi_path', required=True)
    ap.add_argument('--output', required=True, help='destination cache .pth')
    ap.add_argument('--weighted', action='store_true')
    ap.add_argument('--gate_control', type=float, default=0.0,
                    help='If > 0, add gate-control edges intermediate_gate -> transistor '
                         '(typical 0.5). Required to make master/slave internal '
                         'cross-coupling form a true SCC.')
    args = ap.parse_args()

    print('=' * 80)
    print('DFF loop-closing cache regeneration')
    print('=' * 80)
    print(f'  base cache : {args.base_cache}')
    print(f'  output     : {args.output}')
    print(f'  spi_path   : {args.spi_path}')
    print(f'  DFF cells  : {DFF_CELLS}')

    # 1) Load baseline cache (must already exist).  Verifies that all
    #    combinational cells are present in this cache file unchanged.
    base = torch.load(args.base_cache, map_location='cpu', weights_only=False)
    print(f'\n  base cache contains {len(base)} cells: '
          f'{sum(1 for c in base if c in DFF_CELLS)}/3 DFFs, rest combinational')

    # 2) Run extraction limited to DFF cells WITH loop_closing=True into a temp
    #    cache file, then merge.
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        precompute_stage_aware_topology_tsmc(
            args.spi_path, tmp_path,
            weighted=args.weighted,
            gate_control_weight=args.gate_control,
            loop_closing=True,
            only_cells=set(DFF_CELLS),
        )
        new = torch.load(tmp_path, map_location='cpu', weights_only=False)
        print(f'\n  loop-closing extraction produced {len(new)} cell entries')
        for c in DFF_CELLS:
            if c in new:
                topo = new[c].get('output_topologies', {})
                for out_node, t in topo.items():
                    pu = t.get('pull_up', {})
                    pd = t.get('pull_down', {})
                    pu_stages = pu.get('stage_info', {}).get('num_stages', '?')
                    pd_stages = pd.get('stage_info', {}).get('num_stages', '?')
                    print(f'    {c}.{out_node}: pull_up stages={pu_stages}, '
                          f'pull_down stages={pd_stages}')
            else:
                print(f'    WARN: {c} not produced')
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 3) Merge: take baseline entries and OVERWRITE the 3 DFFs with the new
    #    loop-closing variants.  All combinational cells stay exactly as in baseline.
    merged = dict(base)
    for c in DFF_CELLS:
        if c in new:
            merged[c] = new[c]
        else:
            print(f'   WARN: could not regenerate {c}; keeping baseline entry')

    # 4) Persist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(merged, args.output)
    print(f'\n  wrote merged cache -> {args.output}')
    print(f'  total cells in merged cache: {len(merged)}')


if __name__ == '__main__':
    main()
