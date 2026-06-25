#!/usr/bin/env python
"""
Test script to compare original vs v2 (with equivalence collapsing) stage-aware extractor.

Tests:
1. Simple cells (ND2, INV) - should produce same results
2. Complex cells (ND4, NR4) - v2 should be much faster
"""

import time
import sys
sys.path.append('/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn')

# Import both versions
from past_codes.stage_aware_extractor_tsmc_original import TSMCStageAwareExtractor as OriginalExtractor
from stage_aware_extractor_tsmc import TSMCStageAwareExtractor as V2Extractor

SPI_PATH = "/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/tcbn28hpcplusbwp30p140_110a_lpe_typical.spi"

def test_cell(cell_name, external_inputs, output_nodes, max_time=30):
    """Test a single cell with both extractors."""
    print(f"\n{'='*60}")
    print(f"Testing: {cell_name}")
    print(f"Inputs: {external_inputs}, Outputs: {output_nodes}")
    print(f"{'='*60}")

    # Test Original
    print(f"\n--- Original Extractor ---")
    original = OriginalExtractor(SPI_PATH)

    start = time.time()
    try:
        result_orig = original.classify_multi_stage_structure(
            cell_name, external_inputs, 'rise_transition', output_nodes
        )
        orig_time = time.time() - start
        print(f"Time: {orig_time:.3f}s")
        print(f"Stages: {result_orig.num_stages}")
        for stage in result_orig.stages:
            print(f"  Stage {stage.stage_num}: {len(stage.transistors)} transistors, {len(stage.paths)} paths")
    except Exception as e:
        orig_time = time.time() - start
        print(f"Error after {orig_time:.3f}s: {e}")
        result_orig = None

    # Test V2
    print(f"\n--- V2 Extractor (with equivalence collapsing) ---")
    v2 = V2Extractor(SPI_PATH)

    start = time.time()
    try:
        result_v2 = v2.classify_multi_stage_structure(
            cell_name, external_inputs, 'rise_transition', output_nodes
        )
        v2_time = time.time() - start
        print(f"Time: {v2_time:.3f}s")
        print(f"Stages: {result_v2.num_stages}")
        for stage in result_v2.stages:
            print(f"  Stage {stage.stage_num}: {len(stage.transistors)} transistors, {len(stage.paths)} paths")
    except Exception as e:
        v2_time = time.time() - start
        print(f"Error after {v2_time:.3f}s: {e}")
        result_v2 = None

    # Compare
    print(f"\n--- Comparison ---")
    if result_orig and result_v2:
        if result_orig.num_stages == result_v2.num_stages:
            print(f"✓ Same number of stages: {result_orig.num_stages}")
        else:
            print(f"✗ Different stages: orig={result_orig.num_stages}, v2={result_v2.num_stages}")

        # Compare transistors per stage
        for i, (s1, s2) in enumerate(zip(result_orig.stages, result_v2.stages)):
            trans_orig = set(s1.transistors)
            trans_v2 = set(s2.transistors)
            if trans_orig == trans_v2:
                print(f"✓ Stage {i+1} transistors match: {len(trans_orig)}")
            else:
                print(f"✗ Stage {i+1} transistors differ:")
                print(f"    Only in orig: {trans_orig - trans_v2}")
                print(f"    Only in v2: {trans_v2 - trans_orig}")

        speedup = orig_time / v2_time if v2_time > 0 else float('inf')
        print(f"\nSpeedup: {speedup:.1f}x ({orig_time:.3f}s -> {v2_time:.3f}s)")
    elif result_v2 and not result_orig:
        print(f"✓ V2 succeeded where original failed/timed out")
    else:
        print(f"Both failed or original succeeded but v2 failed")

    return result_orig, result_v2


def main():
    print("="*60)
    print("Stage-Aware Extractor: Original vs V2 Comparison")
    print("="*60)

    # Test cases
    test_cases = [
        # Simple cells (should produce same results)
        ("INVD0BWP30P140", ["I"], ["ZN"]),
        ("ND2D0BWP30P140", ["A1", "A2"], ["ZN"]),
        ("NR2D0BWP30P140", ["A1", "A2"], ["ZN"]),

        # 3-input gates
        ("ND3D0BWP30P140", ["A1", "A2", "A3"], ["ZN"]),
        ("NR3D0BWP30P140", ["A1", "A2", "A3"], ["ZN"]),

        # 4-input gates (problematic with original)
        ("ND4D0BWP30P140", ["A1", "A2", "A3", "A4"], ["ZN"]),
        ("NR4D0BWP30P140", ["A1", "A2", "A3", "A4"], ["ZN"]),
    ]

    results = []
    for cell_name, inputs, outputs in test_cases:
        try:
            result = test_cell(cell_name, inputs, outputs)
            results.append((cell_name, result))
        except KeyboardInterrupt:
            print(f"\nSkipping {cell_name} (interrupted)")
            continue
        except Exception as e:
            print(f"\nError testing {cell_name}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cell_name, (orig, v2) in results:
        status = "✓" if (orig and v2) else ("V2 only" if v2 else "Failed")
        print(f"  {cell_name}: {status}")


if __name__ == "__main__":
    main()
