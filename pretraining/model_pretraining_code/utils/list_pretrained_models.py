#!/usr/bin/env python3
"""
List and analyze pretrained model weights in taskdivide_all directory

This utility scans the pretrained models directory and extracts parameters
from model filenames, providing a structured overview of available models.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd


def parse_model_filename(filename: str) -> Optional[Dict]:
    """
    Parse model filename and extract parameters

    Supports multiple filename formats:
    1. New format: {data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_({layer_length})_{iterations}_inner{inner}_upgraded{tech_suffix}.pth
    2. Old format with meta: {data_type}_innerdiv{innerdiv}_meta{meta}_full1DMAML_weights_3hidden_({layer_length})_{iterations}_{tech}_{corner}_{voltage}_test5(dim5)_inner{inner}.pth
    3. Simple format: {data_type}_innerdiv{innerdiv}_full1DMAML_weights_3hidden_({layer_length})_{iterations}_{tech}_{corner}_test5(dim5)_inner{inner}_fixed.pth
    4. Legacy format: {data_type}_full1DMAML_weights_3hidden_({layer_length})_{iterations}_{tech}_{corner}_test5(dim5)_inner{inner}_fixed.pth

    Returns:
        dict with extracted parameters or None if parsing fails
    """

    # Remove .pth extension
    name = filename.replace('.pth', '')

    # Initialize result dictionary
    result = {
        'filename': filename,
        'data_type': None,
        'innerdiv': None,
        'meta': None,
        'model_suffix': None,
        'layer_length': None,
        'iterations': None,
        'inner': None,
        'tech': None,
        'corner': None,
        'voltage': None,
        'upgraded': False,
        'format': 'unknown'
    }

    # Try to extract data_type (cell or transition)
    if name.startswith('cell_'):
        result['data_type'] = 'cell'
    elif name.startswith('transition_'):
        result['data_type'] = 'transition'
    else:
        return None

    # Extract innerdiv
    innerdiv_match = re.search(r'innerdiv(\d+)', name)
    if innerdiv_match:
        result['innerdiv'] = int(innerdiv_match.group(1))

    # Extract meta
    meta_match = re.search(r'meta(\d+)', name)
    if meta_match:
        result['meta'] = int(meta_match.group(1))

    # Extract layer_length
    layer_match = re.search(r'3hidden_\((\d+)\)', name)
    if layer_match:
        result['layer_length'] = int(layer_match.group(1))

    # Extract iterations (the number after layer_length)
    # Pattern: 3hidden_(layer_length)_(iterations)_
    iter_match = re.search(r'3hidden_\(\d+\)_(\d+)_', name)
    if iter_match:
        result['iterations'] = int(iter_match.group(1))
    else:
        # Fallback: try to find number before _inner
        iter_match2 = re.search(r'_(\d+)_inner', name)
        if iter_match2:
            result['iterations'] = int(iter_match2.group(1))

    # Extract inner
    inner_match = re.search(r'inner(\d+)', name)
    if inner_match:
        result['inner'] = int(inner_match.group(1))

    # Check for upgraded flag
    if 'upgraded' in name:
        result['upgraded'] = True
        # Extract tech suffix if present
        tech_suffix_match = re.search(r'upgraded_(\w+)', name)
        if tech_suffix_match:
            result['tech'] = tech_suffix_match.group(1).upper()

    # Check for 519traintask (new format)
    if '519traintask' in name:
        result['format'] = 'new_519traintask'
        # Extract model_suffix (topology_agnostic or intratopology)
        if 'topology_agnostic' in name:
            result['model_suffix'] = 'topology_agnostic'
        elif 'intratopology' in name:
            result['model_suffix'] = 'intratopology'
    else:
        result['format'] = 'legacy'
        # Extract tech, corner, voltage from legacy format
        # Pattern: {TECH}_{CORNER}_{VOLTAGE}_test5 or {TECH}_{CORNER}_test5
        tech_pattern = re.search(r'_(ASAP7|TSMC|LVT|RVT|SLVT|SRAM)_(FF|SS|TT)(?:_(\d+))?_test5', name)
        if tech_pattern:
            result['tech'] = tech_pattern.group(1)
            result['corner'] = tech_pattern.group(2)
            if tech_pattern.group(3):
                result['voltage'] = int(tech_pattern.group(3))

    return result


def scan_pretrained_models(models_dir: str, verbose: bool = False) -> List[Dict]:
    """
    Scan pretrained models directory and parse all .pth files

    Args:
        models_dir: Path to pretrained models directory
        verbose: Print parsing details

    Returns:
        List of dictionaries with parsed model information
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"❌ Error: Directory not found: {models_dir}")
        return []

    pth_files = list(models_path.glob('*.pth'))

    if not pth_files:
        print(f"⚠️ Warning: No .pth files found in {models_dir}")
        return []

    print(f"📂 Scanning directory: {models_dir}")
    print(f"📊 Found {len(pth_files)} .pth files\n")

    parsed_models = []
    unparsed_count = 0

    for pth_file in pth_files:
        result = parse_model_filename(pth_file.name)
        if result:
            parsed_models.append(result)
            if verbose:
                print(f"✓ {pth_file.name}")
        else:
            unparsed_count += 1
            if verbose:
                print(f"✗ {pth_file.name} (could not parse)")

    print(f"\n✅ Successfully parsed: {len(parsed_models)} models")
    if unparsed_count > 0:
        print(f"⚠️ Could not parse: {unparsed_count} files")

    return parsed_models


def analyze_models(models: List[Dict]) -> None:
    """
    Analyze and print statistics about parsed models

    Args:
        models: List of parsed model dictionaries
    """
    if not models:
        print("No models to analyze")
        return

    print("\n" + "="*80)
    print("📊 Model Analysis")
    print("="*80)

    # Group by data_type
    by_data_type = defaultdict(list)
    for model in models:
        by_data_type[model['data_type']].append(model)

    print(f"\n📦 By Data Type:")
    for data_type, models_list in sorted(by_data_type.items()):
        print(f"  {data_type}: {len(models_list)} models")

    # Group by format
    by_format = defaultdict(list)
    for model in models:
        by_format[model['format']].append(model)

    print(f"\n🔧 By Format:")
    for fmt, models_list in sorted(by_format.items()):
        print(f"  {fmt}: {len(models_list)} models")

    # Group by iterations
    by_iterations = defaultdict(list)
    for model in models:
        if model['iterations']:
            by_iterations[model['iterations']].append(model)

    print(f"\n🔄 By Iterations:")
    for iterations in sorted(by_iterations.keys()):
        print(f"  {iterations}: {len(by_iterations[iterations])} models")

    # Group by inner
    by_inner = defaultdict(list)
    for model in models:
        if model['inner']:
            by_inner[model['inner']].append(model)

    print(f"\n🔁 By Inner Steps:")
    for inner in sorted(by_inner.keys()):
        print(f"  inner={inner}: {len(by_inner[inner])} models")

    # Group by innerdiv
    by_innerdiv = defaultdict(list)
    for model in models:
        if model['innerdiv']:
            by_innerdiv[model['innerdiv']].append(model)

    print(f"\n📐 By Inner Divisor:")
    for innerdiv in sorted(by_innerdiv.keys()):
        print(f"  innerdiv={innerdiv}: {len(by_innerdiv[innerdiv])} models")

    # Group by meta
    by_meta = defaultdict(list)
    for model in models:
        if model['meta']:
            by_meta[model['meta']].append(model)

    if by_meta:
        print(f"\n🎯 By Meta Batch Size:")
        for meta in sorted(by_meta.keys()):
            print(f"  meta={meta}: {len(by_meta[meta])} models")

    # Group by model_suffix (topology type)
    by_suffix = defaultdict(list)
    for model in models:
        if model['model_suffix']:
            by_suffix[model['model_suffix']].append(model)

    if by_suffix:
        print(f"\n🏗️ By Model Type:")
        for suffix, models_list in sorted(by_suffix.items()):
            print(f"  {suffix}: {len(models_list)} models")

    # Count upgraded models
    upgraded_count = sum(1 for m in models if m['upgraded'])
    print(f"\n⬆️ Upgraded Models: {upgraded_count}")

    # Group by technology
    by_tech = defaultdict(list)
    for model in models:
        if model['tech']:
            by_tech[model['tech']].append(model)

    if by_tech:
        print(f"\n🔬 By Technology:")
        for tech, models_list in sorted(by_tech.items()):
            print(f"  {tech}: {len(models_list)} models")


def filter_models(models: List[Dict],
                  data_type: Optional[str] = None,
                  iterations: Optional[int] = None,
                  inner: Optional[int] = None,
                  innerdiv: Optional[int] = None,
                  meta: Optional[int] = None,
                  model_suffix: Optional[str] = None,
                  layer_length: Optional[int] = None,
                  tech_suffix: Optional[str] = None,
                  upgraded: Optional[bool] = None) -> List[Dict]:
    """
    Filter models based on criteria

    Args:
        models: List of parsed model dictionaries
        data_type: Filter by data type (cell/transition)
        iterations: Filter by iteration count
        inner: Filter by inner steps
        innerdiv: Filter by inner divisor
        meta: Filter by meta batch size
        model_suffix: Filter by model type (topology_agnostic/intratopology)
        layer_length: Filter by layer length
        tech_suffix: Filter by technology suffix (ASAP7/TSMC)
        upgraded: Filter by upgraded flag

    Returns:
        Filtered list of models
    """
    filtered = models

    if data_type:
        filtered = [m for m in filtered if m['data_type'] == data_type]

    if iterations is not None:
        filtered = [m for m in filtered if m['iterations'] == iterations]

    if inner is not None:
        filtered = [m for m in filtered if m['inner'] == inner]

    if innerdiv is not None:
        filtered = [m for m in filtered if m['innerdiv'] == innerdiv]

    if meta is not None:
        filtered = [m for m in filtered if m['meta'] == meta]

    if model_suffix:
        filtered = [m for m in filtered if m['model_suffix'] == model_suffix]

    if layer_length is not None:
        filtered = [m for m in filtered if m['layer_length'] == layer_length]

    if tech_suffix:
        filtered = [m for m in filtered if m['tech'] and m['tech'].upper() == tech_suffix.upper()]

    if upgraded is not None:
        filtered = [m for m in filtered if m['upgraded'] == upgraded]

    return filtered


def print_model_paths(models: List[Dict], models_dir: str, limit: Optional[int] = None) -> None:
    """
    Print model paths only (simple output format)

    Args:
        models: List of parsed model dictionaries
        models_dir: Path to models directory
        limit: Maximum number of models to print (None for all)
    """
    if not models:
        print("No models found matching the given conditions")
        return

    # Select models to display
    display_models = models[:limit] if limit else models

    print(f"\n📋 Found {len(models)} model(s) matching conditions:\n")

    for model in display_models:
        full_path = os.path.join(models_dir, model['filename'])
        print(full_path)

    if limit and len(models) > limit:
        print(f"\n... and {len(models) - limit} more models (use --limit to show more)")


def print_models_table(models: List[Dict], limit: Optional[int] = None) -> None:
    """
    Print models in a formatted table

    Args:
        models: List of parsed model dictionaries
        limit: Maximum number of models to print (None for all)
    """
    if not models:
        print("No models to display")
        return

    # Select models to display
    display_models = models[:limit] if limit else models

    # Print header
    print("\n" + "="*140)
    print(f"{'Filename':<80} {'Type':<6} {'Iter':<7} {'Inner':<6} {'InDiv':<6} {'Meta':<5} {'Upg':<4}")
    print("="*140)

    # Print models
    for model in display_models:
        filename = model['filename']
        if len(filename) > 75:
            filename = filename[:72] + "..."

        data_type = model['data_type'] or '-'
        iterations = str(model['iterations']) if model['iterations'] else '-'
        inner = str(model['inner']) if model['inner'] else '-'
        innerdiv = str(model['innerdiv']) if model['innerdiv'] else '-'
        meta = str(model['meta']) if model['meta'] else '-'
        upgraded = 'Yes' if model['upgraded'] else 'No'

        print(f"{filename:<80} {data_type:<6} {iterations:<7} {inner:<6} {innerdiv:<6} {meta:<5} {upgraded:<4}")

    if limit and len(models) > limit:
        print(f"\n... and {len(models) - limit} more models (use --limit to show more)")


def export_to_csv(models: List[Dict], output_file: str) -> None:
    """
    Export models to CSV file

    Args:
        models: List of parsed model dictionaries
        output_file: Path to output CSV file
    """
    if not models:
        print("No models to export")
        return

    # Convert to pandas DataFrame
    df = pd.DataFrame(models)

    # Reorder columns
    column_order = ['filename', 'data_type', 'iterations', 'inner', 'innerdiv',
                   'meta', 'model_suffix', 'layer_length', 'upgraded', 'tech',
                   'corner', 'voltage', 'format']

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n💾 Exported {len(models)} models to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='List and analyze pretrained model weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python list_pretrained_models.py

  # Filter by data type and iterations (show paths only)
  python list_pretrained_models.py --data_type cell --iterations 300000 --paths-only

  # Filter by new format parameters
  python list_pretrained_models.py --data_type cell --innerdiv 1 --meta 8 \\
      --model_suffix topology_agnostic --layer_length 300 \\
      --iterations 300000 --inner 5 --tech_suffix ASAP7 --upgraded --paths-only

  # Filter by topology and show paths
  python list_pretrained_models.py --model_suffix topology_agnostic --upgraded --paths-only

  # Export to CSV
  python list_pretrained_models.py --export models.csv

  # Show only first 20 models
  python list_pretrained_models.py --limit 20
        """
    )

    # Auto-detect pretrained models directory
    script_dir = Path(__file__).resolve().parent.parent
    default_models_dir = script_dir.parent.parent / 'pretrained_models' / 'taskdivide_all'

    parser.add_argument('--models_dir', type=str,
                       default=str(default_models_dir),
                       help='Path to pretrained models directory')
    parser.add_argument('--data_type', type=str, choices=['cell', 'transition'],
                       help='Filter by data type')
    parser.add_argument('--iterations', type=int,
                       help='Filter by iteration count')
    parser.add_argument('--inner', type=int,
                       help='Filter by inner steps')
    parser.add_argument('--innerdiv', type=int,
                       help='Filter by inner divisor')
    parser.add_argument('--meta', type=int,
                       help='Filter by meta batch size')
    parser.add_argument('--model_suffix', type=str,
                       choices=['topology_agnostic', 'intratopology'],
                       help='Filter by model type (topology_agnostic or intratopology)')
    parser.add_argument('--layer_length', type=int,
                       help='Filter by layer length')
    parser.add_argument('--tech_suffix', type=str,
                       help='Filter by technology suffix (e.g., ASAP7, TSMC)')
    parser.add_argument('--upgraded', action='store_true',
                       help='Filter only upgraded models')
    parser.add_argument('--paths-only', action='store_true',
                       help='Print only model paths (no analysis or table)')
    parser.add_argument('--export', type=str, metavar='FILE',
                       help='Export to CSV file')
    parser.add_argument('--limit', type=int,
                       help='Limit number of models to display')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed parsing information')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip analysis section')

    args = parser.parse_args()

    # Scan models
    models = scan_pretrained_models(args.models_dir, verbose=args.verbose)

    if not models:
        return

    # Apply filters
    filtered_models = filter_models(
        models,
        data_type=args.data_type,
        iterations=args.iterations,
        inner=args.inner,
        innerdiv=args.innerdiv,
        meta=args.meta,
        model_suffix=args.model_suffix,
        layer_length=args.layer_length,
        tech_suffix=args.tech_suffix,
        upgraded=args.upgraded if args.upgraded else None
    )

    # Check if any filters were applied
    filters_applied = any([
        args.data_type, args.iterations, args.inner, args.innerdiv,
        args.meta, args.model_suffix, args.layer_length, args.tech_suffix, args.upgraded
    ])

    if filters_applied:
        print(f"\n🔍 Filtered to {len(filtered_models)} models")

    # If paths-only mode, just print paths and exit
    if args.paths_only:
        print_model_paths(filtered_models, args.models_dir, limit=args.limit)
        if args.export:
            export_to_csv(filtered_models, args.export)
        return

    # Print analysis
    if not args.no_analysis:
        analyze_models(filtered_models)

    # Print models table
    print_models_table(filtered_models, limit=args.limit)

    # Export to CSV if requested
    if args.export:
        export_to_csv(filtered_models, args.export)


if __name__ == "__main__":
    main()
