#!/usr/bin/env python
"""
Stage-Aware Graph Visualization for Paper

Visualizes the pull-up (rise) and pull-down (fall) current paths
for TSMC standard cells.

Usage:
    python visualize_stage_aware_graph.py --cell ND2D0BWP30P140
    python visualize_stage_aware_graph.py --cell XOR3D1BWP30P140 --output_dir ./figures
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import argparse
from pathlib import Path

# Set style for paper
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def load_topology_cache(cache_path):
    """Load pre-computed topology cache."""
    return torch.load(cache_path, weights_only=False)


def get_node_color(node_name, power_nodes, output_nodes, transistor_info):
    """Get node color based on type."""
    if node_name in power_nodes:
        if node_name == 'VDD':
            return '#FF6B6B'  # Red for VDD
        else:
            return '#4ECDC4'  # Cyan for VSS
    elif node_name in output_nodes:
        return '#FFE66D'  # Yellow for output
    elif node_name in transistor_info:
        trans_type = transistor_info[node_name]['type']
        if trans_type > 0:  # NMOS
            return '#95E1D3'  # Light green for NMOS
        else:  # PMOS
            return '#F38181'  # Light red for PMOS
    else:
        return '#DFE6E9'  # Gray for intermediate nodes


def get_node_shape(node_name, power_nodes, output_nodes, transistor_info):
    """Get node shape based on type."""
    if node_name in power_nodes:
        return 's'  # Square for power
    elif node_name in output_nodes:
        return 'd'  # Diamond for output
    elif node_name in transistor_info:
        return 'o'  # Circle for transistors
    else:
        return '^'  # Triangle for intermediate


def visualize_cell_paths(topology_cache, cell_name, output_dir='.', figsize=(16, 8)):
    """
    Visualize pull-up and pull-down paths for a cell.

    Args:
        topology_cache: Pre-computed topology cache
        cell_name: Cell name to visualize
        output_dir: Output directory for figures
        figsize: Figure size
    """
    if cell_name not in topology_cache:
        print(f"Cell {cell_name} not found in cache!")
        print(f"Available cells: {list(topology_cache.keys())[:10]}...")
        return

    cell_cache = topology_cache[cell_name]
    power_nodes = cell_cache['power_nodes']
    output_nodes = cell_cache['output_nodes']
    transistor_info = cell_cache['transistor_info']
    external_inputs = cell_cache['external_inputs']

    print(f"\n{'='*60}")
    print(f"Cell: {cell_name}")
    print(f"{'='*60}")
    print(f"External inputs: {external_inputs}")
    print(f"Output nodes: {output_nodes}")
    print(f"Power nodes: {power_nodes}")
    print(f"Transistors: {len(transistor_info)}")

    # Create figure with subplots for each output
    num_outputs = len(output_nodes)
    fig, axes = plt.subplots(num_outputs, 2, figsize=(figsize[0], figsize[1] * num_outputs))

    if num_outputs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Stage-Aware Current Paths: {cell_name}', fontsize=14, fontweight='bold')

    for out_idx, output_node in enumerate(output_nodes):
        output_topo = cell_cache['output_topologies'][output_node]

        for path_idx, (path_type, path_name, title) in enumerate([
            ('pull_up', 'rise', f'Pull-Up Path (Rise Transition)\nVDD → {output_node}'),
            ('pull_down', 'fall', f'Pull-Down Path (Fall Transition)\nVSS → {output_node}')
        ]):
            ax = axes[out_idx, path_idx]
            path_cache = output_topo[path_type]

            all_nodes = path_cache['all_nodes']
            edge_index = path_cache['edge_index']
            stage_info = path_cache.get('stage_info', {})
            num_stages = stage_info.get('num_stages', 1)

            print(f"\n  {output_node} - {path_type}:")
            print(f"    Nodes: {len(all_nodes)}")
            print(f"    Edges: {edge_index.shape[1] if edge_index.numel() > 0 else 0}")
            print(f"    Stages: {num_stages}")
            print(f"    Node list: {all_nodes}")

            # Create NetworkX graph
            G = nx.DiGraph()

            # Add nodes
            for node in all_nodes:
                G.add_node(node)

            # Add edges
            if edge_index.numel() > 0:
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0][i].item()
                    dst_idx = edge_index[1][i].item()
                    src = all_nodes[src_idx]
                    dst = all_nodes[dst_idx]
                    G.add_edge(src, dst)

            # Layout - hierarchical for current flow visualization
            if len(all_nodes) > 0:
                # Custom hierarchical layout
                pos = {}

                # Identify node layers
                power_node = 'VDD' if path_type == 'pull_up' else 'VSS'

                # Layer 0: Power node
                # Layer 1-N: Transistors by stage
                # Layer N+1: Output

                stages = stage_info.get('stages', [])

                # Assign y positions based on node type
                y_positions = {}
                y_positions[power_node] = 1.0  # Top for VDD, bottom for VSS
                y_positions[output_node] = 0.0 if path_type == 'pull_up' else 1.0

                # Transistors in between
                transistor_nodes_in_path = [n for n in all_nodes if n in transistor_info]
                intermediate_nodes = [n for n in all_nodes
                                     if n not in power_nodes
                                     and n not in output_nodes
                                     and n not in transistor_info]

                # Assign positions
                num_trans = len(transistor_nodes_in_path)
                num_inter = len(intermediate_nodes)

                if path_type == 'pull_up':
                    # VDD at top, output at bottom
                    for i, node in enumerate(transistor_nodes_in_path):
                        y_positions[node] = 0.7 - (i * 0.4 / max(num_trans, 1))
                    for i, node in enumerate(intermediate_nodes):
                        y_positions[node] = 0.3 - (i * 0.2 / max(num_inter, 1))
                else:
                    # VSS at bottom, output at top
                    for i, node in enumerate(transistor_nodes_in_path):
                        y_positions[node] = 0.3 + (i * 0.4 / max(num_trans, 1))
                    for i, node in enumerate(intermediate_nodes):
                        y_positions[node] = 0.7 + (i * 0.2 / max(num_inter, 1))

                # Assign x positions (spread horizontally)
                x_counter = {}
                for node in all_nodes:
                    y = y_positions.get(node, 0.5)
                    y_key = round(y, 2)
                    if y_key not in x_counter:
                        x_counter[y_key] = []
                    x_counter[y_key].append(node)

                for y_key, nodes in x_counter.items():
                    n = len(nodes)
                    for i, node in enumerate(nodes):
                        x = (i - (n - 1) / 2) * 0.3
                        pos[node] = (x, y_positions.get(node, 0.5))

                # Draw graph
                # Node colors
                node_colors = [get_node_color(n, power_nodes, output_nodes, transistor_info)
                              for n in G.nodes()]

                # Node sizes
                node_sizes = []
                for n in G.nodes():
                    if n in power_nodes:
                        node_sizes.append(800)
                    elif n in output_nodes:
                        node_sizes.append(700)
                    elif n in transistor_info:
                        node_sizes.append(600)
                    else:
                        node_sizes.append(400)

                # Draw
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                      node_size=node_sizes, alpha=0.9, edgecolors='black', linewidths=1.5)
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#2D3436',
                                      arrows=True, arrowsize=15, width=2,
                                      connectionstyle='arc3,rad=0.1', alpha=0.7)

                # Labels
                labels = {}
                for node in G.nodes():
                    if node in transistor_info:
                        trans_type = 'N' if transistor_info[node]['type'] > 0 else 'P'
                        width = transistor_info[node]['width']
                        # Shorten transistor name
                        short_name = node.replace('XM', 'M')
                        labels[node] = f"{short_name}\n({trans_type})"
                    else:
                        labels[node] = node

                nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

            # Add stage info text
            stage_text = f"Stages: {num_stages}"
            if stages:
                stage_details = []
                for s in stages:
                    s_type = 'PMOS' if 'pmos' in s.get('mos_type', '').lower() else 'NMOS'
                    s_trans = len(s.get('transistors', []))
                    stage_details.append(f"S{s['stage_num']}: {s_type} ({s_trans}T)")
                stage_text += "\n" + ", ".join(stage_details)

            ax.text(0.02, 0.98, stage_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='VDD'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='VSS'),
        mpatches.Patch(facecolor='#FFE66D', edgecolor='black', label='Output'),
        mpatches.Patch(facecolor='#F38181', edgecolor='black', label='PMOS'),
        mpatches.Patch(facecolor='#95E1D3', edgecolor='black', label='NMOS'),
        mpatches.Patch(facecolor='#DFE6E9', edgecolor='black', label='Intermediate'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
              fontsize=9, frameon=True, fancybox=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save figure
    output_path = Path(output_dir) / f'stage_aware_graph_{cell_name}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")

    # Also save PDF for paper
    pdf_path = Path(output_dir) / f'stage_aware_graph_{cell_name}.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {pdf_path}")

    plt.close()

    return fig


def visualize_comparison(topology_cache, cell_names, output_dir='.'):
    """
    Create a comparison figure showing multiple cells side by side.

    Args:
        topology_cache: Pre-computed topology cache
        cell_names: List of cell names to compare
        output_dir: Output directory
    """
    n_cells = len(cell_names)
    fig, axes = plt.subplots(n_cells, 2, figsize=(14, 5 * n_cells))

    if n_cells == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Stage-Aware Graph Comparison: Simple vs Complex Cells',
                fontsize=14, fontweight='bold')

    for cell_idx, cell_name in enumerate(cell_names):
        if cell_name not in topology_cache:
            print(f"Skipping {cell_name} - not in cache")
            continue

        cell_cache = topology_cache[cell_name]
        power_nodes = cell_cache['power_nodes']
        output_nodes = cell_cache['output_nodes']
        transistor_info = cell_cache['transistor_info']

        # Use first output
        output_node = output_nodes[0]
        output_topo = cell_cache['output_topologies'][output_node]

        for path_idx, (path_type, arrow_dir) in enumerate([('pull_up', '↓'), ('pull_down', '↑')]):
            ax = axes[cell_idx, path_idx]
            path_cache = output_topo[path_type]

            all_nodes = path_cache['all_nodes']
            edge_index = path_cache['edge_index']
            stage_info = path_cache.get('stage_info', {})
            num_stages = stage_info.get('num_stages', 1)

            # Create graph
            G = nx.DiGraph()
            for node in all_nodes:
                G.add_node(node)

            if edge_index.numel() > 0:
                for i in range(edge_index.shape[1]):
                    src = all_nodes[edge_index[0][i].item()]
                    dst = all_nodes[edge_index[1][i].item()]
                    G.add_edge(src, dst)

            # Spring layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Colors
            node_colors = [get_node_color(n, power_nodes, output_nodes, transistor_info)
                          for n in G.nodes()]

            # Draw
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                  node_size=500, alpha=0.9, edgecolors='black')
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#2D3436',
                                  arrows=True, arrowsize=12, width=1.5, alpha=0.7)

            # Simplified labels
            labels = {n: n.replace('XM', 'M').replace('BWP30P140', '') for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

            path_name = 'Pull-Up (Rise)' if path_type == 'pull_up' else 'Pull-Down (Fall)'
            ax.set_title(f'{cell_name}\n{path_name} - {num_stages} stage(s)', fontsize=10)
            ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='VDD'),
        mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='VSS'),
        mpatches.Patch(facecolor='#FFE66D', edgecolor='black', label='Output'),
        mpatches.Patch(facecolor='#F38181', edgecolor='black', label='PMOS'),
        mpatches.Patch(facecolor='#95E1D3', edgecolor='black', label='NMOS'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    output_path = Path(output_dir) / 'stage_aware_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")

    pdf_path = Path(output_dir) / 'stage_aware_comparison.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage-Aware Graph for Paper')
    parser.add_argument('--cache_path', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/stage_aware_topology_cache_tsmc_tcbn28hpcplusbwp30p140_110a_lpe_typical.pth',
                       help='Path to topology cache')
    parser.add_argument('--cell', type=str, default=None,
                       help='Cell name to visualize (e.g., ND2D0BWP30P140)')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='List of cells to compare (e.g., ND2D0BWP30P140 XOR3D1BWP30P140)')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--list_cells', action='store_true',
                       help='List available cells in cache')

    args = parser.parse_args()

    # Load cache
    print(f"Loading topology cache: {args.cache_path}")
    topology_cache = load_topology_cache(args.cache_path)
    print(f"Loaded {len(topology_cache)} cells")

    if args.list_cells:
        print("\nAvailable cells:")
        for cell in sorted(topology_cache.keys()):
            cell_cache = topology_cache[cell]
            n_trans = len(cell_cache.get('transistor_info', {}))
            n_outputs = len(cell_cache.get('output_nodes', []))
            print(f"  {cell}: {n_trans} transistors, {n_outputs} outputs")
        return

    if args.compare:
        visualize_comparison(topology_cache, args.compare, args.output_dir)
    elif args.cell:
        visualize_cell_paths(topology_cache, args.cell, args.output_dir)
    else:
        # Default: show a few example cells
        example_cells = ['ND2D0BWP30P140', 'NR2D0BWP30P140', 'AN2D0BWP30P140',
                        'XOR2D1BWP30P140', 'XNR2D1BWP30P140']

        # Find available cells
        available = [c for c in example_cells if c in topology_cache]

        if not available:
            # Try to find any NAND, NOR, XOR cells
            for cell in topology_cache.keys():
                if 'ND2' in cell or 'NR2' in cell:
                    available.append(cell)
                    if len(available) >= 2:
                        break
            for cell in topology_cache.keys():
                if 'XOR' in cell or 'XNR' in cell:
                    available.append(cell)
                    if len(available) >= 4:
                        break

        if available:
            print(f"\nVisualizing example cells: {available[:4]}")
            for cell in available[:4]:
                visualize_cell_paths(topology_cache, cell, args.output_dir)

            # Also create comparison
            if len(available) >= 2:
                visualize_comparison(topology_cache, available[:4], args.output_dir)
        else:
            print("No suitable cells found. Use --list_cells to see available cells.")


if __name__ == '__main__':
    main()
