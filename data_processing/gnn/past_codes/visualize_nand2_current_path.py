#!/usr/bin/env python
"""
2-Input NAND Current Path Visualization for Paper

Creates a comprehensive figure showing:
1. NAND2 circuit schematic
2. Pull-up path (rise transition) graph
3. Pull-down path (fall transition) graph
4. Node feature encoding table

Usage:
    python visualize_nand2_current_path.py --output_dir ./figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.lines as mlines
import numpy as np
import argparse
from pathlib import Path

# Paper-quality settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'


def draw_transistor(ax, x, y, trans_type='nmos', width=0.3, label='', gate_label=''):
    """Draw a MOSFET symbol."""
    h = 0.4  # height
    w = width

    if trans_type == 'nmos':
        color = '#95E1D3'  # Light green
        # Draw NMOS symbol
        # Channel
        ax.plot([x-w/2, x+w/2], [y, y], 'k-', linewidth=2)
        # Gate
        ax.plot([x-w/2-0.1, x-w/2-0.1], [y-h/3, y+h/3], 'k-', linewidth=2)
        ax.plot([x-w/2-0.2, x-w/2-0.1], [y, y], 'k-', linewidth=1.5)
        # Source/Drain lines
        ax.plot([x, x], [y-h/2, y], 'k-', linewidth=1.5)
        ax.plot([x, x], [y, y+h/2], 'k-', linewidth=1.5)
        # Arrow (pointing in for NMOS)
        ax.annotate('', xy=(x-w/2+0.05, y), xytext=(x-w/2-0.05, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))
    else:  # pmos
        color = '#F38181'  # Light red
        # Draw PMOS symbol
        # Channel
        ax.plot([x-w/2, x+w/2], [y, y], 'k-', linewidth=2)
        # Gate with bubble
        ax.plot([x-w/2-0.1, x-w/2-0.1], [y-h/3, y+h/3], 'k-', linewidth=2)
        ax.plot([x-w/2-0.25, x-w/2-0.15], [y, y], 'k-', linewidth=1.5)
        circle = Circle((x-w/2-0.125, y), 0.03, fill=False, color='black', linewidth=1.5)
        ax.add_patch(circle)
        # Source/Drain lines
        ax.plot([x, x], [y-h/2, y], 'k-', linewidth=1.5)
        ax.plot([x, x], [y, y+h/2], 'k-', linewidth=1.5)

    # Label
    if label:
        ax.text(x+w/2+0.1, y, label, fontsize=9, va='center', fontweight='bold', color=color)
    if gate_label:
        ax.text(x-w/2-0.35, y, gate_label, fontsize=9, va='center', ha='right')

    # Background box
    rect = FancyBboxPatch((x-w/2-0.05, y-h/2), w+0.1, h,
                          boxstyle="round,pad=0.02", facecolor=color, alpha=0.3,
                          edgecolor='none')
    ax.add_patch(rect)

    return (x, y-h/2), (x, y+h/2)  # Return drain, source positions


def draw_nand2_schematic(ax):
    """Draw NAND2 circuit schematic."""
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) NAND2 Circuit Schematic', fontsize=11, fontweight='bold', pad=10)

    # VDD rail
    ax.plot([-1, 2], [4, 4], 'r-', linewidth=3)
    ax.text(0.5, 4.2, 'VDD', fontsize=10, ha='center', fontweight='bold', color='#FF6B6B')

    # VSS rail
    ax.plot([-1, 2], [0, 0], 'b-', linewidth=3)
    ax.text(0.5, -0.3, 'VSS', fontsize=10, ha='center', fontweight='bold', color='#4ECDC4')

    # PMOS transistors (parallel)
    # P1 (left)
    ax.add_patch(FancyBboxPatch((-0.3, 2.8), 0.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#F38181', alpha=0.4, edgecolor='black', linewidth=1.5))
    ax.text(0, 3.1, 'P1', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.plot([0, 0], [3.4, 4], 'k-', linewidth=1.5)  # to VDD
    ax.plot([0, 0], [2.8, 2.4], 'k-', linewidth=1.5)  # to output
    ax.plot([-0.5, -0.3], [3.1, 3.1], 'k-', linewidth=1.5)  # gate
    ax.text(-0.7, 3.1, 'A', fontsize=10, ha='right', fontweight='bold')

    # P2 (right)
    ax.add_patch(FancyBboxPatch((0.7, 2.8), 0.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#F38181', alpha=0.4, edgecolor='black', linewidth=1.5))
    ax.text(1, 3.1, 'P2', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.plot([1, 1], [3.4, 4], 'k-', linewidth=1.5)  # to VDD
    ax.plot([1, 1], [2.8, 2.4], 'k-', linewidth=1.5)  # to output
    ax.plot([1.3, 1.5], [3.1, 3.1], 'k-', linewidth=1.5)  # gate
    ax.text(1.7, 3.1, 'B', fontsize=10, ha='left', fontweight='bold')

    # Output node (ZN)
    ax.plot([0, 1], [2.4, 2.4], 'k-', linewidth=1.5)
    ax.plot([0.5, 0.5], [2.4, 2.0], 'k-', linewidth=1.5)
    ax.plot([0.5, 2], [2.4, 2.4], 'k-', linewidth=1.5)
    ax.add_patch(Circle((2, 2.4), 0.1, facecolor='#FFE66D', edgecolor='black', linewidth=1.5))
    ax.text(2.2, 2.4, 'ZN', fontsize=10, ha='left', fontweight='bold', color='#B8860B')

    # NMOS transistors (series)
    # N1 (top)
    ax.add_patch(FancyBboxPatch((-0.3, 1.4), 0.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#95E1D3', alpha=0.4, edgecolor='black', linewidth=1.5))
    ax.text(0, 1.7, 'N1', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.plot([0, 0.5], [2.0, 2.0], 'k-', linewidth=1.5)  # to output
    ax.plot([0, 0], [2.0, 2.0], 'k-', linewidth=1.5)
    ax.plot([0, 0], [1.4, 1.0], 'k-', linewidth=1.5)  # to N2
    ax.plot([-0.5, -0.3], [1.7, 1.7], 'k-', linewidth=1.5)  # gate
    ax.text(-0.7, 1.7, 'A', fontsize=10, ha='right', fontweight='bold')

    # N2 (bottom)
    ax.add_patch(FancyBboxPatch((-0.3, 0.4), 0.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#95E1D3', alpha=0.4, edgecolor='black', linewidth=1.5))
    ax.text(0, 0.7, 'N2', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.plot([0, 0], [1.0, 1.0], 'k-', linewidth=1.5)  # from N1
    ax.plot([0, 0], [0.4, 0], 'k-', linewidth=1.5)  # to VSS
    ax.plot([-0.5, -0.3], [0.7, 0.7], 'k-', linewidth=1.5)  # gate
    ax.text(-0.7, 0.7, 'B', fontsize=10, ha='right', fontweight='bold')

    # Annotations
    ax.annotate('', xy=(1.8, 3.5), xytext=(1.8, 2.6),
               arrowprops=dict(arrowstyle='->', color='#F38181', lw=2))
    ax.text(2.0, 3.0, 'Pull-up\n(parallel)', fontsize=8, color='#F38181', ha='left')

    ax.annotate('', xy=(-0.8, 0.5), xytext=(-0.8, 1.9),
               arrowprops=dict(arrowstyle='->', color='#95E1D3', lw=2))
    ax.text(-1.3, 1.2, 'Pull-down\n(series)', fontsize=8, color='#2D8A6E', ha='center')


def draw_pullup_graph(ax):
    """Draw pull-up path (rise transition) as graph."""
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Pull-Up Path Graph\n(Rise Transition: VDD → ZN)', fontsize=11, fontweight='bold', pad=10)

    # Node positions
    nodes = {
        'VDD': (1.5, 3),
        'P1': (0.5, 2),
        'P2': (2.5, 2),
        'ZN': (1.5, 1)
    }

    # Draw edges (current flow)
    edges = [
        ('VDD', 'P1', 'k'),
        ('VDD', 'P2', 'k'),
        ('P1', 'ZN', 'k'),
        ('P2', 'ZN', 'k'),
    ]

    for src, dst, color in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate('', xy=(x2, y2+0.25), xytext=(x1, y1-0.25),
                   arrowprops=dict(arrowstyle='->', color='#2D3436', lw=2,
                                  connectionstyle='arc3,rad=0'))

    # Draw nodes
    node_styles = {
        'VDD': {'color': '#FF6B6B', 'size': 800, 'shape': 's'},
        'P1': {'color': '#F38181', 'size': 700, 'shape': 'o'},
        'P2': {'color': '#F38181', 'size': 700, 'shape': 'o'},
        'ZN': {'color': '#FFE66D', 'size': 700, 'shape': 'd'},
    }

    for name, (x, y) in nodes.items():
        style = node_styles[name]
        if style['shape'] == 's':
            marker = plt.scatter([x], [y], s=style['size'], c=style['color'],
                               marker='s', edgecolors='black', linewidths=2, zorder=5)
        elif style['shape'] == 'd':
            marker = plt.scatter([x], [y], s=style['size'], c=style['color'],
                               marker='D', edgecolors='black', linewidths=2, zorder=5)
        else:
            marker = plt.scatter([x], [y], s=style['size'], c=style['color'],
                               marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax.text(x, y, name, fontsize=10, ha='center', va='center', fontweight='bold', zorder=6)

    # Feature annotation
    ax.text(1.5, 0.3, 'Nodes: 4 (VDD, P1, P2, ZN)\nEdges: 4 (current paths)',
           fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def draw_pulldown_graph(ax):
    """Draw pull-down path (fall transition) as graph."""
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) Pull-Down Path Graph\n(Fall Transition: ZN → VSS)', fontsize=11, fontweight='bold', pad=10)

    # Node positions
    nodes = {
        'ZN': (1.5, 3),
        'N1': (1.5, 2),
        'N2': (1.5, 1),
        'VSS': (1.5, 0)
    }

    # Draw edges (current flow)
    edges = [
        ('ZN', 'N1'),
        ('N1', 'N2'),
        ('N2', 'VSS'),
    ]

    for src, dst in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate('', xy=(x2, y2+0.25), xytext=(x1, y1-0.25),
                   arrowprops=dict(arrowstyle='->', color='#2D3436', lw=2))

    # Draw nodes
    node_styles = {
        'ZN': {'color': '#FFE66D', 'size': 700, 'shape': 'd'},
        'N1': {'color': '#95E1D3', 'size': 700, 'shape': 'o'},
        'N2': {'color': '#95E1D3', 'size': 700, 'shape': 'o'},
        'VSS': {'color': '#4ECDC4', 'size': 800, 'shape': 's'},
    }

    for name, (x, y) in nodes.items():
        style = node_styles[name]
        if style['shape'] == 's':
            ax.scatter([x], [y], s=style['size'], c=style['color'],
                      marker='s', edgecolors='black', linewidths=2, zorder=5)
        elif style['shape'] == 'd':
            ax.scatter([x], [y], s=style['size'], c=style['color'],
                      marker='D', edgecolors='black', linewidths=2, zorder=5)
        else:
            ax.scatter([x], [y], s=style['size'], c=style['color'],
                      marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax.text(x, y, name, fontsize=10, ha='center', va='center', fontweight='bold', zorder=6)

    # Feature annotation
    ax.text(1.5, -0.3, 'Nodes: 4 (ZN, N1, N2, VSS)\nEdges: 3 (series path)',
           fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def draw_feature_table(ax):
    """Draw node feature encoding table."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('(d) Node Feature Encoding (7D Base Features)', fontsize=11, fontweight='bold', pad=10)

    # Table data
    headers = ['Node', 'is_power', 'is_port', 'type', 'width', 'V', 'slew', 'load']

    # Pull-up path data
    pullup_data = [
        ['VDD', '1', '0', '0', '0', 'V', '0', '0'],
        ['P1', '0', '0', '-1', 'W₁', 'V', 'τᵢₙ', '0'],
        ['P2', '0', '0', '-1', 'W₂', 'V', 'τᵢₙ', '0'],
        ['ZN', '0', '1', '0', '0', 'V', '0', 'Cₗ'],
    ]

    # Draw table
    cell_width = 1.1
    cell_height = 0.6
    start_x = 0.5
    start_y = 5.0

    # Colors
    header_color = '#DFE6E9'
    row_colors = ['#FFEAA7', '#F38181', '#F38181', '#FFE66D']

    # Draw header
    for j, header in enumerate(headers):
        x = start_x + j * cell_width
        rect = FancyBboxPatch((x, start_y), cell_width, cell_height,
                              boxstyle="round,pad=0.02", facecolor=header_color,
                              edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + cell_width/2, start_y + cell_height/2, header,
               fontsize=8, ha='center', va='center', fontweight='bold')

    # Draw rows
    for i, row in enumerate(pullup_data):
        y = start_y - (i + 1) * cell_height
        for j, cell in enumerate(row):
            x = start_x + j * cell_width
            color = row_colors[i] if j == 0 else 'white'
            alpha = 0.4 if j == 0 else 1.0
            rect = FancyBboxPatch((x, y), cell_width, cell_height,
                                  boxstyle="round,pad=0.02", facecolor=color,
                                  edgecolor='black', linewidth=0.5, alpha=alpha)
            ax.add_patch(rect)
            ax.text(x + cell_width/2, y + cell_height/2, cell,
                   fontsize=8, ha='center', va='center')

    # Legend
    ax.text(5, 1.8, 'Feature Description:', fontsize=9, fontweight='bold')
    descriptions = [
        'is_power: Power rail indicator (VDD/VSS)',
        'is_port: Output port indicator',
        'type: PMOS(-1) / NMOS(+1) / Other(0)',
        'width: Transistor width (μm)',
        'V: Operating voltage',
        'slew: Input transition time (gate-connected MOS)',
        'load: Output capacitive load (output node)',
    ]
    for i, desc in enumerate(descriptions):
        ax.text(0.5, 1.4 - i * 0.22, f'• {desc}', fontsize=7, ha='left')


def create_full_figure(output_dir='.'):
    """Create the complete figure with all components."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          left=0.05, right=0.95, top=0.92, bottom=0.05)

    # (a) Circuit schematic - larger
    ax1 = fig.add_subplot(gs[0, 0])
    draw_nand2_schematic(ax1)

    # (b) Pull-up graph
    ax2 = fig.add_subplot(gs[0, 1])
    draw_pullup_graph(ax2)

    # (c) Pull-down graph
    ax3 = fig.add_subplot(gs[0, 2])
    draw_pulldown_graph(ax3)

    # (d) Feature table - spans bottom row
    ax4 = fig.add_subplot(gs[1, :])
    draw_feature_table(ax4)

    # Main title
    fig.suptitle('Stage-Aware Current Path Representation for 2-Input NAND Gate',
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    png_path = output_path / 'nand2_current_path_encoding.png'
    pdf_path = output_path / 'nand2_current_path_encoding.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')

    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")

    plt.close()


def create_simple_figure(output_dir='.'):
    """Create a simpler 2-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    draw_nand2_schematic(axes[0])
    draw_pullup_graph(axes[1])
    draw_pulldown_graph(axes[2])

    fig.suptitle('Stage-Aware Current Path: 2-Input NAND Gate',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    png_path = output_path / 'nand2_current_path_simple.png'
    pdf_path = output_path / 'nand2_current_path_simple.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')

    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize NAND2 Current Path for Paper')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--simple', action='store_true',
                       help='Create simpler 3-panel figure')

    args = parser.parse_args()

    if args.simple:
        create_simple_figure(args.output_dir)
    else:
        create_full_figure(args.output_dir)
