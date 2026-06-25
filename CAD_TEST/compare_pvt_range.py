#!/usr/bin/env python3
"""
Compare PVT variation range between CAD_TEST and typical TSMC
"""

import numpy as np

print("=" * 80)
print("CAD_TEST PVT VARIATION ANALYSIS")
print("=" * 80)

# CAD_TEST PVT conditions
cad_test_conditions = {
    # Support (10 conditions)
    'ff0p88vm40c': {'corner': 'FF', 'voltage': 0.88, 'temp': -40},
    'ss0p72vm40c': {'corner': 'SS', 'voltage': 0.72, 'temp': -40},
    'tt0p8v25c': {'corner': 'TT', 'voltage': 0.80, 'temp': 25},
    'ff0p99v125c': {'corner': 'FF', 'voltage': 0.99, 'temp': 125},
    'ss0p81v125c': {'corner': 'SS', 'voltage': 0.81, 'temp': 125},
    'tt0p9v25c': {'corner': 'TT', 'voltage': 0.90, 'temp': 25},
    'ff1p1v125c': {'corner': 'FF', 'voltage': 1.10, 'temp': 125},
    'ff1p1vm40c': {'corner': 'FF', 'voltage': 1.10, 'temp': -40},
    'ss0p9v125c': {'corner': 'SS', 'voltage': 0.90, 'temp': 125},
    'ss0p9vm40c': {'corner': 'SS', 'voltage': 0.90, 'temp': -40},
    # Test (5 conditions)
    'ff0p88v125c': {'corner': 'FF', 'voltage': 0.88, 'temp': 125},
    'ss0p72v125c': {'corner': 'SS', 'voltage': 0.72, 'temp': 125},
    'ff0p99vm40c': {'corner': 'FF', 'voltage': 0.99, 'temp': -40},
    'ss0p81vm40c': {'corner': 'SS', 'voltage': 0.81, 'temp': -40},
    'tt1p0v25c': {'corner': 'TT', 'voltage': 1.00, 'temp': 25},
}

voltages = [v['voltage'] for v in cad_test_conditions.values()]
temps = [v['temp'] for v in cad_test_conditions.values()]

print("\nCAD_TEST Voltage Range:")
print(f"  Min: {min(voltages):.2f}V")
print(f"  Max: {max(voltages):.2f}V")
print(f"  Range: {max(voltages) - min(voltages):.2f}V")
print(f"  Unique voltages: {sorted(set(voltages))}")

print("\nCAD_TEST Temperature Range:")
print(f"  Min: {min(temps)}C")
print(f"  Max: {max(temps)}C")
print(f"  Range: {max(temps) - min(temps)}C")
print(f"  Unique temps: {sorted(set(temps))}")

# Delay variation (from first task analysis)
# Support outputs for first task
support_delays = [0.264363, 2.527260, 0.492759, 0.198350, 0.744809,
                  0.363336, 0.168710, 0.174016, 0.540752, 0.651044]
test_delays = [0.240873, 1.168940, 0.210771, 1.083850, 0.289540]
all_delays = support_delays + test_delays

print("\nCAD_TEST Delay Variation (first task):")
print(f"  Min delay: {min(all_delays):.4f} ns")
print(f"  Max delay: {max(all_delays):.4f} ns")
print(f"  Max/Min ratio: {max(all_delays)/min(all_delays):.2f}x")

print("\n" + "=" * 80)
print("TYPICAL TSMC PVT VARIATION (for comparison)")
print("=" * 80)

# Typical TSMC voltage variation (intra-topology)
# Usually ~±10% from nominal
print("\nTypical TSMC Voltage Range:")
print("  Usually ±10% from nominal")
print("  Example: 0.72V to 0.88V (for 0.8V nominal)")
print("  Range: ~0.16V")

# Typical TSMC temperature range
print("\nTypical TSMC Temperature Range:")
print("  Usually: -40C to 125C (industrial)")
print("  Or: 0C to 85C (commercial)")
print("  Range: 165C (industrial)")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

# CAD_TEST has multiple voltage groups
print("\nVoltage Groups in CAD_TEST:")
voltage_groups = {
    '0.8V group': [0.72, 0.80, 0.88],
    '0.9V group': [0.81, 0.90, 0.99],
    '1.0V group': [0.90, 1.00, 1.10],
}
for group, voltages in voltage_groups.items():
    print(f"  {group}: {voltages} (range: {max(voltages)-min(voltages):.2f}V)")

# Calculate voltage variation percentage
nominal_voltages = [0.8, 0.9, 1.0]
print("\nVoltage Variation from Nominal:")
for nom in nominal_voltages:
    relevant_v = [v for v in set([c['voltage'] for c in cad_test_conditions.values()])
                  if abs(v - nom) < 0.15]
    if relevant_v:
        min_v, max_v = min(relevant_v), max(relevant_v)
        print(f"  Around {nom}V: {min_v}V to {max_v}V ({(min_v-nom)/nom*100:+.1f}% to {(max_v-nom)/nom*100:+.1f}%)")

print("\n" + "=" * 80)
print("KEY FINDING: CAD_TEST has 3 SEPARATE VOLTAGE DOMAINS")
print("=" * 80)
print("""
CAD_TEST covers 3 voltage domains simultaneously:
  1. 0.8V nominal: 0.72V (-10%) to 0.88V (+10%)
  2. 0.9V nominal: 0.81V (-10%) to 0.99V (+10%)
  3. 1.0V nominal: 0.90V (-10%) to 1.10V (+10%)

This is UNUSUAL compared to typical TSMC single-voltage variation.
The model sees a much wider absolute voltage range (0.72V to 1.10V = 0.38V)
compared to typical ±10% variation around single nominal (~0.16V).

This may cause challenges for the MAML model trained on single-voltage tasks.
""")
