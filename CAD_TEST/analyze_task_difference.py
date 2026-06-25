#!/usr/bin/env python3
"""
Analyze the difference between TSMC task definition and CAD_TEST task definition
"""

import torch
import numpy as np

print("=" * 80)
print("TSMC vs CAD_TEST TASK DEFINITION COMPARISON")
print("=" * 80)

# Load TSMC training data
TSMC_TRAIN_PATH = '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/combined_data/tsmc_topology_agnostic_train_input_cell.pth'

print("\n[1] TSMC Training Data Structure")
print("-" * 60)
tsmc_data = torch.load(TSMC_TRAIN_PATH)
print(f"Shape: {tsmc_data.shape}")
print(f"  - Dim 0: {tsmc_data.shape[0]} tasks (cell/pin/delay combinations)")
print(f"  - Dim 1: {tsmc_data.shape[1]} samples per task (voltage sweep)")
print(f"  - Dim 2: {tsmc_data.shape[2]} features")

# Analyze one task
task_0 = tsmc_data[0]
print(f"\nTask 0 analysis (61 samples):")
print(f"  Feature indices: [a, b, c, temp, voltage, dim, delay, slew, load]")
print(f"  Temperature range: {task_0[:, 3].min():.1f} ~ {task_0[:, 3].max():.1f}")
print(f"  Voltage range: {task_0[:, 4].min():.4f} ~ {task_0[:, 4].max():.4f}")
print(f"  Slew (unique): {torch.unique(task_0[:, 7])}")
print(f"  Load (unique): {torch.unique(task_0[:, 8])}")

# Check if temperature/corner is constant within a task
temp_std = task_0[:, 3].std().item()
a_std = task_0[:, 0].std().item()
print(f"\n  Temperature std within task: {temp_std:.6f} (should be ~0 if constant)")
print(f"  'a' param std within task: {a_std:.6f} (should be ~0 if same corner)")

print("\n" + "=" * 80)
print("[2] TSMC TASK DEFINITION")
print("=" * 80)
print("""
TSMC Task = (cell, delay_type, pin, slew, load, corner, temperature)
           fixed within task: cell, delay_type, pin, slew, load, corner, temperature
           varies within task: VOLTAGE (61 points sweep)

Support samples: 3-5 voltage points (e.g., indices 5, 30, 55)
Query samples: remaining voltage points

=> Model learns voltage variation curve within SAME PVT corner
""")

print("\n" + "=" * 80)
print("[3] CAD_TEST TASK DEFINITION")
print("=" * 80)
print("""
CAD_TEST Task = (cell, delay_type, pin, slew_idx, load_idx)
               fixed within task: cell, delay_type, pin, slew, load
               varies within task: PVT CONDITIONS (15 different combinations)

Support samples: 10 PVT conditions
  - ff0p88vm40c, ss0p72vm40c, tt0p8v25c, ff0p99v125c, ss0p81v125c
  - tt0p9v25c, ff1p1v125c, ff1p1vm40c, ss0p9v125c, ss0p9vm40c

Query samples: 5 different PVT conditions
  - ff0p88v125c, ss0p72v125c, ff0p99vm40c, ss0p81vm40c, tt1p0v25c

=> Model needs to predict across DIFFERENT corners AND temperatures simultaneously
""")

print("\n" + "=" * 80)
print("[4] KEY DIFFERENCE")
print("=" * 80)

print("""
TSMC Training:
  - 61 points in a task = voltage sweep (0.72V ~ 0.88V)
  - Same corner (e.g., TT), Same temperature (e.g., 25C)
  - Model learns: f(voltage) -> delay, given fixed PVT corner

CAD_TEST Validation:
  - 15 points in a task = 15 different PVT conditions
  - Different corners (FF, TT, SS)
  - Different temperatures (-40C, 25C, 125C)
  - Different voltages (0.72V ~ 1.10V)
  - Model needs to learn: f(corner, temperature, voltage) -> delay

MISMATCH:
  - Model was trained to extrapolate/interpolate VOLTAGE within fixed corner/temp
  - CAD_TEST requires extrapolation across MULTIPLE corners AND temperatures
  - This is a much harder generalization task!
""")

# Calculate variation ranges
print("\n" + "=" * 80)
print("[5] VARIATION RANGE COMPARISON")
print("=" * 80)

# TSMC: voltage variation within a task
task_voltages = task_0[:, 4]
tsmc_volt_range = task_voltages.max() - task_voltages.min()
print(f"\nTSMC (within task):")
print(f"  Voltage range: {task_voltages.min():.4f} ~ {task_voltages.max():.4f} ({tsmc_volt_range:.4f}V)")
print(f"  Temperature: CONSTANT")
print(f"  Corner: CONSTANT")

# CAD_TEST: PVT variation
print(f"\nCAD_TEST (within task):")
print(f"  Voltage range: 0.72V ~ 1.10V (0.38V) - 2x wider than TSMC!")
print(f"  Temperature: -40C ~ 125C (165C range)")
print(f"  Corner: FF, TT, SS (3 different corners)")

print("\n" + "=" * 80)
print("[6] WHY RESULTS ARE BAD")
print("=" * 80)
print("""
1. DOMAIN SHIFT:
   - Training: Voltage interpolation within fixed PVT
   - Testing: Cross-PVT generalization

2. TASK COMPLEXITY:
   - TSMC: 1D interpolation (voltage axis only)
   - CAD_TEST: 3D extrapolation (voltage + temperature + corner)

3. ADAPTATION LIMITATION:
   - 10 support points may not be enough to capture complex 3D surface
   - Model initialized for voltage-only adaptation

4. FEATURE DISTRIBUTION:
   - Temperature feature was constant during training
   - Model may not have learned temperature dependency well
""")

print("\n" + "=" * 80)
print("[7] POTENTIAL SOLUTIONS")
print("=" * 80)
print("""
1. RETRAIN with multi-PVT tasks:
   - Create TSMC tasks that vary across corners/temperatures

2. INCREASE support shots:
   - Use more than 10 support samples
   - Sample more densely in PVT space

3. TASK FORMULATION change:
   - Group by voltage domain (0.8V, 0.9V, 1.0V) separately
   - Adapt within each voltage domain

4. FEATURE ENGINEERING:
   - Normalize PVT features differently
   - Add corner one-hot encoding
""")

del tsmc_data
