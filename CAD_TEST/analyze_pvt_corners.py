#!/usr/bin/env python3
"""
Analyze PVT corner delay patterns
"""

# All conditions with their outputs (from first task: AN4YM16 / cell_fall)
all_data = {
    # Support conditions
    'ff0p88vm40c': {'corner': 'FF', 'voltage': 0.88, 'temp': -40, 'delay': 0.264363},
    'ss0p72vm40c': {'corner': 'SS', 'voltage': 0.72, 'temp': -40, 'delay': 2.527260},
    'tt0p8v25c': {'corner': 'TT', 'voltage': 0.80, 'temp': 25, 'delay': 0.492759},
    'ff0p99v125c': {'corner': 'FF', 'voltage': 0.99, 'temp': 125, 'delay': 0.198350},
    'ss0p81v125c': {'corner': 'SS', 'voltage': 0.81, 'temp': 125, 'delay': 0.744809},
    'tt0p9v25c': {'corner': 'TT', 'voltage': 0.90, 'temp': 25, 'delay': 0.363336},
    'ff1p1v125c': {'corner': 'FF', 'voltage': 1.10, 'temp': 125, 'delay': 0.168710},
    'ff1p1vm40c': {'corner': 'FF', 'voltage': 1.10, 'temp': -40, 'delay': 0.174016},
    'ss0p9v125c': {'corner': 'SS', 'voltage': 0.90, 'temp': 125, 'delay': 0.540752},
    'ss0p9vm40c': {'corner': 'SS', 'voltage': 0.90, 'temp': -40, 'delay': 0.651044},
    # Test conditions
    'ff0p88v125c': {'corner': 'FF', 'voltage': 0.88, 'temp': 125, 'delay': 0.240873},
    'ss0p72v125c': {'corner': 'SS', 'voltage': 0.72, 'temp': 125, 'delay': 1.168940},
    'ff0p99vm40c': {'corner': 'FF', 'voltage': 0.99, 'temp': -40, 'delay': 0.210771},
    'ss0p81vm40c': {'corner': 'SS', 'voltage': 0.81, 'temp': -40, 'delay': 1.083850},
    'tt1p0v25c': {'corner': 'TT', 'voltage': 1.00, 'temp': 25, 'delay': 0.289540},
}

# Sort by delay
sorted_by_delay = sorted(all_data.items(), key=lambda x: x[1]['delay'])

print("=" * 80)
print("PVT CONDITIONS SORTED BY DELAY (Fastest → Slowest)")
print("=" * 80)
print(f"{'Rank':<5} {'Condition':<15} {'Corner':<6} {'Voltage':<8} {'Temp':<8} {'Delay (ns)':<12}")
print("-" * 80)

for i, (cond, data) in enumerate(sorted_by_delay):
    print(f"{i+1:<5} {cond:<15} {data['corner']:<6} {data['voltage']:<8.2f} {data['temp']:<8.0f} {data['delay']:<12.6f}")

print("\n" + "=" * 80)
print("CORNER ANALYSIS")
print("=" * 80)

# Fastest conditions (smallest delay)
print("\n🚀 FASTEST (smallest delay):")
for i, (cond, data) in enumerate(sorted_by_delay[:5]):
    print(f"  {i+1}. {cond}: {data['delay']:.4f} ns ({data['corner']}, {data['voltage']}V, {data['temp']}C)")

# Slowest conditions (largest delay)
print("\n🐢 SLOWEST (largest delay):")
for i, (cond, data) in enumerate(sorted_by_delay[-5:][::-1]):
    print(f"  {i+1}. {cond}: {data['delay']:.4f} ns ({data['corner']}, {data['voltage']}V, {data['temp']}C)")

# Analyze by corner
print("\n" + "=" * 80)
print("BY CORNER")
print("=" * 80)

for corner in ['FF', 'TT', 'SS']:
    corner_data = [(k, v) for k, v in all_data.items() if v['corner'] == corner]
    corner_data.sort(key=lambda x: x[1]['delay'])
    delays = [v['delay'] for _, v in corner_data]
    print(f"\n{corner} corner: min={min(delays):.4f} ns, max={max(delays):.4f} ns")
    for cond, data in corner_data:
        print(f"  {cond}: {data['delay']:.4f} ns ({data['voltage']}V, {data['temp']}C)")

# Temperature effect analysis
print("\n" + "=" * 80)
print("TEMPERATURE EFFECT (same corner, same voltage)")
print("=" * 80)

# Compare pairs with same corner/voltage but different temperature
pairs = [
    ('ff0p88vm40c', 'ff0p88v125c'),  # FF, 0.88V
    ('ff0p99vm40c', 'ff0p99v125c'),  # FF, 0.99V
    ('ff1p1vm40c', 'ff1p1v125c'),    # FF, 1.1V
    ('ss0p72vm40c', 'ss0p72v125c'),  # SS, 0.72V
    ('ss0p81vm40c', 'ss0p81v125c'),  # SS, 0.81V
    ('ss0p9vm40c', 'ss0p9v125c'),    # SS, 0.9V
]

print(f"\n{'Condition Pair':<35} {'-40C':<12} {'125C':<12} {'Diff':<12} {'Effect'}")
print("-" * 80)
for cold, hot in pairs:
    if cold in all_data and hot in all_data:
        cold_delay = all_data[cold]['delay']
        hot_delay = all_data[hot]['delay']
        diff = hot_delay - cold_delay
        effect = "Hot slower" if diff > 0 else "Cold slower"
        print(f"{cold} vs {hot}: {cold_delay:<12.4f} {hot_delay:<12.4f} {diff:+.4f}      {effect}")
