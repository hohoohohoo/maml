"""
Parallel Liberty extractor for *constraint* LUTs (setup / hold / recovery /
removal / non_seq_setup / non_seq_hold).

Sibling of `libdata_extract_MAML_cell.py` and `libdata_extract_MAML_transition.py`
— independent module so the existing cell / transition pipeline stays untouched.

Scope (this file):
  - Parses ONLY `constraint_template_3x3` LUTs.
  - Skips `mpw_constraint_template_*` (min_pulse_width — different physical
    quantity, single-pin width constraint rather than two-pin race).
  - Skips delay/transition LUTs (those have their own extractors).

Output organization:
  - `parse_liberty_pin_blocks(lines)` returns a flat list of `pin` dicts (one per
    pin block, same shape as the cell extractor) with each `timing` carrying
    `timing_type`, `when`, `rise_constraint` / `fall_constraint` LUTs, and the
    LUT indices.  Insertion order is preserved (LUTs appear in the order they
    occur in the Liberty file).
  - `flatten_pin_data(pin_data, include_categories=None)` returns
    `(rows_by_category, cap_rows)` where `rows_by_category` is a dict keyed by
    the timing category — `setup`, `hold`, `recovery`, `removal`,
    `non_seq_setup`, `non_seq_hold` — each value a list of row dicts in LUT
    order.  Pass `include_categories` to restrict which keys are populated; by
    default all six are.

Downstream consumer (builder) is expected to write one PTH per category, so
`setup` and `hold` are saved unconditionally and the remaining four are gated
by builder-level flags.
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Timing-type → category map. Categories drive separate output files.
# ---------------------------------------------------------------------------
CONSTRAINT_TIMING_CATEGORIES = {
    'setup':         {'setup_rising',         'setup_falling'},
    'hold':          {'hold_rising',          'hold_falling'},
    'recovery':      {'recovery_rising',      'recovery_falling'},
    'removal':       {'removal_rising',       'removal_falling'},
    'non_seq_setup': {'non_seq_setup_rising', 'non_seq_setup_falling'},
    'non_seq_hold':  {'non_seq_hold_rising',  'non_seq_hold_falling'},
}
DEFAULT_CATEGORIES = ('setup', 'hold')                         # always emitted by builder
OPTIONAL_CATEGORIES = ('recovery', 'removal', 'non_seq_setup', 'non_seq_hold')

TIMING_TYPE_TO_CATEGORY = {
    tt: cat for cat, tts in CONSTRAINT_TIMING_CATEGORIES.items() for tt in tts
}
CONSTRAINT_TYPES = ('rise_constraint', 'fall_constraint')      # LUT container names
CONSTRAINT_TEMPLATE_KEEP = 'constraint_template_3x3'           # accept
CONSTRAINT_TEMPLATE_SKIP = 'mpw_constraint_template'           # reject prefix (mpw_*)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_liberty_pin_blocks(lines):
    """Parse a Liberty file's pin blocks for constraint LUTs only.

    Returns a list of `pin` dicts; each pin carries a 'timings' list, each
    timing carries 'timing_type', 'related_pin', 'when' (if present), and the
    'rise_constraint' / 'fall_constraint' nested LUTs (2D values + index_1 +
    index_2).  Timings whose only LUT was a mpw_constraint_template_* are
    dropped.  Order of appearance in the Liberty file is preserved.
    """
    pins = []
    current_cell_name = None
    cell_size         = None
    input_port_num    = None
    current_pin       = None
    current_timing    = None
    current_template  = None        # template name inside the open constraint(...)
    delay_type        = None        # 'rise_constraint' or 'fall_constraint'
    delay_values      = []
    inside_pin           = False
    inside_timing        = False
    inside_delaytype     = False    # inside rise_constraint(...) or fall_constraint(...)
    inside_values_block  = False
    inside_PVT_condition = False

    process_value = temp_value = voltage_value = None

    for line in lines:
        line = line.strip()

        # PVT block ------------------------------------------------------
        if line.startswith("operating_conditions"):
            inside_PVT_condition = True
        if inside_PVT_condition:
            m = re.search(r'process\s*:\s*(\d+)', line)
            if m: process_value = int(m.group(1))
            m = re.search(r'temperature\s*:\s*([-+]?\d+)', line)
            if m: temp_value = int(m.group(1))
            m = re.search(r'voltage\s*:\s*([-+]?\d*\.?\d+)', line)
            if m: voltage_value = float(m.group(1))
            if line == "}":
                inside_PVT_condition = False

        # cell -----------------------------------------------------------
        if line.startswith("cell"):
            match = re.search(r'cell\s*\((.*?)\)', line)
            if match:
                full_cell_name = match.group(1)
                current_cell_name = full_cell_name
                m_ipn = re.search(r'(\d+)(?=[xX])', full_cell_name)
                input_port_num = int(m_ipn.group(1)) if m_ipn else None
                size_match = re.search(r'[xX](\d+)(p(\d+))?|p(\d+)', full_cell_name)
                if size_match:
                    if size_match.group(2):
                        if float(size_match.group(3)) >= 10:
                            cell_size = str(float(size_match.group(1)) + float(size_match.group(3)) / 100)
                        else:
                            cell_size = str(float(size_match.group(1)) + float(size_match.group(3)) / 10)
                    elif 'p' in size_match.group(0):
                        s2 = re.search(r'p(\d+)', full_cell_name)
                        v = float(s2.group(1))
                        cell_size = str(v / 100) if v >= 10 else str(v / 10)
                    else:
                        cell_size = size_match.group(1)
                else:
                    cell_size = None

        # pin block ------------------------------------------------------
        if line.startswith("pin"):
            inside_pin = True
            current_pin = {}
            m = re.search(r'pin\s*\((.*?)\)', line)
            current_pin['cell']           = current_cell_name
            current_pin['size']           = cell_size
            current_pin['input_port_num'] = input_port_num
            if m:
                current_pin['pin_name'] = m.group(1)

        # timing block start --------------------------------------------
        elif inside_pin and line.startswith("timing()"):
            inside_timing = True
            current_timing = {}

        # streaming a values(...) block ---------------------------------
        elif inside_values_block:
            delay_values.extend(re.findall(r'"(.*?)"', line))
            if line.endswith(");") or ");" in line:
                inside_values_block = False

        # close a constraint(...) container -----------------------------
        elif inside_delaytype and line == "}":
            if delay_values and current_template == CONSTRAINT_TEMPLATE_KEEP:
                current_timing[delay_type] = [
                    list(map(float, row.split(','))) for row in delay_values
                ]
            delay_values = []
            inside_delaytype = False
            current_template = None

        # close a timing() block ----------------------------------------
        elif inside_timing and line == "}":
            if current_timing and current_pin is not None:
                # Drop timing entries that ended up empty (e.g., only mpw was inside).
                has_constraint_lut = any(k in current_timing for k in CONSTRAINT_TYPES)
                if has_constraint_lut:
                    current_pin.setdefault("timings", []).append(current_timing)
            current_timing = {}
            inside_timing = False

        # close a pin() block -------------------------------------------
        elif inside_pin and line == "}":
            if current_pin and current_pin.get("timings"):
                # Only keep pin entries that produced at least one constraint LUT.
                current_pin["Process"]     = process_value
                current_pin["Temperature"] = temp_value
                current_pin["Voltage"]     = voltage_value
                pins.append(current_pin)
            current_pin = None
            inside_pin = False

        # inside a timing() block ---------------------------------------
        elif inside_timing:
            if "related_pin" in line:
                m = re.search(r'"(.*?)"', line)
                if m: current_timing["related_pin"] = m.group(1)
            elif "timing_type" in line:
                # Capture timing_type so categories can be assigned later.
                # Strip whitespace AND trailing semicolon (order matters: ';' may sit before/after spaces).
                current_timing["timing_type"] = line.split(":", 1)[1].strip().rstrip(';').strip()
            elif line.startswith("when"):
                m = re.search(r'"(.*?)"', line)
                if m: current_timing["when"] = m.group(1)
            elif "sdf_cond" in line:
                m = re.search(r'"(.*?)"', line)
                if m: current_timing["sdf_cond"] = m.group(1)

            # Constraint container: detect both name and the template inside ().
            elif any(ct in line for ct in CONSTRAINT_TYPES) and '(' in line:
                m = re.search(r'(rise_constraint|fall_constraint)\s*\(\s*([\w]+)\s*\)', line)
                if m:
                    container = m.group(1)
                    template  = m.group(2)
                    # Reject mpw_constraint_template_*; accept constraint_template_3x3 (and any other
                    # constraint_template_NxN that may appear in libs, sized > 3x3).
                    if template.startswith(CONSTRAINT_TEMPLATE_SKIP):
                        current_template = template            # mark so values() inside is ignored
                        delay_type = None
                    else:
                        current_template = template
                        delay_type = container
                    inside_delaytype = True

            elif "index_1" in line and inside_delaytype and current_template == CONSTRAINT_TEMPLATE_KEEP:
                # Match plain decimals only (slew axis values in lib are non-scientific).
                # The first match is the "1" in "index_1", so drop it via [1:].
                current_timing["index_1"] = list(map(float, re.findall(r'[\d.]+', line)[1:]))
            elif "index_2" in line and inside_delaytype and current_template == CONSTRAINT_TEMPLATE_KEEP:
                current_timing["index_2"] = list(map(float, re.findall(r'[\d.]+', line)[1:]))
            elif "values" in line and inside_delaytype and current_template == CONSTRAINT_TEMPLATE_KEEP:
                inside_values_block = True
                delay_values.extend(re.findall(r'"(.*?)"', line))

        # generic pin-level attributes (capacitance etc.) ---------------
        elif inside_pin and ":" in line and not inside_timing:
            key, val = line.split(":", 1)
            val_clean = val.strip().strip(';').strip('"')
            current_pin[key.strip()] = val_clean

    return pins


# ---------------------------------------------------------------------------
# Flatten — split rows by category, preserving lib order
# ---------------------------------------------------------------------------
def flatten_pin_data(pin_data, include_categories=None):
    """Flatten pin/timing/constraint-LUT structure into per-category row lists.

    Args
    ----
    pin_data : list[dict]
        Output of `parse_liberty_pin_blocks`.
    include_categories : Iterable[str] | None
        Subset of {'setup','hold','recovery','removal','non_seq_setup',
        'non_seq_hold'} to populate. `None` populates all six.

    Returns
    -------
    rows_by_category : dict[str, list[dict]]
        Keys are categories, values are lists of row dicts in the order LUTs
        appeared in the Liberty file (pin order × timing order × constraint
        type order).  Empty lists when no entries fall in that category.
    cap_rows : list[dict]
        Per-pin capacitance metadata rows (same as cell / transition
        extractors), included for downstream symmetry.
    """
    if include_categories is None:
        include_categories = tuple(CONSTRAINT_TIMING_CATEGORIES.keys())
    include_categories = set(include_categories)

    rows_by_category = {cat: [] for cat in CONSTRAINT_TIMING_CATEGORIES}
    cap_rows = []

    for pin in pin_data:
        process     = pin.get("Process", "")
        temperature = pin.get("Temperature", "")
        voltage     = pin.get("Voltage", "")
        cell_name   = pin.get("cell", "")
        cell_size   = pin.get("size", "")
        pin_name    = pin.get("pin_name", "")
        input_port_num = pin.get("input_port_num", "")
        capacitance      = pin.get("capacitance", "")
        rise_capacitance = pin.get("rise_capacitance", "")
        fall_capacitance = pin.get("fall_capacitance", "")

        if capacitance:
            cap_rows.append({
                "cell": cell_name, "size": cell_size,
                "input_port_num": input_port_num, "pin_name": pin_name,
                "capacitance": capacitance,
                "rise_capacitance": rise_capacitance,
                "fall_capacitance": fall_capacitance,
            })

        for t in pin.get("timings", []):
            timing_type = t.get("timing_type", "")
            category = TIMING_TYPE_TO_CATEGORY.get(timing_type)
            if category is None or category not in include_categories:
                continue
            related_pin = t.get("related_pin", "")
            when        = t.get("when", "")
            sdf_cond    = t.get("sdf_cond", "")
            index_1     = t.get("index_1", [])
            index_2     = t.get("index_2", [])

            for ctype in CONSTRAINT_TYPES:                          # rise then fall, lib order
                if ctype not in t:
                    continue
                rows_by_category[category].append({
                    "Process":        process,
                    "Temperature":    temperature,
                    "Voltage":        voltage,
                    "cell":           cell_name,
                    "size":           cell_size,
                    "type":           "constraint",
                    "pin_name":       pin_name,
                    "input_port_num": input_port_num,
                    "capacitance":    capacitance,
                    "related_pin":    related_pin,
                    "timing_type":    timing_type,
                    "category":       category,
                    "when":           when,
                    "sdf_cond":       sdf_cond,
                    "delay_type":     ctype,                        # 'rise_constraint' / 'fall_constraint'
                    "values":         t[ctype],                     # 2D list (3x3)
                    "index_1":        index_1,                      # related_pin_transition slews
                    "index_2":        index_2,                      # constrained_pin_transition slews
                })

    return rows_by_category, cap_rows


# ---------------------------------------------------------------------------
# Convenience: convert a category's rows to a pandas DataFrame (matches the
# cell / transition extractors' downstream interface).
# ---------------------------------------------------------------------------
def rows_to_dataframe(rows):
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Minimal smoke test:
    #   python libdata_extract_MAML_constraint.py <path/to/.lib>
    import sys
    if len(sys.argv) < 2:
        print("usage: python libdata_extract_MAML_constraint.py <lib>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        pins = parse_liberty_pin_blocks(f.readlines())
    by_cat, _ = flatten_pin_data(pins)
    print(f"parsed {len(pins)} pin blocks with constraint LUTs")
    for cat, rows in by_cat.items():
        if rows:
            cells = sorted({r['cell'] for r in rows})
            tts   = sorted({r['timing_type'] for r in rows})
            print(f"  [{cat:13s}] rows={len(rows):5d}  cells={len(cells)}  timing_types={tts}")
        else:
            print(f"  [{cat:13s}] rows=0")
