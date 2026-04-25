import pandas as pd
import re

def one_hot_encode(index, length):
    return [1 if i == index else 0 for i in range(length)]
    
def get_related_pin_index(related_pin, input_port_names):
    try:
        return input_port_names.index(related_pin)
    except ValueError:
        return -1  # related_pin이 리스트에 없을 경우
    
def get_cap(s, cap):
    for c in cap:
        if s['cell'] == c['cell'] and s['size'] == c['size'] and s['related_pin'] == c['pin_name'] and s['input_port_num'] == c['input_port_num']:
            capacitance = c["capacitance"]
            rise_cap = c["rise_capacitance"]
            fall_cap = c["fall_capacitance"]
    if capacitance is not None:
        return capacitance, rise_cap, fall_cap
    else:
        return -1

def transform_sample(sample, cap, cell_types, delay_types, size_types, port_nums, lib_prefix="", abc_params=None):
    """
    Transform sample for invbuf files with 9 features:
    - Original 5 features: [voltage, additional_dim, delay_indicator, index_1_val, index_2_val]
    - Additional 4 features: [a_param, b_param, c_param, temperature]
    Total: 9 features
    """
    # Get original 5 features logic (same as transform_sample_MAML_5feature.py)
    process = int(sample.get('Process', 0))
    process_types = [1, 2, 3]
    process_one_hot = one_hot_encode(process-1, len(process_types)) if process != -1 else [0]*len(process_types)
    temperature = float(sample.get('Temperature', 0))
    voltage = float(sample.get('Voltage', 0))
    
    input_port_num = port_nums.index(str(sample['input_port_num'])) if str(sample['input_port_num']) in port_nums else -1
    num_one_hot = one_hot_encode(input_port_num, len(port_nums)) if input_port_num != -1 else [0]*len(port_nums)
    
    # Cell one-hot
    cell_index = cell_types.index(sample['cell']) if sample['cell'] in cell_types else -1
    cell_one_hot = one_hot_encode(cell_index, len(cell_types)) if cell_index != -1 else [0]*len(cell_types)
   
    # Delay type one-hot
    delay_index = delay_types.index(sample['delay_type']) if sample['delay_type'] in delay_types else -1
    delay_one_hot = one_hot_encode(delay_index, len(delay_types)) if delay_index != -1 else [0]*len(delay_types)
    
    size_index = size_types.index(sample['size']) if sample['size'] in size_types else -1
    size_one_hot = one_hot_encode(size_index, len(size_types)) if size_index != -1 else [0]*len(size_types)
    
    position2 = get_related_pin_index(sample["related_pin"], sample["input_port_name"])
    
    # Numeric features
    size = float(sample.get('size', 0))
    process = int(sample.get('Process', 0))
    index_1 = sample.get('index_1', [])
    index_2 = sample.get('index_2', [])

    # Flatten values and calculate position
    values_2d = sample.get('values', [])
    flat_values = [v for row in values_2d for v in row]
    
    # Add delay type indicator: fall -> 1, rise -> -1
    delay_indicator = 0
    if 'fall' in sample['delay_type']:
        delay_indicator = 1
    elif 'rise' in sample['delay_type']:
        delay_indicator = -1
    
    # Check if lib_prefix starts with tsmc for additional_dim logic
    additional_dim = 0
    if cell_index != -1:
        cell_name = cell_types[cell_index].upper()
        #print("hi2")
        if lib_prefix.startswith("tsmc"):
            # TSMC specific cell name patterns
            # additional_dim=1: NAND(ND), NOR(NR), INV, HA, and inverted complex gates (IAO, IOA)
            if any(gate in cell_name for gate in ['ND2','ND3','ND4','NR2','NR3','NR4','INV','IAO','IOA']):
                additional_dim = 1
            # additional_dim=2: AND(AN), OR, BUF, XOR, XNOR(XNR), FA, and complex gates (AO, OA)
            elif any(gate in cell_name for gate in ['AN2','AN3','AN4','OR2','OR3','OR4','BUFF','XOR','XNR','FA','AO','OA','HA']):
                additional_dim = 2
            else:
                additional_dim = 2  # Default
        elif lib_prefix.startswith("invbuf") or lib_prefix.startswith("ex2") or lib_prefix.startswith("simple"):
            # Original invbuf logic
            if any(gate in cell_name for gate in ['NAND','NOR','INV','HA', 'MAJI','CKINVDC']):
                additional_dim = 1
            elif any(gate in cell_name for gate in ['AND','OR','BUF','FA', 'MAJ','XNOR','XOR','HB']):
                additional_dim = 2
            else:
                additional_dim = 1  # Default for invbuf
    
    # Get abc parameter values (new feature for invbuf)
    # A, B and C parameters all have nmos(_n) and pmos(_p) versions
    a_n_param = abc_params.get('a_n', 0.0) if abc_params else 0.0  # nmos version
    a_p_param = abc_params.get('a_p', 0.0) if abc_params else 0.0  # pmos version
    b_n_param = abc_params.get('b_n', 0.0) if abc_params else 0.0  # nmos version
    b_p_param = abc_params.get('b_p', 0.0) if abc_params else 0.0  # pmos version
    c_n_param = abc_params.get('c_n', 0.0) if abc_params else 0.0  # nmos version  
    c_p_param = abc_params.get('c_p', 0.0) if abc_params else 0.0  # pmos version
    
    # Select A, B and C parameters based on additional_dim and delay_indicator
    if additional_dim == 2:
        # Both nmos and pmos are used
        a_param = (a_n_param + a_p_param) / 2  # Average for a
        b_param = b_n_param + b_p_param  # Sum for b
        c_param = c_n_param + c_p_param  # Sum for c
    elif additional_dim == 1:
        # Only one type is used - select based on delay_indicator
        if delay_indicator == 1:  # fall -> typically nmos dominant
            a_param = a_n_param
            b_param = b_n_param
            c_param = c_n_param
        elif delay_indicator == -1:  # rise -> typically pmos dominant  
            a_param = a_p_param
            b_param = b_p_param
            c_param = c_p_param
        else:  # neutral case
            a_param = (a_n_param + a_p_param) / 2
            b_param = (b_n_param + b_p_param) / 2
            c_param = (c_n_param + c_p_param) / 2
    else:
        # Default case
        a_param = (a_n_param + a_p_param) / 2
        b_param = (b_n_param + b_p_param) / 2
        c_param = (c_n_param + c_p_param) / 2
    
    # Create input arrays for each delay data position
    # Each delay data corresponds to a specific (row, col) position in the 2D values array
    input_arrays = []
    
    if len(values_2d) > 0 and len(values_2d[0]) > 0:
        rows = len(values_2d)
        cols = len(values_2d[0])
        
        for row_idx in range(rows):
            for col_idx in range(cols):
                # Get corresponding index values for this position
                index_1_val = index_1[row_idx] if row_idx < len(index_1) else 0
                index_2_val = index_2[col_idx] if col_idx < len(index_2) else 0

                # Create 9-feature input array:
                # [voltage, additional_dim, delay_indicator, index_1_val, index_2_val, a_param, b_param, c_param, temperature]
                input_array = [
                    a_param,        # Feature 0 (new)
                    b_param,        # Feature 1 (new)
                    c_param,        # Feature 2 (new)
                    temperature,     # Feature 3 (new)
                    voltage,        # Feature 4
                    additional_dim, # Feature 5
                    delay_indicator,# Feature 6
                    index_1_val,    # Feature 7
                    index_2_val    # Feature 8


                ]

                input_arrays.append(input_array)
    else:
        # Fallback for empty values
        input_arrays = [[
            a_param, b_param, c_param, temperature,
            voltage, additional_dim, delay_indicator, 0, 0
        ]]
    
    # Return multiple samples - each with one delay value and corresponding index values
    # This creates individual samples for each delay position
    transformed_samples = []
    
    for i, (input_array, output_value) in enumerate(zip(input_arrays, flat_values)):
        sample_dict = {
            'input': input_array,
            'output': output_value  # Single output value, not a list
        }
        transformed_samples.append(sample_dict)
    
    return transformed_samples

def transform_all_samples(flatten, cap, lib_prefix="", abc_params=None):
    """
    Transform all samples for invbuf files
    abc_params: dict with 'a', 'b', 'c' keys containing parameter values
    """
    # step 1: 전체 cell type과 delay_type 추출
    cell_types = sorted(list(set(s['cell'] for s in flatten)))
    port_nums = sorted(list(set(str(s['input_port_num']) for s in flatten)))
    port_nums = list(filter(lambda x: x != '', port_nums))
    size_types = sorted(list(set(s['size'] for s in flatten)))
    delay_types = ['cell_fall', 'cell_rise', 'fall_transition', 'rise_transition']
    
    # step 2: 각 sample 변환
    transformed_list = []
    for s in flatten:
        if s["type"] == 'timing':
            transformed_samples = transform_sample(s, cap, cell_types, delay_types, size_types, port_nums, lib_prefix, abc_params)
            
            # Now transform_sample returns a list of samples, add them all
            if transformed_samples:
                transformed_list.extend(transformed_samples)

    return transformed_list