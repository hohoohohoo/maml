import pandas as pd

def one_hot_encode(index, length):
    return [1 if i == index else 0 for i in range(length)]
    
def get_related_pin_index(related_pin, input_port_names):
    try:
        return input_port_names.index(related_pin)
    except ValueError:
        return -1  # related_pin이 리스트에 없을 경우
    
def get_cap(s,cap):
    for c in cap:
        if s['cell'] == c['cell'] and s['size'] == c['size'] and s['related_pin'] == c['pin_name'] and s['input_port_num'] == c['input_port_num']:
            capacitance = c["capacitance"]
            rise_cap = c["rise_capacitance"]
            fall_cap = c["fall_capacitance"]
    if capacitance is not None:
        return capacitance,rise_cap,fall_cap
    else:
        return -1
def transform_sample(sample,cap,cell_types, delay_types, lib_prefix=""):
#def transform_sample(sample,cell_types, delay_types, size_types, port_nums):
    #PVT one-hot

    process = int(sample.get('Process', 0))
    process_types = [1,2,3]
    process_one_hot = one_hot_encode(process-1, len(process_types)) if process != -1 else [0]*len(process_types)
    temperature = int(sample.get('Temperature', 0))
    voltage = float(sample.get('Voltage', 0))
    #input_port_num = int(sample.get('input_port_num', 0))
    cell_index = cell_types.index(sample['cell']) if sample['cell'] in cell_types else -1
    cell_one_hot = one_hot_encode(cell_index, len(cell_types)) if cell_index != -1 else [0]*len(cell_types)
   
    #capacitance,rise_cap,fall_cap = get_cap(sample,cap)
    #capacitance = float(capacitance)
    #rise_cap = float(rise_cap)
    #fall_cap = float(fall_cap)
    #capacitance = (rise_cap+fall_cap)/2
    process = int(sample.get('Process', 0))
    index_1 = sample.get('index_1', [])
    index_2 = sample.get('index_2', [])

    
    # Flatten values and calculate position
    values_2d = sample.get('values', [])
    flat_values = [v for row in values_2d for v in row]
    #print(capacitance,sample["related_pin"],sample['input_port_num'],size)
    # 조합
    transformed = {}
    #for i, v in enumerate(cell_one_hot):
    #    transformed[f"cell_{cell_types[i]}"] = v
    #for i, v in enumerate(delay_one_hot):
     #   transformed[f"delay_type_{delay_types[i]}"] = v
    #print(sample['cell'],delay_index)
    #if(input_port_num == 0 and sample['cell'] == 'AND'):
    #if(cell_index != 0):
        #print(capacitance)
        #print(voltage,capacitance,sample['input_port_num'],sample["related_pin"])
        #print(port_nums)
        #print(sample["related_pin"], sample['cell'], [voltage,size]+delay_one_hot+ [position], flat_values)

        #print(cell_types,size_types)
        #transformed['input'] = [voltage,capacitance,size] + delay_one_hot + cell_one_hot + [position2]

        #one = [1,0]
        #two = [0,1]
        #if(position2 == 0):
            #transformed['input'] = [process,temperature,voltage,size] + cell_one_hot + delay_one_hot + one
        #print(position2 , input_port_num,sample["related_pin"],sample["input_port_name"] )
    #transformed['input'] = [voltage,temperature] + process_one_hot
    
    # Add delay type indicator: fall -> 1, rise -> -1
    delay_indicator = 0
    if 'fall' in sample['delay_type']:
        delay_indicator = 1
    elif 'rise' in sample['delay_type']:
        delay_indicator = -1
    
    # Check if lib_prefix starts with INVBUF or simple and cell ends with OA/AO
    additional_dim = 0
    if (lib_prefix.startswith("INVBUF") or lib_prefix.startswith("simple")or lib_prefix.startswith("TSMC")) and cell_index != -1:
        cell_name = cell_types[cell_index]
        if any(gate in cell_name for gate in ['NAND','ND','NOR','NR','INV','HA', 'MAJI','CKINVDC','IAO','IOA']):
            additional_dim = 1
        elif any(gate in cell_name for gate in ['AND','AN','OR','BUF','FA', 'MAJ','XNOR','XNR','XOR','HB','AO','OA']):
            additional_dim =2  
            #print(f"ho{cell_name}")
        else:
            print(cell_name)
    elif (lib_prefix.startswith("AO") or lib_prefix.startswith("OA")) and cell_index != -1:
        cell_name = cell_types[cell_index]
        if "I" in cell_name:
            additional_dim = 1
        else:
            additional_dim = 2
    #print(additional_dim)
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
                
                if additional_dim > 0:
                    input_array = [voltage, additional_dim, delay_indicator, index_1_val, index_2_val]
                else:
                    input_array = [voltage, delay_indicator]
                
                if additional_dim > 0:
                    input_arrays.append(input_array)
    else:
        # Fallback for empty values
        if additional_dim > 0:
            input_arrays = [[voltage, additional_dim, delay_indicator, 0, 0]]
        else:
            input_arrays = [[voltage, delay_indicator]]
    
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

#transformed['size'] = size
#transformed['input_port_num'] = input_port_num
    #transformed['input'] = [process,temperature,voltage] + cell_one_hot + delay_one_hot + [size,input_port_num,position]
#transformed['input'] = [process,temperature,voltage] + delay_one_hot + [size,input_port_num,position]
    #transformed['input'] = [process,temperature,voltage] + [size,position]
    #transformed['input'] = [voltage,size,delay_index,position]
#transformed['input'] = [voltage,size,position]+ num_one_hot + [delay_index]
    #transformed['input'] = [voltage]+size_one_hot+[position]


def transform_all_samples(flatten,cap,lib_prefix=""):
#def transform_all_samples(flatten):
    # step 1: 전체 cell type과 delay_type 추출
    #data_types = sorted(list(set(s['type'] for s in flatten)))
    cell_types = sorted(list(set(s['cell'] for s in flatten)))
    # port_nums = sorted(list(set(str(s['input_port_num']) for s in flatten)))
    # port_nums = list(filter(lambda x: x !=  '', port_nums))
    # size_types = sorted(list(set(s['size'] for s in flatten)))
    #delay_types = sorted(list(set(s['delay_type'] for s in flatten)))
    #print(delay_types)
    #print(cap)
    delay_types=['cell_fall', 'cell_rise','fall_transition','rise_transition']
    # step 2: 각 sample 변환
    transformed_list = []
    for s in flatten:
        #print(s)
        if(s["type"] == 'timing'):
            transformed_samples = transform_sample(s,cap,cell_types, delay_types, lib_prefix)
            #transformed = transform_sample(s, cell_types, delay_types, size_types, port_nums)
            #print(transformed['input'])
            #print(transformed)
            # Now transform_sample returns a list of samples, add them all
            if(transformed_samples):
                transformed_list.extend(transformed_samples)

    return transformed_list
