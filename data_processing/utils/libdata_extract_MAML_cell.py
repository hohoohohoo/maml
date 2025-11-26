import re
import pandas as pd

def parse_liberty_pin_blocks(lines):
    pins = []
    current_cell_name = None
    current_pin = None
    current_timing = None
    delay_type = None
    power_type = None
    inside_powertype = False
    power_values = []
    delay_values = []
    inside_pin = False
    inside_timing = False
    inside_internal_power = False
    inside_delaytype = False
    inside_values_block = False
    inside_values_block1 = False
    inside_PVT_condition = False
    for line in lines:
        line = line.strip()
        # PVT condition input
        if line.startswith("operating_conditions"):
            inside_PVT_condition = True
        
        if inside_PVT_condition:
            if "process" in line:
                match = re.search(r'process\s*:\s*(\d+)', line)
                if match:
                    process_value = int(match.group(1))
            if "temperature" in line:
                match = re.search(r'temperature\s*:\s*([-+]?\d+)', line)
                if match:
                    temp_value = int(match.group(1))
            if "voltage" in line:
                match = re.search(r'voltage\s*:\s*([-+]?\d*\.?\d+)', line)
                if match:
                    voltage_value = float(match.group(1))

        if inside_PVT_condition and line == "}":
            inside_PVT_condition = False

        # cell 블록 시작
        if line.startswith("cell"):
            match = re.search(r'cell\s*\((.*?)\)', line)
            if match:
                full_cell_name = match.group(1)  # 예: AND2x2_ASAP7_75t_L
                # 전체 cell name을 사용해서 정확한 topology 매핑
                #logic_type = re.match(r'[A-Z]+\d*', full_cell_name)  # A-Z와 숫자까지 포함
                #print(f"Full cell name: {full_cell_name}")  # 디버그용
                current_cell_name = full_cell_name  # 전체 이름 사용
                #size_match = re.search(r'[xX](\d+(\.\d+)?)', full_cell_name)  # 정수 또는 소수 모두 처리
                input_port_num = re.search(r'(\d+)(?=[xX])', full_cell_name)
                if(input_port_num is not None):
                    input_port_num = int(input_port_num.group(1))
                size_match = re.search(r'[xX](\d+)(p(\d+))?|p(\d+)', full_cell_name)    # p 뒤에 숫자도 처리
                #cell_size = size_match.group(1) if size_match else None
                if size_match:
                    #print(size_match.group(0),size_match.group(1),size_match.group(3))
                    if (size_match.group(2)):  # p 뒤에 숫자가 있으면
                        if(float(size_match.group(3))>=10):
                            cell_size = str(float(size_match.group(1))+ float(size_match.group(3)) / 100)  # p33 -> 0.33 
                        else:
                            cell_size = str(float(size_match.group(1))+ float(size_match.group(3)) / 10) 
                    elif 'p' in size_match.group(0):
                        size_match = re.search(r'p(\d+)', full_cell_name)
                        #print(size_match)
                        cell_size = size_match.group(0)
                        #print(cell_size)
                        if(float(size_match.group(1))>=10):
                            cell_size = str(float(size_match.group(1)) / 100)  
                        else:
                            cell_size = str(float(size_match.group(1)) / 10) 
                    else:
                        cell_size = size_match.group(1)  # 정수만 있는 경우
                else:
                    cell_size = None
                #print(size_match,cell_size)
                #print(cell_size)
        # pin 시작
        if line.startswith("pin"):
            inside_pin = True
            current_pin = {}
            match = re.search(r'pin\s*\((.*?)\)', line)
            current_pin['cell']=current_cell_name
            #print(current_cell_name)
            current_pin['size']=cell_size
            current_pin["input_port_num"] = input_port_num
            #print(cell_size)
            if match:
                current_pin['pin_name'] = match.group(1)
        
        # timing 블록 시작
        elif inside_pin and line.startswith("timing()"):
            inside_timing = True
            current_timing = {}
            #print(current_timing)
        elif inside_pin and line.startswith("internal_power()"):
            inside_internal_power = True
            current_internal = {}

        elif inside_values_block:
            delay_values.extend(re.findall(r'"(.*?)"', line))
            if line.endswith(");") or ");" in line:
            #if line == "}" in line:
                #current_timing["related_pin"]
                inside_values_block = False
                #print(inside_delaytype)
                #print(inside_timing , line, delay_values)
        elif inside_values_block1:
            power_values.extend(re.findall(r'"(.*?)"', line))
            if line.endswith(");") or ");" in line:
            #if line == "}" in line:
                inside_values_block1 = False

        elif inside_powertype and line == "}":
            if power_values:
                    #print(power_values)
                    current_internal[power_type] = [
                        list(map(float, row.split(','))) for row in power_values
                    ]
                    #print(current_internal)
                    #print(current_timing)
                    power_values = []
            inside_powertype = False

        elif inside_delaytype and line == "}":
            if delay_values:
                    #print('hi')
                    #print(delay_values)
                    current_timing[delay_type] = [
                        list(map(float, row.split(','))) for row in delay_values
                    ]
                    #print(current_timing["related_pin"])
                    delay_values = []
            inside_delaytype = False
            #print(current_timing)
        elif inside_timing and line == "}":
            #print(delay_values)
            if current_timing and current_pin is not None:
                #print(current_timing)
                current_pin.setdefault("timings", []).append(current_timing)
                #print(current_timing)
                current_timing = {}
            inside_timing = False   

        elif inside_internal_power and line == "}":
            if current_internal and current_pin is not None:
                current_pin.setdefault("internal_powers", []).append(current_internal)
            current_internal = {}
            inside_internal_power = False

        # pin 종료
        elif inside_pin and line == "}":
            if current_pin:
                current_pin["Process"] = process_value
                current_pin["Temperature"] = temp_value
                current_pin["Voltage"] = voltage_value
                #print(current_pin)
                pins.append(current_pin)
            current_pin = None
            inside_pin = False

        # timing 내부 처리
        elif inside_timing:
            if "related_pin" in line:
                #print(list(set(ports)))
                current_timing["related_pin"] = re.search(r'"(.*?)"', line).group(1)
            #elif "timing_sense" in line:
            #    current_timing["timing_sense"] = line.split(":")[1].strip().strip(';')
            #elif "timing_type" in line:
            #    current_timing["timing_type"] = line.split(":")[1].strip().strip(';')
            
            elif any(dt in line for dt in ["cell_rise","cell_fall","rise_transition","fall_transition"]):
                #print(current_timing)
                inside_delaytype = True
                delay_types = ["cell_rise","cell_fall","rise_transition","fall_transition"]
                for dt in delay_types:
                    if dt in line:
                        delay_type =dt

            elif "index_1" in line and inside_delaytype:
                # Skip first number (1 from "index_1") by slicing [1:]
                current_timing["index_1"] = list(map(float, re.findall(r'[\d.]+', line)[1:]))
            elif "index_2" in line and inside_delaytype:
                # Skip first number (2 from "index_2") by slicing [1:]
                current_timing["index_2"] = list(map(float, re.findall(r'[\d.]+', line)[1:]))
            elif "values" in line and inside_delaytype:
                inside_values_block = True
                delay_values.extend(re.findall(r'"(.*?)"', line))
                #delay_values = re.findall(r'"(.*?)"', line)

        elif inside_internal_power:
            if "related_pin" in line:
                current_internal["related_pin"] = re.search(r'"(.*?)"', line).group(1)
            elif "related_pg_pin" in line:
                current_internal["related_pg_pin"] = re.search(r'(.*?)', line).group(1)    
            elif "when" in line:
                current_internal["when"] = re.search(r'"(.*?)"', line).group(1)
            elif any(dt in line for dt in ["rise_power", "fall_power"]):
                inside_powertype = True
                power_types = ["rise_power","fall_power"]
                for dt in power_types:
                    if dt in line:
        #                #print(dt)
                        power_type =dt
            elif "values" in line  and inside_powertype:
                #print(line)
                if not line.endswith(");"):
                    inside_values_block1 = True
                power_values.extend(re.findall(r'"(.*?)"', line))

        # 일반 pin 속성
        elif inside_pin and ":" in line and not (inside_timing or inside_internal_power):
            key, val = line.split(":", 1)
            val_clean = val.strip().strip(';').strip('"')
            current_pin[key.strip()] = val_clean

            # ✅ function 항목이라면 input port 이름 파싱
            if key.strip() == "function":
                # A, B, A1, A2 등 알파벳+숫자 조합된 식별자 추출
                # 예: (A1 * A2 + B) → ['A1', 'A2', 'B']
                ports = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', val_clean)
                # 논리 키워드 또는 예약어는 제외 (예: VDD, VSS, 1, 0 등)
                exclude = {"VDD", "VSS", "1", "0"}
                ports = [p for p in ports if p not in exclude and not p.isdigit()]
                #print(ports)
                current_pin["input_port_names"] = list(set(ports))
                #current_pin["input_port_num"] = input_port_num


    return pins


# 파일 읽기
#with open("OA_LVT_2_25_070.lib", "r") as f:
#        lines = f.readlines()
# 파싱 실행
#pin_data = parse_liberty_pin_blocks(lines)

# 첫 pin 블록 확인
from pprint import pprint
#pprint(pin_data)

def flatten_pin_data(pin_data):
    rows = []
    cap = []
    for pin in pin_data:
        Process = pin.get("Process","")
        Temperature = pin.get("Temperature","")
        voltage = pin.get("Voltage","")
        cell_name = pin.get("cell","")
        cell_size = pin.get("size","")
        pin_name = pin.get("pin_name", "")
        direction = pin.get("direction","")
        related_pg_pin = pin.get("related_pg_pin","")
        input_port_num = pin.get("input_port_num","")
        input_port_name = pin.get("input_port_names","")
        input_port_name = sorted(input_port_name)
       # related_power_pin = pin.get("related_power_pin","")
        max_transition = pin.get("max_transition","")
        capacitance = pin.get("capacitance","")
        rise_capacitance = pin.get("rise_capacitance","")
        fall_capacitance = pin.get("fall_capacitance","")
        #print(input_port_num)

        # internal_power 항목 정리
        for ip in pin.get("internal_powers", []):
            for dtype in ["rise_power", "fall_power"]:
                if dtype in ip:
                    rows.append({
                        "Process":Process,
                        "Temperature":Temperature,
                        "Voltage":voltage,
                        "cell":cell_name,
                        "size":cell_size,
                        "type":"internal_powers",
                        "pin_name": pin_name,
                        #"direction": direction,
                        #"related_pg_pin": related_pg_pin,
                        "input_port_num": input_port_num,
                        "input_port_name":input_port_name,
                        #"related_power_pin": related_power_pin,
                        #"max_transition":max_transition,
                        "capacitance":capacitance,
                        "rise_capacitance":rise_capacitance,
                        "fall_capacitance":fall_capacitance,
                        #"related_pin": ip.get("related_pin", ""),
                        #"when": ip.get("when", ""),
                        #"power": "",
                        #"timing_type": ip.get("timing_type", ""),
                        #"timing_sense": ip.get("timing_sense", ""),
                        "delay_type": dtype,
                        "values": ip[dtype],  # 2D 리스트
                        "index_1": ip.get("index_1", []),
                        "index_2": ip.get("index_2", [])
                    }) 
                if(capacitance):
                    cap.append({
                        "cell":cell_name,
                        "size":cell_size,
                        "input_port_num": input_port_num,
                        "pin_name":pin_name,
                        "capacitance":capacitance,
                        "rise_capacitance":rise_capacitance,
                        "fall_capacitance":fall_capacitance
                    })

        # timing 항목 정리
        for t in pin.get("timings", []):
            #for dtype in [ "rise_transition", "fall_transition"]:
            for dtype in ["cell_rise", "cell_fall"]:
                if dtype in t:
                    rows.append({
                        "Process":Process,
                        "Temperature":Temperature,
                        "Voltage":voltage,
                        "cell":cell_name,
                        "size":cell_size,
                        "type":"timing",
                        "pin_name": pin_name,
                        #"direction": direction,
                        #"related_pg_pin": related_pg_pin,
                        "input_port_num": input_port_num,
                        "input_port_name":input_port_name,
                        #"related_power_pin": related_power_pin,
                        #"max_transition":max_transition,
                        "capacitance":capacitance,
                        #"rise_capacitance":rise_capacitance,
                        #"fall_capacitance":fall_capacitance,
                        "related_pin": t.get("related_pin", ""),
                        #"when": t.get("when",""),
                        #"power": "",
                        #"timing_type": t.get("timing_type", ""),
                        #"timing_sense": t.get("timing_sense", ""),
                        "delay_type": dtype,
                        "values": t[dtype],  # 2D 리스트
                        "index_1": t.get("index_1", []),
                        "index_2": t.get("index_2", [])
                    })
                    #print(rows["related_pin"])
    return rows ,cap
    #return rows

