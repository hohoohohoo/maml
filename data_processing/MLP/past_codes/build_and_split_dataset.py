# build_and_split_dataset.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML import transform_all_samples
import sys

def filetered_data(data_input,data_output):
    mask = data_input[..., -1] == 0 # 맨 뒤의 index가 0인 경우의 data만 남김
    #mask = torch.all(data_input[..., -3:] == 0, dim=-1) # 맨 뒤에서부터 3개의 index가 모두 0인 경우의 data만 남김
    filtered_input = data_input[mask]
    filtered_output = data_output[mask]
    return filtered_input,filtered_output

def dataextract(text):
    with open(text, "r") as f:
        lines = f.readlines()
    pin_data = parse_liberty_pin_blocks(lines)
    flattened,cap = flatten_pin_data(pin_data)
    dataset= transform_all_samples(flattened,cap)
    return dataset

def data_management(text):
    from pathlib import Path
    file_path = Path(text)
    dataset = dataextract(file_path)
    dataset = libdata(dataset)
    return dataset
    
def build_all_data(start=40, end=101, prefix="OA_LVT_2_25_", save_input="full_input_tensor.pth", save_output="full_output_tensor.pth",data_dir = "OA_LVT"):
    data_input = []
    data_output = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{data_dir}/{prefix}{v_str}.lib"
        print(f"📥 Processing {filename}")
        dataset = data_management(filename)
        data_input.append(dataset.X)
        data_output.append(dataset.Y)
        print(i,dataset.X.size())

    data_input = torch.stack(data_input).permute(1, 0, 2)    # [x, 61, 5]
    data_output = torch.stack(data_output).permute(1, 0, 2)  # [x, 61, 49]
    torch.save(data_input, save_input)
    torch.save(data_output, save_output)
    print(f"✅ Saved: {save_input}, {save_output}")
    return data_input, data_output

def split_and_save_nodewise(input_path, output_path, output_dir, prefix, start_node, end_node):
    Path(output_dir).mkdir(exist_ok=True)
    data_input = torch.load(input_path)[:, :, 0].unsqueeze(2)        # [x,61,1]
    data_output_all = torch.load(output_path)                        # [x,61,49]

    for i in range(start_node, end_node):
        data_output = data_output_all[:, :, i].unsqueeze(2)
        #mean = data_output.mean()
        #std = data_output.std()
        #data_output = (data_output-mean)/std
        input_save = f"{output_dir}/transition_{prefix}input_{i+1}nodes_align.pth"
        output_save = f"{output_dir}/transition_{prefix}output_{i+1}nodes_align.pth"
        print(data_input.size(),data_output.size())
        torch.save(data_input, input_save)
        torch.save(data_output, output_save)
        print(f"💾 Node {i+1} saved → {input_save}, {output_save}")

if __name__ == "__main__":
    output_dir = sys.argv[1]       # e.g., nodewise_pth
    lib_prefix = sys.argv[2]       # e.g., OA_LVT_2_25_
    start_node = int(sys.argv[3])  # 0 
    end_node = int(sys.argv[4])    # 49
    data_dir = sys.argv[5]
    # Step 1: build full dataset from .lib
    build_all_data(start=40, end=101, prefix=lib_prefix,data_dir = data_dir)

    # Step 2: split into nodewise .pth files
    split_and_save_nodewise("full_input_tensor.pth", "full_output_tensor.pth",
                            output_dir=output_dir, prefix=lib_prefix,
                            start_node=start_node, end_node=end_node)
