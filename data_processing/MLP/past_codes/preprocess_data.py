import torch
import sys

# shell에서 인자 받기
tag = sys.argv[1]            # ex: 'simpleinvbuf'
start_nodes = int(sys.argv[2]) # ex: 7
num_nodes = int(sys.argv[3])
prefix = sys.argv[4]         # ex: '2_25'

data_input_list = []
data_output_list = []

# 각 input/output .pth 파일을 반복적으로 로딩
for i in range(start_nodes + 1,start_nodes + num_nodes + 1):
    in_path = f"{prefix}_{tag}input_{i}nodes_align.pth"
    out_path = f"{prefix}_{tag}output_{i}nodes_align.pth"

    print(f"Loading {in_path}, {out_path}")
    data_input_list.append(torch.load(in_path))
    data_output_list.append(torch.load(out_path))

# concat
data_input = torch.cat(data_input_list, dim=0)
data_output = torch.cat(data_output_list, dim=0)

# 정규화
voltage_mean = data_input[:, :, 0].mean()
voltage_std = data_input[:, :, 0].std()
data_input = ((data_input[:, :, 0] - voltage_mean) / voltage_std).unsqueeze(2)

# 장치 전송
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_input = data_input.to(device)
data_output = data_output.to(device)

# 확인
print("Final tensor shapes:", data_input.shape, data_output.shape)
print("Output value range:", data_output.min().item(), "~", data_output.max().item())
