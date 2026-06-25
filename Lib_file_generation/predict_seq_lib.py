"""Parameterized seq cell delay/transition/constraint prediction driver.

Constraint uses inner_lr=3e-3 (lr-optimized; was the separate lr3e3 pass
before consolidation). Delay/transition stay at default inner_lr=3e-4.

Usage:
    python predict_seq_lib.py <corner> <temp> <gpu_id>

Examples:
    python predict_seq_lib.py FF 0 0
    python predict_seq_lib.py SS 100 3
"""
import os, sys, copy, shutil

if len(sys.argv) != 4:
    print("Usage: predict_seq_lib.py <corner> <temp> <gpu_id>")
    print("       corner in {FF,TT,SS,SF,FS}; temp in {0,25,50,75,100}; gpu_id in {0,1,2,3}")
    sys.exit(1)

CORNER = sys.argv[1]
TEMP   = int(sys.argv[2])
GPU_ID = sys.argv[3]

assert CORNER in ('FF','TT','SS','SF','FS'), f'bad corner {CORNER}'
assert TEMP in (0, 25, 50, 75, 100), f'bad temp {TEMP}'

sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/Lib_file_generation')
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code')
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import gc
import torch
from predict_comb_lib import (
    CellTestDataset, get_task_indices_for_pvt,
    run_predictions_with_adaptation,
    run_predictions_with_lib_support, extract_lib_support_data,
    run_predictions_with_lib_support_seq_delay,
    LibFileParser, update_lib_file_for_cell,
    cache_minimal_cell_meta,
)
from gnn_maml import create_maml_gcn_model

SEQ_CELLS = ['DFCNQD1BWP30P140','SDFSNQD0BWP30P140','SDFCSNQD1BWP30P140']
CONSTRAINT_CATS = ('setup', 'hold', 'recovery', 'removal',
                   'non_seq_setup', 'non_seq_hold')

DATASET_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC'
LOOPCLOSE_CACHE = '/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/stage_aware_topology_cache_tsmc_tcbn28hpcplusbwp30p140_110a_lpe_typical_loopclose.pth'

SEQ_LIB_DIR = f'/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files/TSMC_seq_cell/TSMC_{CORNER}seq_{TEMP}'
LIB_FNAME   = f'TSMC_{CORNER}_Seq_{TEMP}_{{v:03d}}.lib'

OUT_DIR = f'/home/tkdgn2907/Deepsets_test/MAML/Projects/Lib_file_generation/predicted_seq/{CORNER}_{TEMP}'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda')

class A: pass
args = A()
args.mode = 'interpolation'
args.adaptation_method = 'selective_adam'
args.graph_mode = 'stage_aware'

print(f'### {CORNER}/{TEMP} on GPU {GPU_ID} — loopclose + _x10 strict pipeline ###')
print(f'### Loading loopclose topology cache')
topology_cache = torch.load(LOOPCLOSE_CACHE, weights_only=False, map_location='cpu', mmap=True)

train_pth = f'{DATASET_DIR}/train_cell_stage_aware.pth'
train_data = torch.load(train_pth, weights_only=False, map_location='cpu', mmap=True)
norm_stats = copy.deepcopy(train_data['norm_stats'])
# output_load → input_slew alias for constraint is auto-applied inside
# run_predictions_with_adaptation when data_type ∈ CONSTRAINT_CATS.

model_pth_cell  = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/gnn_maml_tsmc_process_checkpoints/gnn_maml_tsmc_process_cell_stage_aware_innerdiv10_meta16_iter300000_inner1_conv64x2_fc256x2.pth'
model_pth_trans = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/gnn_maml_tsmc_process_checkpoints/gnn_maml_tsmc_process_transition_stage_aware_innerdiv10_meta16_iter300000_inner1_conv64x2_fc256x2_pooloutput.pth'

def load_model(pth, pooling='mean'):
    ckpt = torch.load(pth, weights_only=False, map_location=device)
    nfd = ckpt['model_state_dict']['convs.0.lin.weight'].shape[1]
    m = create_maml_gcn_model(node_features=nfd, pooling=pooling, output_dim=1, dropout=0.0,
        conv_hidden_dim=64, num_conv_layers=2, fc_hidden_dim=256, num_fc_layers=2).to(device)
    m.load_state_dict(ckpt['model_state_dict']); m.eval()
    return m

args.inner_lr = 3e-3   # lr-optimized constraint pass
print(f'\n### Constraint predictions @ inner_lr={args.inner_lr} ###')
all_constraint_preds = {}
constraint_model = load_model(model_pth_cell, pooling='mean')

for cat in CONSTRAINT_CATS:
    print(f'\n=== {cat} ===')
    for cell in SEQ_CELLS:
        pth = f'{DATASET_DIR}/test_by_{cat}_stage_aware/{cell}.pth'
        if not os.path.exists(pth): print(f'  {cell}: no PTH'); continue
        ds = CellTestDataset(pth)
        fidx = get_task_indices_for_pvt(ds, CORNER, float(TEMP))
        print(f'  {cell}: {len(fidx)} {CORNER}/{TEMP} tasks')
        preds = run_predictions_with_adaptation(
            constraint_model, ds, topology_cache, norm_stats, device, args,
            task_indices=fidx, data_type=cat)
        # Cache only the small per-task metadata the write phase needs and drop
        # the heavy PTH tensors — keeps total RSS ~flat over 3 cells × 6 cats.
        ds_cached = cache_minimal_cell_meta(ds, preds)
        all_constraint_preds.setdefault(cell, {})[cat] = (ds_cached, preds)
        del ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

args.inner_lr = 3e-4   # back to default for delay/transition
print(f'\n### Cell delay + transition (loopclose PTH + _x10 lib_few_shot) @ inner_lr={args.inner_lr} ###')
all_delay_preds = {}
for dt in ('cell','transition'):
    print(f'\n=== {dt} ===')
    model = load_model(model_pth_trans if dt=='transition' else model_pth_cell,
                       pooling='output' if dt=='transition' else 'mean')
    for cell in SEQ_CELLS:
        pth = f'{DATASET_DIR}/test_by_{dt}_stage_aware/{cell}.pth'
        if not os.path.exists(pth): print(f'  {cell}: no PTH'); continue
        ds = CellTestDataset(pth)
        fidx = get_task_indices_for_pvt(ds, CORNER, float(TEMP))
        print(f'  {cell} {dt}: {len(fidx)} tasks')
        lib_support = extract_lib_support_data(
            lib_dir='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files',
            corner=CORNER, temperature=float(TEMP), cell_name=cell,
            support_indices=[0,13,30,45,60],
            seq_lib_dir='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files/TSMC_seq_cell',
            seq_folder_suffix='',
        )
        if not lib_support: print(f'  {cell}: support failed'); continue
        preds = run_predictions_with_lib_support_seq_delay(
            model, ds, topology_cache, norm_stats, device, args,
            lib_support_data=lib_support, data_type=dt, task_indices=fidx)
        # Drop the heavy PTH; keep just the small per-task lookup cache.
        ds_cached = cache_minimal_cell_meta(ds, preds)
        all_delay_preds.setdefault(cell, {})[dt] = (ds_cached, preds)
        del ds, lib_support
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f'\n### Writing predictions into {OUT_DIR} ###')
for vi in range(61):
    v = 60 + vi
    src = os.path.join(SEQ_LIB_DIR, LIB_FNAME.format(v=v))
    dst = os.path.join(OUT_DIR, f'predicted_{LIB_FNAME.format(v=v)}')
    shutil.copy2(src, dst)
    parser = LibFileParser(dst)
    total_m = total_um = 0
    for cell, perd in all_constraint_preds.items():
        for cat, (ds, preds) in perd.items():
            n, _, m, um = update_lib_file_for_cell(parser, cell, preds, ds,
                                                    lib_idx=vi, data_type=cat,
                                                    collect_comparison=False)
            total_m += m; total_um += um
    for cell, perd in all_delay_preds.items():
        for dt, (ds, preds) in perd.items():
            n, _, m, um = update_lib_file_for_cell(parser, cell, preds, ds,
                                                    lib_idx=vi, data_type=dt,
                                                    collect_comparison=False)
            total_m += m; total_um += um
    parser.save(dst)
    if vi % 10 == 0 or vi == 60:
        print(f'  V={v/100:.2f}: matched={total_m}, unmatched={total_um}')
print(f'\nDONE {CORNER}/{TEMP}')
