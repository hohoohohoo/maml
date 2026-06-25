"""25 PT sweep coordinator — comb + seq + merge for all (corner, temp) pairs.

Distribution: 4 GPU × ~6-7 PT each.
Per PT: comb (main script, lib_few_shot supportgrad) → seq (parameterized driver,
strict + order-based + loopclose + _x10; constraint @ inner_lr=3e-3, delay @ 3e-4)
→ merge into final lropt lib.

Usage:
    nohup python run_lib_generation.py > run_lib_generation.log 2>&1 &

Outputs (per PT):
    predicted_comb_sweep/maml_conv64x2_fc256x2_all_stage_aware_5shot/all_voltages_<corner>_<temp>.0/
    predicted_seq/<corner>_<temp>/
    predicted_lropt/predicted_<corner>_<temp>_all_predictions_lropt/

Idempotent: skips a phase if output dir exists and looks complete (61 .lib files).
"""
import os, sys, subprocess, time, shutil
from multiprocessing import Pool

LIB_GEN_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/Lib_file_generation'
# Filter via env var CORNERS_FILTER (CSV) — e.g. CORNERS_FILTER=FF,SF,FS for split runs.
_FILTER = os.environ.get('CORNERS_FILTER', '').strip()
CORNERS = tuple(_FILTER.split(',')) if _FILTER else ('FF', 'TT', 'SS', 'SF', 'FS')
TEMPS = (0, 25, 50, 75, 100)
ALL_PT = [(c, t) for c in CORNERS for t in TEMPS]
N_GPUS = int(os.environ.get('N_GPUS', 4))
WORKERS_PER_GPU = int(os.environ.get('WORKERS_PER_GPU', 1))   # 1 = no oversubscribe (safer for memory)
N_WORKERS = N_GPUS * WORKERS_PER_GPU

# Comb output base (mirrors supportgrad naming)
COMB_OUT_BASE = f'{LIB_GEN_DIR}/predicted_comb_sweep'
COMB_SUBDIR_FMT = 'maml_conv64x2_fc256x2_all_stage_aware_5shot/all_voltages_{corner}_{temp_f}'

# Seq output base (matches predict_seq_lib.py's OUT_DIR)
SEQ_OUT_BASE = f'{LIB_GEN_DIR}/predicted_seq'

# Final merged
FINAL_FMT = f'{LIB_GEN_DIR}/predicted_lropt/predicted_{{corner}}_{{temp}}_all_predictions_lropt'

# Comb main script args.  Use _constraint.py (canonical, all fixes).
# Per-PT invocation: pass --corners X --temps Y to scope main script's
# --all_corners_temps loop to that single PT.
COMB_SCRIPT = f'{LIB_GEN_DIR}/predict_comb_lib.py'
COMB_DATASET = '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC'
COMB_LIB_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files'

SEQ_SCRIPT = f'{LIB_GEN_DIR}/predict_seq_lib.py'

# ---------- helpers ----------

def comb_out_dir(corner, temp):
    sub = COMB_SUBDIR_FMT.format(corner=corner, temp_f=f'{temp}.0')
    return f'{COMB_OUT_BASE}/{sub}'

def seq_out_dir(corner, temp):
    return f'{SEQ_OUT_BASE}/{corner}_{temp}'

def final_out_dir(corner, temp):
    return FINAL_FMT.format(corner=corner, temp=temp)

def n_libs(d):
    if not os.path.exists(d): return 0
    return sum(1 for f in os.listdir(d) if f.endswith('.lib'))

def comb_done(corner, temp):
    return n_libs(comb_out_dir(corner, temp)) == 61

def seq_done(corner, temp):
    return n_libs(seq_out_dir(corner, temp)) == 61

def final_done(corner, temp):
    return n_libs(final_out_dir(corner, temp)) == 61

def run(cmd, log_path, env=None):
    print(f'  → {" ".join(cmd[:6])} ... > {os.path.basename(log_path)}', flush=True)
    # Force GNU OpenMP runtime so libgomp + Intel MKL don't conflict — without
    # this, late-loading MKL hits "incompatible with libgomp.so.1" inside
    # multiprocessing workers that the parent loaded libgomp into first.
    sub_env = dict(env) if env else dict(os.environ)
    sub_env.setdefault('MKL_THREADING_LAYER', 'GNU')
    sub_env.setdefault('MKL_SERVICE_FORCE_INTEL', '0')
    with open(log_path, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=sub_env)
    return rc

# ---------- per-PT pipeline ----------

def run_pt(args):
    corner, temp, gpu_id = args
    tag = f'[{corner}/{temp}|gpu{gpu_id}]'
    log_dir = f'{LIB_GEN_DIR}/sweep_logs'
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    print(f'{tag} START ({time.strftime("%H:%M:%S")})', flush=True)

    # 1) Comb
    if comb_done(corner, temp):
        print(f'{tag} comb already complete — skip', flush=True)
    else:
        cmd = [
            'python', '-u', COMB_SCRIPT,
            '--lib_few_shot',
            '--all_cells',
            '--data_type', 'all',
            '--graph_mode', 'stage_aware',
            '--mode', 'interpolation',
            '--adaptation_method', 'selective_adam',
            '--all_voltages',
            '--all_corners_temps',
            '--corners', corner,
            '--temps', str(temp),
            '--dataset_dir', COMB_DATASET,
            '--lib_dir', COMB_LIB_DIR,
            '--output_dir', COMB_OUT_BASE,
            '--gpu', str(gpu_id),
        ]
        log = f'{log_dir}/comb_{corner}_{temp}.log'
        rc = run(cmd, log)
        if rc != 0:
            print(f'{tag} COMB FAILED rc={rc} — see {log}', flush=True)
            return (corner, temp, 'comb_failed', time.time()-t0)
        print(f'{tag} comb done ({n_libs(comb_out_dir(corner,temp))} libs)', flush=True)

    # 2) Seq
    if seq_done(corner, temp):
        print(f'{tag} seq already complete — skip', flush=True)
    else:
        cmd = ['python', '-u', SEQ_SCRIPT, corner, str(temp), str(gpu_id)]
        log = f'{log_dir}/seq_{corner}_{temp}.log'
        rc = run(cmd, log)
        if rc != 0:
            print(f'{tag} SEQ FAILED rc={rc} — see {log}', flush=True)
            return (corner, temp, 'seq_failed', time.time()-t0)
        print(f'{tag} seq done ({n_libs(seq_out_dir(corner,temp))} libs)', flush=True)

    # 3) Merge
    if final_done(corner, temp):
        print(f'{tag} merge already complete — skip', flush=True)
    else:
        # In-process merge; no GPU needed.
        sys.path.insert(0, LIB_GEN_DIR)
        from predict_comb_lib import merge_libs
        cdir = comb_out_dir(corner, temp); sdir = seq_out_dir(corner, temp)
        odir = final_out_dir(corner, temp); os.makedirs(odir, exist_ok=True)
        n = 0
        for vi in range(61):
            v = 60 + vi
            cp = f'{cdir}/predicted_TSMC_{corner}_{temp}_{v:03d}.lib'
            sp = f'{sdir}/predicted_TSMC_{corner}_Seq_{temp}_{v:03d}.lib'
            op = f'{odir}/predicted_TSMC_{corner}_{temp}_{v:03d}.lib'
            if not os.path.exists(cp) or not os.path.exists(sp):
                print(f'{tag} V={v/100:.2f}: missing source — skip', flush=True)
                continue
            merge_libs(cp, sp, op); n += 1
        print(f'{tag} merge done ({n}/61 files → {odir})', flush=True)

    dt = time.time() - t0
    print(f'{tag} ALL DONE in {dt/60:.1f}min', flush=True)
    return (corner, temp, 'ok', dt)

# ---------- main: distribute 25 PT round-robin across 4 GPUs ----------

def assign_workers(pt_list, n_workers, workers_per_gpu):
    """Round-robin: PT i → worker (i % n_workers). Worker w runs on GPU (w // workers_per_gpu)."""
    return [(c, t, w % n_workers, (w % n_workers) // workers_per_gpu)
            for w, (c, t) in enumerate(pt_list)]

def gpu_worker(jobs):
    """Run one PT job at a time, sequentially, on the assigned GPU."""
    results = []
    for c, t, w, g in jobs:
        results.append(run_pt((c, t, g)))
    return results

if __name__ == '__main__':
    assigned = assign_workers(ALL_PT, N_WORKERS, WORKERS_PER_GPU)
    by_worker = {w: [] for w in range(N_WORKERS)}
    for c, t, w, g in assigned:
        by_worker[w].append((c, t, w, g))
    print(f'=== 25 PT sweep — {N_WORKERS} workers ({WORKERS_PER_GPU} per GPU × {N_GPUS} GPU) ===')
    for w, lst in by_worker.items():
        gpu = w // WORKERS_PER_GPU
        print(f'  worker {w} (GPU {gpu}): {len(lst)} PT — {", ".join(f"{c}/{t}" for c,t,_,_ in lst)}')
    print(flush=True)

    with Pool(N_WORKERS) as pool:
        all_results = pool.map(gpu_worker, [by_worker[w] for w in range(N_WORKERS)])

    print('\n=== 25 PT sweep — SUMMARY ===')
    ok = fail = 0
    for batch in all_results:
        for c, t, status, dt in batch:
            line = f'  {c}/{t:3d}: {status:12s} ({dt/60:5.1f} min)'
            print(line)
            if status == 'ok': ok += 1
            else: fail += 1
    print(f'\nTOTAL: {ok}/25 ok, {fail} failed')
