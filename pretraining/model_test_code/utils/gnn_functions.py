"""
GNN-specific functions for extrapolation testing
Adapted from mlp_functions.py for GCN models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


def _make_train_criterion(asym_alpha=None, pinball_tau=None):
    """Build the inner-loop training criterion.

    Three modes (asym_alpha / pinball_tau are mutually exclusive, pinball wins if both set):

    1) asym_alpha in (0, 1) — Asymmetric MSE (expectile-like, smooth quadratic).
       = 0.5 / None → standard MSE.
       > 0.5 → biased toward over-estimate (safe direction for STA delay).
       Normalized by 2*a*(1-a) so magnitude matches MSE at a=0.5 — lr/wd stay tuned.

    2) pinball_tau in (0, 1) — Pinball / quantile loss (linear, classic L1-asym).
       = 0.5 → equivalent to L1 / MAE.
       > 0.5 → minimizer is the tau-quantile of y → ~tau fraction of preds end up ≥ target.
       loss = (tau * relu(target - pred) + (1 - tau) * relu(pred - target)).mean()

    3) Both None → standard MSE.
    """
    if pinball_tau is not None:
        t = float(pinball_tau)
        def pinball(pred, target):
            err = target - pred                  # +ve = under-pred
            return (t * F.relu(err) + (1.0 - t) * F.relu(-err)).mean()
        return pinball
    if asym_alpha is None or float(asym_alpha) == 0.5:
        return nn.MSELoss()
    a = float(asym_alpha)
    norm = 2.0 * a * (1.0 - a)
    def asym_mse(pred, target):
        diff = pred - target
        w = torch.where(diff < 0,
                        torch.full_like(diff, a),
                        torch.full_like(diff, 1.0 - a))
        return (w * diff.pow(2)).mean() / norm
    return asym_mse


# ---------------- Shared GNN training/eval helpers ----------------

def _to_device_tensor(x, device):
    """Cast x to a device tensor (identity for tensors, wrap-then-cast for scalars)."""
    return x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(device)


def _setup_norm_tensors(y, true_function, std, mean, move, device):
    """Move y/true_function to device (y as column vector); wrap std/mean/move as device tensors."""
    y = y.to(device).view(-1, 1)
    true_function = true_function.to(device)
    return (y, true_function,
            _to_device_tensor(std, device),
            _to_device_tensor(mean, device),
            _to_device_tensor(move, device))


def _detect_gnn_model_type(initial_model):
    """
    Detect model family + node_features dim from a pretrained GNN.

    Returns ("hetero" | "gat" | "gcn", node_features).
    HeteroGNN is identified by input_linears; GAT has 'heads' on plain GCNConv models.
    """
    if hasattr(initial_model, 'input_linears'):
        return 'hetero', initial_model.input_linears[0].weight.shape[1]
    node_features = initial_model.convs[0].lin.weight.shape[1]
    if hasattr(initial_model, 'heads'):
        return 'gat', node_features
    return 'gcn', node_features


def _clone_gnn_model(initial_model, model_type, node_features, device):
    """Build a fresh model matching initial_model's config, load its state, move to device."""
    common_kwargs = dict(
        node_features=node_features,
        conv_hidden_dim=initial_model.conv_hidden_dim,
        num_conv_layers=initial_model.num_conv_layers,
        fc_hidden_dim=initial_model.fc_hidden_dim,
        num_fc_layers=initial_model.num_fc_layers,
        pooling=initial_model.pooling_type,
        output_dim=1,
        dropout=0.0,
    )
    if model_type == 'hetero':
        from hetero_gnn_maml import create_maml_hetero_gnn_model
        model = create_maml_hetero_gnn_model(
            **common_kwargs,
            num_node_types=initial_model.num_node_types,
            conv_type=initial_model.conv_type,
            heads=initial_model.heads if hasattr(initial_model, 'heads') else 4,
        ).to(device)
    elif model_type == 'gat':
        from gnn_maml import create_maml_gat_model
        model = create_maml_gat_model(
            **common_kwargs,
            heads=initial_model.heads,
        ).to(device)
    else:  # gcn
        from gnn_maml import create_maml_gcn_model
        model = create_maml_gcn_model(**common_kwargs).to(device)
    model.load_state_dict(initial_model.state_dict())
    return model


def _get_sample_adjacency(cell_cache, sample, cache_type):
    """Adjacency matrix for a minimal sample, dispatching on cache_type."""
    if cache_type == 'stage_aware':
        output_topo = cell_cache['output_topologies'][sample['output_name']]
        key = 'pull_up' if 'rise' in sample['delay_type'] else 'pull_down'
        return output_topo[key]['adjacency_matrix']
    return cell_cache['adjacency_matrix']


def _make_pyg_data(sample, topology_cache, cache_type, validate=False):
    """
    Build a PyG Data object from a minimal_sample + topology_cache.

    validate=True raises ValueError on missing cell/output and on edge indices
    that exceed the node count.
    """
    cell_name = sample['cell_name']
    if validate and cell_name not in topology_cache:
        raise ValueError(f"Cell {cell_name} not found in topology cache")
    cell_cache = topology_cache[cell_name]
    if (validate and cache_type == 'stage_aware'
            and sample['output_name'] not in cell_cache['output_topologies']):
        raise ValueError(f"Output {sample['output_name']} not found for cell {cell_name}")

    adjacency = _get_sample_adjacency(cell_cache, sample, cache_type)
    node_features = sample['node_features']
    edge_index = adjacency.nonzero().t()

    if validate and edge_index.numel() > 0:
        max_idx = edge_index.max().item()
        num_nodes = node_features.shape[0]
        if max_idx >= num_nodes:
            raise ValueError(
                f"Edge index out of bounds for cell {cell_name}: "
                f"max_idx={max_idx}, num_nodes={num_nodes}"
            )
    return Data(x=node_features, edge_index=edge_index)


def _run_inner_adam_loop(model, X_batch, y_train, train_criterion, K, num_steps, inner_adam_lr,
                         record_losses=None):
    """Inner-loop Adam optimisation on the support batch; append per-step loss if record_losses is given."""
    optimizer = torch.optim.Adam(model.parameters(), lr=inner_adam_lr, weight_decay=1e-4)
    for _ in range(num_steps):
        loss = train_criterion(model(X_batch), y_train) / K
        if record_losses is not None:
            record_losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()


def model_functions_at_training_gnn(initial_model, X_samples, y, true_samples, true_function,
                                   topology_cache, cache_type, norm_stats, normalize_fn,
                                   optim=torch.optim.SGD, lr=0.003, adam_step=0, std=1, mean=10, move=0,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                   asym_alpha=None, pinball_tau=None, inner_adam_lr=3e-4):
    """
    Trains the GNN model on X_samples, y and measures the loss curve.
    For each n in sampled_steps, records model(x_axis) after n gradient updates.
    mode: 'extrapolation' or 'interpolation' - determines whether to calculate left/right extrapolation metrics

    Args:
        initial_model: Pretrained GNN model
        X_samples: List of minimal samples for support set
        y: Support set outputs (normalized)
        true_samples: All 61 minimal samples for evaluation
        true_function: Ground truth outputs for all 61 points
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        norm_stats: Normalization statistics
        normalize_fn: Function to normalize node features
        optim: Optimizer class
        lr: Learning rate
        adam_step: Number of Adam optimization steps if SGD loss is high
        std: Standard deviation for denormalization
        mean: Mean for denormalization
        move: Move parameter for additional normalization
        left_bound: Left boundary for interpolation region
        right_bound: Right boundary for interpolation region
        total_points: Total number of data points (default: 61)
        mode: 'extrapolation' or 'interpolation'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y, true_function, std, mean, move = _setup_norm_tensors(y, true_function, std, mean, move, device)

    # Rebuild a fresh copy of the pretrained model for inner-loop adaptation.
    model_type, node_features = _detect_gnn_model_type(initial_model)
    model = _clone_gnn_model(initial_model, model_type, node_features, device)

    criterion = nn.MSELoss()                       # for eval / metric reporting (fair comparison)
    train_criterion = _make_train_criterion(asym_alpha, pinball_tau)  # for inner-loop training only
    _ = optim(model.parameters(), lr, weight_decay=1e-4)  # signature-compat; adaptation uses Adam below
    adam_condition_triggered = False

    K = len(X_samples)
    losses = []
    outputs = {}

    # Build support batch (validated pyg data) and probe the initial loss.
    support_batch_data = [
        _make_pyg_data(s, topology_cache, cache_type, validate=True) for s in X_samples
    ]
    X_batch = Batch.from_data_list(support_batch_data).to(device)
    loss = train_criterion(model(X_batch), y) / K

    # Adam training if initial loss is still high.
    if loss > 1e-4:
        adam_condition_triggered = True
        _run_inner_adam_loop(model, X_batch, y, train_criterion, K,
                             num_steps=adam_step, inner_adam_lr=inner_adam_lr,
                             record_losses=losses)

    # Evaluate on all total_points using grad/move denormalization.
    total_loss = 0
    total_mape_loss = 0
    total_rmse_loss = 0
    predictions = []
    actual_values = []

    model.eval()
    with torch.no_grad():
        for i in range(total_points):
            sample_data = _make_pyg_data(true_samples[i], topology_cache, cache_type, validate=True)
            sample_batch = Batch.from_data_list([sample_data]).to(device)

            pred_value = ((model(sample_batch).item() - move) * std + mean).item()
            actual_value = ((true_function[i] - move) * std + mean).item()
            predictions.append(pred_value)
            actual_values.append(actual_value)

            loss = criterion((model(sample_batch).item() - move) * std + mean,
                             (true_function[i] - move) * std + mean)

            squared_error = (pred_value - actual_value) ** 2
            # Guard MAPE at zero-valued targets (falls through to 0 contribution).
            mape_loss = (
                abs((pred_value - actual_value) / actual_value)
                if abs(actual_value) > 1e-8 else 0
            )

            total_loss += loss
            total_mape_loss += mape_loss
            total_rmse_loss += squared_error

    import math
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_rmse = math.sqrt(total_rmse_loss / total_points)

    return (model, outputs, losses, avg_total_loss, avg_total_mape, predictions, actual_values,
            adam_condition_triggered, avg_total_rmse)


def adapt_bilinear_residual_gnn(
    initial_model, X_samples, y, true_samples, true_function,
    topology_cache, cache_type, total_points,
):
    """
    Adaptation = bilinear surface prior (from 4 V×T corner support outputs)
                 + scalar residual rescaling (from 1 center support point).

    No model weight updates. Designed for 2-D V×T validation where support
    is exactly 4 corners + 1 center; each X_sample must carry
    'voltage_idx' / 'temp_idx' from CellTestDataset2D.

    Final prediction at any (V, T):
        pred(V, T) = data_prior(V, T) + alpha * (m_raw(V, T) - model_prior(V, T))

    where
        data_prior  : bilinear interp through the 4 corner y values
        model_prior : bilinear interp through the model's raw outputs at the 4 corners
        alpha       : data_resid_at_center / model_resid_at_center

    Returns the same 7-tuple shape as the other branches of
    evaluate_model_performance_gnn so the validation caller is unchanged.
    """
    import math
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initial_model
    model.eval()

    # PyG data factory (unvalidated — bilinear path is downstream of edge-index checks).
    def make_data(sample):
        return _make_pyg_data(sample, topology_cache, cache_type)

    # --- Classify support into 4 corners + 1 center ---
    if not all('voltage_idx' in s and 'temp_idx' in s for s in X_samples):
        raise ValueError(
            "bilinear_residual adaptation requires X_samples to carry "
            "'voltage_idx' and 'temp_idx' (use CellTestDataset2D)."
        )
    vs = [s['voltage_idx'] for s in X_samples]
    ts = [s['temp_idx'] for s in X_samples]
    v_min, v_max = min(vs), max(vs)
    t_min, t_max = min(ts), max(ts)
    if v_min == v_max or t_min == t_max:
        raise ValueError(
            "bilinear_residual needs support points spanning both V and T axes."
        )

    y_corner = {}            # (v_id, t_id) -> y value
    corner_pos = {}          # (v_id, t_id) -> index in X_samples
    center_idx = None
    center_v = center_t = None
    for i, s in enumerate(X_samples):
        v, t = int(s['voltage_idx']), int(s['temp_idx'])
        is_corner = v in (v_min, v_max) and t in (t_min, t_max)
        key = (v, t)
        if is_corner and key not in y_corner:
            y_corner[key] = float(y[i].item())
            corner_pos[key] = i
        elif center_idx is None:
            center_idx = i
            center_v, center_t = v, t

    if len(y_corner) != 4:
        raise ValueError(
            f"bilinear_residual needs exactly 4 distinct V×T corners in support; "
            f"got {len(y_corner)} ({list(y_corner.keys())})."
        )
    if center_idx is None:
        raise ValueError(
            "bilinear_residual needs at least one non-corner support point as center."
        )

    y_00 = y_corner[(v_min, t_min)]
    y_01 = y_corner[(v_min, t_max)]
    y_10 = y_corner[(v_max, t_min)]
    y_11 = y_corner[(v_max, t_max)]
    dv = float(v_max - v_min)
    dt = float(t_max - t_min)

    def bilinear(v, t, y00, y01, y10, y11):
        vn = (v - v_min) / dv
        tn = (t - t_min) / dt
        return ((1 - vn) * (1 - tn) * y00
                + (1 - vn) * tn * y01
                + vn * (1 - tn) * y10
                + vn * tn * y11)

    # --- Model raw outputs at the 4 corners and center ---
    with torch.no_grad():
        m_at = {}
        for key, idx in corner_pos.items():
            batch = Batch.from_data_list([make_data(X_samples[idx])]).to(device)
            m_at[key] = float(model(batch).item())
        c_batch = Batch.from_data_list([make_data(X_samples[center_idx])]).to(device)
        m_center = float(model(c_batch).item())

    m_00 = m_at[(v_min, t_min)]
    m_01 = m_at[(v_min, t_max)]
    m_10 = m_at[(v_max, t_min)]
    m_11 = m_at[(v_max, t_max)]

    # --- alpha (residual rescaling) ---
    data_prior_center  = bilinear(center_v, center_t, y_00, y_01, y_10, y_11)
    model_prior_center = bilinear(center_v, center_t, m_00, m_01, m_10, m_11)
    data_resid_center  = float(y[center_idx].item()) - data_prior_center
    model_resid_center = m_center - model_prior_center

    EPS = 1e-8
    alpha = 0.0 if abs(model_resid_center) < EPS \
            else data_resid_center / model_resid_center

    # --- Predict every (V, T) sample ---
    predictions = []
    actual_values = []
    total_loss = 0.0
    total_mape = 0.0
    total_sse = 0.0
    with torch.no_grad():
        for i in range(total_points):
            s = true_samples[i]
            v = int(s['voltage_idx'])
            t = int(s['temp_idx'])
            batch = Batch.from_data_list([make_data(s)]).to(device)
            m_raw = float(model(batch).item())

            data_prior  = bilinear(v, t, y_00, y_01, y_10, y_11)
            model_prior = bilinear(v, t, m_00, m_01, m_10, m_11)
            pred = data_prior + alpha * (m_raw - model_prior)

            actual = float(true_function[i].item())
            predictions.append(pred)
            actual_values.append(actual)

            err = pred - actual
            total_sse += err * err
            total_loss += err * err   # MSE-style accumulator
            if abs(actual) > 1e-8:
                total_mape += abs(err / actual)

    avg_loss = total_loss / total_points
    avg_mape = total_mape / total_points
    avg_rmse = math.sqrt(total_sse / total_points)

    # Wrap loss as a tensor so callers that use tensor math (** 0.5, isinf, isnan)
    # behave identically to the selective_adam branch.
    return (
        torch.tensor(avg_loss, dtype=torch.float32),
        avg_mape,
        predictions,
        actual_values,
        model,
        False,         # adam_used = False (no weight updates)
        avg_rmse,
    )


def evaluate_model_performance_gnn(initial_model, model_name, X_samples, y, true_samples, true_function,
                                   grad, move, topology_cache, cache_type, norm_stats, normalize_fn,
                                   optim=torch.optim.SGD, lr=0.001,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                   adaptation_method='selective_adam', asym_alpha=None, safe_eps=None,
                                   pinball_tau=None, inner_adam_lr=3e-4):
    """
    Evaluate GNN model performance with grad/move normalization parameters

    Args:
        initial_model: Pretrained GNN model
        model_name: Name of the model (for logging)
        X_samples: Support set samples (minimal format)
        y: Support set outputs
        true_samples: All 61 minimal samples
        true_function: Ground truth outputs for all 61 points
        grad: Gradient scaling parameter
        move: Move parameter for additional normalization
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        norm_stats: Normalization statistics
        normalize_fn: Function to normalize node features
        optim: Optimizer class
        lr: Learning rate
        left_bound: Left boundary for interpolation region
        right_bound: Right boundary for interpolation region
        total_points: Total number of data points (default: 61)
        mode: 'extrapolation' or 'interpolation'
        adaptation_method: 'selective_adam' (grad/move + conditional Adam) or 'adam' (direct Adam, no grad/move)

    Returns:
        tuple: (total_loss, total_mape, predictions, actual_values, model, adam_used, total_rmse)
    """
    import numpy as np

    # Bilinear-prior + residual scaling (2-D V×T). No weight updates.
    if adaptation_method == 'bilinear_residual':
        return adapt_bilinear_residual_gnn(
            initial_model=initial_model,
            X_samples=X_samples,
            y=y,
            true_samples=true_samples,
            true_function=true_function,
            topology_cache=topology_cache,
            cache_type=cache_type,
            total_points=total_points,
        )

    # If using 'adam' method, use direct Adam without grad/move scaling
    if adaptation_method == 'adam':
        result = model_functions_with_optim_mode_gnn(
            initial_model=initial_model,
            X_samples=X_samples,
            y=y,
            true_samples=true_samples,
            true_function=true_function,
            topology_cache=topology_cache,
            cache_type=cache_type,
            norm_stats=norm_stats,
            normalize_fn=normalize_fn,
            optim_mode='adam',
            num_steps=40,
            lr=0.003,
            std=1,
            mean=0,
            move=0,
            grad=1,
            left_bound=left_bound,
            right_bound=right_bound,
            total_points=total_points,
            mode=mode,
            asym_alpha=asym_alpha,
            pinball_tau=pinball_tau,
            inner_adam_lr=inner_adam_lr,
        )

        return (result['total_loss'], result['total_mape'],
                result['predictions'], result['actual_values'], result['model'], True,
                result['total_rmse'])

    # Original selective_adam method with grad/move scaling
    y_mean = y.mean()
    y_std = y.std()
    mean_values = [y_mean]
    std_values = [y_std * grad]

    # Store all predictions and actuals for final plotting
    all_predictions = []
    all_actuals = []

    # For each combination of mean and std
    for mean in mean_values:
        for std in std_values:
            y_mean1 = mean  # Update mean
            y_std1 = std    # Update std

            y_test = (y-y_mean1) / y_std1 + move
            true_function1 = (true_function-y_mean1) / y_std1 + move

            # Safe-margin shift: nudge ONLY the support-training target up by safe_eps
            # (in normalized units). The model fits this shifted target during inner-loop
            # adaptation, so its predictions for unseen test points end up shifted up by
            # roughly safe_eps * y_std1 in raw units — biasing toward over-estimate
            # (the safe direction for STA delay). true_function1 stays unshifted so the
            # reported actual_values / NRMSE compare against the true ground truth.
            if safe_eps is not None and safe_eps != 0:
                y_test = y_test + safe_eps

            # Pass the updated mean and std to model_functions_at_training
            (model, outputs, losses, total_loss, total_mape_loss, predictions, actual_values,
             adam_condition_triggered, avg_total_rmse) = model_functions_at_training_gnn(
                initial_model,
                X_samples,
                y=y_test,
                true_samples=true_samples,
                true_function=true_function1,
                topology_cache=topology_cache,
                cache_type=cache_type,
                norm_stats=norm_stats,
                normalize_fn=normalize_fn,
                optim=optim,
                lr=lr,
                adam_step=40,
                std=y_std1,
                mean=y_mean1,
                move=move,
                left_bound=left_bound,
                right_bound=right_bound,
                total_points=total_points,
                mode=mode,
                asym_alpha=asym_alpha,
                pinball_tau=pinball_tau,
                inner_adam_lr=inner_adam_lr,
            )
            adam_used = adam_condition_triggered

            # Collect predictions and actuals
            all_predictions.extend(predictions)
            all_actuals.extend(actual_values)

            model_min = model
            loss_min = total_loss
            mape_min = total_mape_loss
            rmse_min = avg_total_rmse

    return (loss_min, mape_min, all_predictions, all_actuals, model_min, adam_used, rmse_min)


def model_functions_with_optim_mode_gnn(initial_model, X_samples, y, true_samples, true_function,
                                        topology_cache, cache_type, norm_stats, normalize_fn,
                                        optim_mode='selective_adam', num_steps=50, lr=0.003,
                                        std=1, mean=10, move=0, grad=1,
                                        left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                        asym_alpha=None, pinball_tau=None, inner_adam_lr=3e-4):
    """
    Train GNN model with different optimization modes for comparison.

    Args:
        optim_mode:
            - 'none': Grad+Move only (no optimization)
            - 'sgd': Direct SGD optimization (no grad/move)
            - 'adam': Direct Adam optimization (no grad/move)
            - 'selective_adam': Grad+Move + Adam if loss > threshold
    """
    import math
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 'none' / 'selective_adam' rely on caller-supplied grad/move scaling; 'sgd'/'adam' normalize locally.
    use_grad_move = optim_mode in ['none', 'selective_adam']

    y_tensor, true_function_tensor, std_tensor, mean_tensor, move_tensor = _setup_norm_tensors(
        y.clone(), true_function.clone(), std, mean, move, device,
    )

    # Rebuild a fresh copy of the pretrained model.
    model_type, node_features = _detect_gnn_model_type(initial_model)
    model = _clone_gnn_model(initial_model, model_type, node_features, device)

    criterion = nn.MSELoss()                       # for eval / metric reporting
    train_criterion = _make_train_criterion(asym_alpha, pinball_tau)  # for inner-loop training only

    K = len(X_samples)
    losses = []

    support_batch_data = [_make_pyg_data(s, topology_cache, cache_type) for s in X_samples]
    X_batch = Batch.from_data_list(support_batch_data).to(device)

    # grad/move modes assume caller already normalized y; direct-optim modes normalize locally.
    if use_grad_move:
        y_train = y_tensor
        y_mean_local = y_std_local = None
    else:
        y_mean_local = y_tensor.mean()
        y_std_local = y_tensor.std() + 1e-8
        y_train = (y_tensor - y_mean_local) / y_std_local

    initial_loss = train_criterion(model(X_batch), y_train) / K
    losses.append(initial_loss.item())

    # Apply optimization strategy.
    if optim_mode == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(num_steps):
            loss = train_criterion(model(X_batch), y_train) / K
            losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
    elif optim_mode == 'adam':
        _run_inner_adam_loop(model, X_batch, y_train, train_criterion, K,
                             num_steps=num_steps, inner_adam_lr=inner_adam_lr,
                             record_losses=losses)
    elif optim_mode == 'selective_adam' and initial_loss > 1e-4:
        _run_inner_adam_loop(model, X_batch, y_train, train_criterion, K,
                             num_steps=num_steps, inner_adam_lr=inner_adam_lr,
                             record_losses=losses)
    # optim_mode == 'none': no updates.

    # Evaluate on all total_points.
    predictions = []
    actual_values = []
    total_loss = 0
    total_mape_loss = 0
    total_rmse_loss = 0

    model.eval()
    with torch.no_grad():
        for i in range(total_points):
            sample_data = _make_pyg_data(true_samples[i], topology_cache, cache_type)
            sample_batch = Batch.from_data_list([sample_data]).to(device)
            raw_pred = model(sample_batch).item()

            if use_grad_move:
                pred_value = ((raw_pred - move_tensor) * std_tensor + mean_tensor).item()
                actual_value = ((true_function_tensor[i] - move_tensor) * std_tensor + mean_tensor).item()
            else:
                pred_value = (raw_pred * y_std_local + y_mean_local).item()
                actual_value = true_function[i].item()

            predictions.append(pred_value)
            actual_values.append(actual_value)

            squared_error = (pred_value - actual_value) ** 2
            mape_loss = abs((pred_value - actual_value) / (actual_value + 1e-8))

            total_loss += squared_error
            total_mape_loss += mape_loss
            total_rmse_loss += squared_error

    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_rmse = math.sqrt(total_rmse_loss / total_points)

    return {
        'model': model,
        'losses': losses,
        'predictions': predictions,
        'actual_values': actual_values,
        'total_loss': avg_total_loss,
        'total_mape': avg_total_mape,
        'total_rmse': avg_total_rmse,
        'optim_mode': optim_mode,
        'num_steps': num_steps,
        'use_grad_move': use_grad_move
    }


def compare_optimization_methods_gnn(initial_model, X_samples, y, true_samples, true_function,
                                     grad, move, topology_cache, cache_type, norm_stats, normalize_fn,
                                     num_steps=50, left_bound=5, right_bound=56, total_points=61,
                                     mode='extrapolation'):
    """
    Compare different optimization methods on the same task.

    Methods:
        - 'none': Grad+Move only (no optimization)
        - 'sgd': Direct SGD (no grad/move)
        - 'adam': Direct Adam (no grad/move)
        - 'selective_adam': Grad+Move + Adam if loss > threshold
    """
    y_mean = y.mean()
    y_std = y.std()
    std_scaled = y_std * grad

    # For grad_move methods: normalize targets
    y_norm = (y - y_mean) / std_scaled + move
    true_function_norm = (true_function - y_mean) / std_scaled + move

    methods = ['none', 'sgd', 'adam', 'selective_adam']
    method_names = {
        'none': 'Grad+Move Only',
        'sgd': f'SGD {num_steps} steps',
        'adam': f'Adam {num_steps} steps',
        'selective_adam': f'Selective Adam'
    }

    results = {}
    for method in methods:
        use_grad_move = method in ['none', 'selective_adam']

        if use_grad_move:
            y_input = y_norm
            true_fn_input = true_function_norm
        else:
            y_input = y  # Original scale
            true_fn_input = true_function  # Original scale

        result = model_functions_with_optim_mode_gnn(
            initial_model=initial_model,
            X_samples=X_samples,
            y=y_input,
            true_samples=true_samples,
            true_function=true_fn_input,
            topology_cache=topology_cache,
            cache_type=cache_type,
            norm_stats=norm_stats,
            normalize_fn=normalize_fn,
            optim_mode=method,
            num_steps=num_steps,
            lr=0.003,
            std=std_scaled,
            mean=y_mean,
            move=move,
            grad=grad,
            left_bound=left_bound,
            right_bound=right_bound,
            total_points=total_points,
            mode=mode
        )
        result['name'] = method_names[method]
        results[method] = result

    return results


def plot_optimization_comparison_gnn(results, indices, total_points=61, left_bound=5, right_bound=56,
                                     mode='extrapolation', cell_name='', task_id=0):
    """Plot comparison of different optimization methods."""
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ['none', 'sgd', 'adam', 'selective_adam']
    colors = {'none': '#e74c3c', 'sgd': '#3498db', 'adam': '#2ecc71', 'selective_adam': '#9b59b6'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Predictions
    ax = axes[0]
    x_axis = np.arange(total_points)
    actual = results['none']['actual_values']
    ax.plot(x_axis, actual, 'o-', label='Ground Truth', color='gray', alpha=0.5, markersize=3)

    for method in methods:
        preds = results[method]['predictions']
        ax.plot(x_axis, preds, '-', label=results[method]['name'], color=colors[method], alpha=0.8, linewidth=1.5)

    support_y = [actual[i] for i in indices]
    ax.scatter(indices, support_y, color='red', s=100, zorder=5, marker='x', label='Support Set')

    if mode == 'extrapolation':
        ax.axvspan(0, left_bound, alpha=0.1, color='blue')
        ax.axvspan(left_bound, right_bound, alpha=0.1, color='green')
        ax.axvspan(right_bound, total_points, alpha=0.1, color='orange')

    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Delay (s)', fontsize=11)
    ax.set_title(f'Predictions Comparison\nCell: {cell_name}, Task: {task_id}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    ax = axes[1]
    for method in methods:
        losses = results[method]['losses']
        if len(losses) > 1:
            ax.plot(losses, '-', label=results[method]['name'], color=colors[method], linewidth=1.5)
        else:
            ax.axhline(y=losses[0], linestyle='--', label=results[method]['name'], color=colors[method])
    ax.set_xlabel('Optimization Step', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: RMSE/NRMSE comparison (Total only)
    ax = axes[2]
    method_labels = [results[m]['name'] for m in methods]
    x_pos = np.arange(len(methods))
    width = 0.35

    rmse_total = [results[m]['total_rmse'] for m in methods]

    # Calculate NRMSE: RMSE / mean * 100
    actual_mean = np.mean(actual) if np.mean(actual) != 0 else 1e-8
    nrmse_total = [results[m]['total_rmse'] / actual_mean * 100 for m in methods]

    ax.bar(x_pos - width/2, rmse_total, width, label='RMSE', color='#34495e')
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, nrmse_total, width, label='NRMSE (%)', color='#27ae60', alpha=0.7)

    ax.set_xlabel('Optimization Method', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax2.set_ylabel('NRMSE (%)', fontsize=11)
    ax.set_title('Total RMSE/NRMSE Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(' ', '\n') for m in method_labels], fontsize=8)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_optimization_comparison_summary_gnn(results, mode='extrapolation'):
    """Print summary table."""
    methods = ['none', 'sgd', 'adam', 'selective_adam']

    print("\n" + "="*70)
    print("OPTIMIZATION METHOD COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Method':<20} | {'RMSE Total':<12} | {'MAPE Total (%)':<14}")
    print("-"*70)

    for method in methods:
        r = results[method]
        name = r['name']
        print(f"{name:<20} | {r['total_rmse']:<12.6f} | {r['total_mape']*100:<14.3f}")

    print("="*70)
