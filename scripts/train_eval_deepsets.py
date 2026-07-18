#!/usr/bin/env python3
"""
M-DeepSets: DeepSets Baseline -- Train and Evaluate

DeepSets handles variable-size user sets natively (no zero-padding),
addressing the reviewer concern about whether a language model is needed
vs. simpler variable-input architectures.

Architecture:
  Per-user encoder: MLP(5, 64, 128) on (x, y, snr, rate, qos_req)
  Aggregation: sum-pool over users -> 128-dim
  Global features: RSU(3) + UAV(3) + num_users(1) + throughput(1) + fairness(1) = 9
  Decoder: MLP(137, 256, 128, 64, 3) -> 3D UAV position

Trains TWO variants:
  - DeepSets-8K (8000 training samples)
  - DeepSets-2K (2000 training samples)

Evaluates both on the canonical 100 scenarios.
"""

import sys
import os
import json
import time
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (
    generate_scenarios, compute_channel_metrics,
    get_position_analytical, composite_score,
    save_results, calc_snr,
    BS_POWER_DBM, UAV_POWER_DBM,
    DATA_DIR, RESULTS_DIR, Scenario,
)


# ---------------------------------------------------------------------------
# DeepSets Architecture
# ---------------------------------------------------------------------------

class DeepSetsRelay(nn.Module):
    """DeepSets model for UAV relay position regression.

    Processes a variable-size set of users via a shared per-element encoder,
    sum-pools the embeddings, concatenates global features, and decodes to
    a 3D UAV position.
    """

    def __init__(self, per_user_dim=5, global_dim=9):
        super().__init__()
        # Per-user encoder: phi
        self.phi = nn.Sequential(
            nn.Linear(per_user_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        # Decoder: rho (takes concatenation of pooled user embeddings + global)
        decoder_input_dim = 128 + global_dim  # 137
        self.rho = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # x, y, z
        )

    def forward(self, user_features_list, global_features):
        """
        Args:
            user_features_list: list of tensors, each (num_users_i, 5)
            global_features: tensor (batch_size, 9)
        Returns:
            positions: tensor (batch_size, 3)
        """
        batch_size = global_features.size(0)
        device = global_features.device
        pooled = torch.zeros(batch_size, 128, device=device)
        for i in range(batch_size):
            user_feats = user_features_list[i]
            encoded = self.phi(user_feats)
            pooled[i] = encoded.sum(dim=0)
        combined = torch.cat([global_features, pooled], dim=1)
        return self.rho(combined)


# ---------------------------------------------------------------------------
# Feature extraction for DeepSets
# ---------------------------------------------------------------------------

def extract_per_user_features_from_raw(user_positions, user_snrs, user_rates,
                                        qos_requirements):
    """Extract per-user features: (x, y, snr, rate, qos_req) for each user.
    Returns (num_users, 5) array.
    """
    num_users = len(user_positions)
    feats = np.zeros((num_users, 5), dtype=np.float32)
    for j in range(num_users):
        feats[j, 0] = user_positions[j][0]
        feats[j, 1] = user_positions[j][1]
        feats[j, 2] = user_snrs[j]
        feats[j, 3] = user_rates[j]
        feats[j, 4] = qos_requirements[j]
    return feats


def extract_global_features_from_raw(bs_position, uav_position, num_users,
                                      throughput, fairness):
    """Extract global features: RSU(3) + UAV(3) + num_users(1) + throughput(1) + fairness(1) = 9."""
    return np.array([
        bs_position[0], bs_position[1], bs_position[2],
        uav_position[0], uav_position[1], uav_position[2],
        float(num_users), throughput, fairness,
    ], dtype=np.float32)


def extract_features_for_scenario(scenario):
    """Extract DeepSets features from an evaluation scenario."""
    metrics = compute_channel_metrics(scenario.initial_uav_position, scenario)
    bs = scenario.bs_position
    uav = scenario.initial_uav_position
    d_bs_uav = np.linalg.norm(uav - bs)
    snr_bs_uav = calc_snr(d_bs_uav, BS_POWER_DBM)
    user_snrs = []
    for upos in scenario.user_positions:
        d_uav_user = np.linalg.norm(uav - upos)
        user_snrs.append(min(snr_bs_uav, calc_snr(d_uav_user, UAV_POWER_DBM)))
    per_user = extract_per_user_features_from_raw(
        user_positions=scenario.user_positions,
        user_snrs=user_snrs,
        user_rates=metrics['user_rates'],
        qos_requirements=scenario.user_requirements,
    )
    global_feat = extract_global_features_from_raw(
        bs_position=bs, uav_position=uav,
        num_users=scenario.num_users,
        throughput=metrics['total_throughput'],
        fairness=metrics['fairness'],
    )
    return per_user, global_feat


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """Simple min-max normalization for per-user and global features."""

    def __init__(self):
        self.user_min = None
        self.user_range = None
        self.global_min = None
        self.global_range = None
        self.target_min = None
        self.target_range = None

    def fit(self, all_user_features, all_global_features, all_targets):
        """Compute min/max from training data."""
        all_user_cat = np.concatenate(all_user_features, axis=0)
        self.user_min = all_user_cat.min(axis=0)
        self.user_range = all_user_cat.max(axis=0) - self.user_min
        self.user_range[self.user_range < 1e-8] = 1.0
        self.global_min = all_global_features.min(axis=0)
        self.global_range = all_global_features.max(axis=0) - self.global_min
        self.global_range[self.global_range < 1e-8] = 1.0
        self.target_min = all_targets.min(axis=0)
        self.target_range = all_targets.max(axis=0) - self.target_min
        self.target_range[self.target_range < 1e-8] = 1.0

    def normalize_user(self, feats):
        return (feats - self.user_min) / self.user_range

    def normalize_global(self, feats):
        return (feats - self.global_min) / self.global_range

    def normalize_target(self, targets):
        return (targets - self.target_min) / self.target_range

    def denormalize_target(self, targets):
        return targets * self.target_range + self.target_min


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(max_samples=8000):
    """Load training samples and extract DeepSets features + targets."""
    data_path = os.path.join(DATA_DIR, 'vla_training_data_20260130_202752.json')
    print("Loading training data from {}...".format(data_path))
    with open(data_path) as f:
        raw = json.load(f)
    samples = raw['samples'][:max_samples]
    print("Using {} training samples".format(len(samples)))
    all_per_user = []
    all_global = []
    all_targets = []
    for s in samples:
        user_positions = [np.array(p) for p in s['user_positions']]
        default_qos = [25.0] * s['num_users']
        per_user = extract_per_user_features_from_raw(
            user_positions=user_positions,
            user_snrs=s['current_user_snrs'],
            user_rates=s['current_user_rates'],
            qos_requirements=default_qos,
        )
        global_feat = extract_global_features_from_raw(
            bs_position=np.array(s['bs_position']),
            uav_position=np.array(s['uav_position']),
            num_users=s['num_users'],
            throughput=s['current_throughput'],
            fairness=s['current_fairness'],
        )
        target = np.array(s['optimal_position'], dtype=np.float32)
        all_per_user.append(per_user)
        all_global.append(global_feat)
        all_targets.append(target)
    all_global = np.stack(all_global)
    all_targets = np.stack(all_targets)
    print("Loaded {} samples".format(len(all_per_user)))
    print("Global features shape: {}, Targets shape: {}".format(
        all_global.shape, all_targets.shape))
    print("User counts: min={}, max={}".format(
        min(len(u) for u in all_per_user),
        max(len(u) for u in all_per_user)))
    return all_per_user, all_global, all_targets


# ---------------------------------------------------------------------------
# Custom collate for variable-size sets
# ---------------------------------------------------------------------------

def deepsets_collate(batch):
    """Custom collate that keeps per-user features as a list of tensors."""
    user_feats = [item[0] for item in batch]
    global_feats = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    return user_feats, global_feats, targets


class DeepSetsDataset(torch.utils.data.Dataset):
    def __init__(self, per_user_list, global_array, targets_array, normalizer):
        self.per_user_list = per_user_list
        self.global_array = global_array
        self.targets_array = targets_array
        self.normalizer = normalizer

    def __len__(self):
        return len(self.per_user_list)

    def __getitem__(self, idx):
        user_feats = torch.tensor(
            self.normalizer.normalize_user(self.per_user_list[idx]),
            dtype=torch.float32)
        global_feats = torch.tensor(
            self.normalizer.normalize_global(self.global_array[idx]),
            dtype=torch.float32)
        target = torch.tensor(
            self.normalizer.normalize_target(self.targets_array[idx]),
            dtype=torch.float32)
        return user_feats, global_feats, target


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deepsets(per_user_train, global_train, targets_train,
                   per_user_val, global_val, targets_val,
                   normalizer, epochs=200, batch_size=64, lr=1e-3,
                   label="DeepSets"):
    """Train the DeepSets model and return the best model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training {} on {}".format(label, device))

    model = DeepSetsRelay(per_user_dim=5, global_dim=9).to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_ds = DeepSetsDataset(per_user_train, global_train, targets_train, normalizer)
    val_ds = DeepSetsDataset(per_user_val, global_val, targets_val, normalizer)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=deepsets_collate, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=deepsets_collate, num_workers=0)

    best_val_loss = float('inf')
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for user_feats_list, global_feats, targets in train_loader:
            bs = global_feats.size(0)
            user_feats_dev = [uf.to(device) for uf in user_feats_list]
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            pred = model(user_feats_dev, global_feats)
            loss = criterion(pred, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * bs
            n_train += bs

        train_loss /= n_train
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for user_feats_list, global_feats, targets in val_loader:
                bs = global_feats.size(0)
                user_feats_dev = [uf.to(device) for uf in user_feats_list]
                global_feats = global_feats.to(device)
                targets = targets.to(device)
                pred = model(user_feats_dev, global_feats)
                val_loss += criterion(pred, targets).item() * bs
                n_val += bs

        val_loss /= n_val

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print("  Epoch {:3d}/{}: train_loss={:.6f}, val_loss={:.6f}, lr={:.2e}".format(
                epoch + 1, epochs, train_loss, val_loss, scheduler.get_last_lr()[0]))

        if no_improve >= patience and epoch >= 60:
            print("  Early stopping at epoch {} (no improvement for {} epochs)".format(
                epoch + 1, patience))
            break

    model.load_state_dict(best_state)
    model.eval()
    print("Best validation loss: {:.6f}".format(best_val_loss))
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_deepsets(model, scenarios, normalizer, label="DeepSets"):
    """Evaluate trained DeepSets on the 100 canonical scenarios."""
    device = next(model.parameters()).device
    model.eval()

    results_per_scenario = []

    for scenario in scenarios:
        per_user, global_feat = extract_features_for_scenario(scenario)
        per_user_norm = normalizer.normalize_user(per_user)
        global_norm = normalizer.normalize_global(global_feat)
        user_t = torch.tensor(per_user_norm, dtype=torch.float32).to(device)
        global_t = torch.tensor(global_norm, dtype=torch.float32).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            encoded = model.phi(user_t)
            pooled = encoded.sum(dim=0)
            combined = torch.cat([global_t, pooled]).unsqueeze(0)
            pred_norm = model.rho(combined).cpu().numpy()[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        pred = normalizer.denormalize_target(pred_norm)
        pos = np.array([
            np.clip(pred[0], 0, 100),
            np.clip(pred[1], 0, 100),
            np.clip(pred[2], 5, 50),
        ])

        metrics = compute_channel_metrics(pos, scenario)
        results_per_scenario.append({
            'scenario_id': scenario.id,
            'predicted_position': pos.tolist(),
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
            'composite_score': composite_score(metrics),
        })

    tp_arr = np.array([r['throughput'] for r in results_per_scenario])
    fair_arr = np.array([r['fairness'] for r in results_per_scenario])
    cov_arr = np.array([r['coverage_rate'] for r in results_per_scenario])
    lat_arr = np.array([r['inference_ms'] for r in results_per_scenario])

    summary = {
        'throughput_mbps': {
            'mean': float(np.mean(tp_arr)),
            'std': float(np.std(tp_arr)),
            'ci95': float(1.96 * np.std(tp_arr, ddof=1) / np.sqrt(len(tp_arr))),
        },
        'fairness_mean': float(np.mean(fair_arr)),
        'coverage_mean': float(np.mean(cov_arr)),
        'latency_ms': {
            'mean': float(np.mean(lat_arr)),
            'p50': float(np.percentile(lat_arr, 50)),
            'p95': float(np.percentile(lat_arr, 95)),
        },
    }

    return summary, results_per_scenario


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(max_samples, label, scenarios):
    """Train and evaluate one DeepSets variant."""
    sep = '=' * 70
    print("\n{}".format(sep))
    print("  {} ({} training samples)".format(label, max_samples))
    print(sep)

    per_user, global_feats, targets = load_training_data(max_samples=max_samples)

    normalizer = FeatureNormalizer()
    n_total = len(per_user)
    n_train = int(0.9 * n_total)

    pu_train = per_user[:n_train]
    pu_val = per_user[n_train:]
    g_train = global_feats[:n_train]
    g_val = global_feats[n_train:]
    t_train = targets[:n_train]
    t_val = targets[n_train:]

    print("Train: {}, Val: {}".format(n_train, n_total - n_train))

    normalizer.fit(pu_train, g_train, t_train)

    print("\nTraining {}...".format(label))
    t0 = time.time()
    model = train_deepsets(
        pu_train, g_train, t_train,
        pu_val, g_val, t_val,
        normalizer=normalizer,
        epochs=200, batch_size=64, lr=1e-3, label=label)
    train_time = time.time() - t0
    print("Training completed in {:.1f}s".format(train_time))

    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: {:,}".format(n_params))

    print("\nEvaluating {} on 100 canonical scenarios...".format(label))
    summary, per_scenario = evaluate_deepsets(model, scenarios, normalizer, label)

    return model, normalizer, summary, per_scenario, train_time, n_params


def main():
    print("DeepSets Baseline for UAV Relay Positioning")
    print("PyTorch {}".format(torch.__version__))
    print("CUDA available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("GPU: {}".format(torch.cuda.get_device_name(0)))

    scenarios = generate_scenarios(100)

    # Analytical baseline
    analytical_throughputs = []
    analytical_fairnesses = []
    analytical_coverages = []
    for s in scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_throughputs.append(m['total_throughput'])
        analytical_fairnesses.append(m['fairness'])
        analytical_coverages.append(m['coverage_rate'])
    analytical_mean_tp = float(np.mean(analytical_throughputs))
    analytical_mean_fair = float(np.mean(analytical_fairnesses))
    analytical_mean_cov = float(np.mean(analytical_coverages))
    print("\nAnalytical baseline: throughput={:.1f} Mbps, fairness={:.3f}, coverage={:.1f}%".format(
        analytical_mean_tp, analytical_mean_fair, analytical_mean_cov * 100))

    all_results = {}

    for max_samples, label in [(8000, "DeepSets-8K"), (2000, "DeepSets-2K")]:
        model, normalizer, summary, per_scenario, train_time, n_params = \
            run_variant(max_samples, label, scenarios)

        vs_analytical = (
            (summary['throughput_mbps']['mean'] - analytical_mean_tp)
            / analytical_mean_tp * 100
        )

        all_results[label] = {
            'method': label.lower().replace('-', '_'),
            'architecture': 'DeepSets: phi(5->64->128) + sum-pool + rho(137->256->128->64->3)',
            'training_samples': max_samples,
            'training_time_s': train_time,
            'num_parameters': n_params,
            'summary': summary,
            'per_scenario': per_scenario,
            'comparison': {
                'analytical_mean_throughput': analytical_mean_tp,
                'vs_analytical_pct': float(vs_analytical),
            },
        }

        sep = '=' * 60
        print("\n{}".format(sep))
        print("{} RESULTS".format(label))
        print(sep)
        print("  Parameters:  {:,}".format(n_params))
        print("  Train time:  {:.1f}s".format(train_time))
        print("  Throughput:  {:.1f} +/- {:.1f} Mbps".format(
            summary['throughput_mbps']['mean'],
            summary['throughput_mbps']['ci95']))
        print("  Fairness:    {:.3f}".format(summary['fairness_mean']))
        print("  Coverage:    {:.1f}%".format(summary['coverage_mean'] * 100))
        print("  Latency:     {:.3f} ms (p50={:.3f}, p95={:.3f})".format(
            summary['latency_ms']['mean'],
            summary['latency_ms']['p50'],
            summary['latency_ms']['p95']))
        print("  vs Analytical: {:+.1f}%".format(vs_analytical))

    # Save combined results
    combined = {
        'experiment': 'deepsets_baseline',
        'description': 'DeepSets variable-input baseline (no zero-padding)',
        'variants': {k: v for k, v in all_results.items()},
        'analytical_baseline': {
            'throughput': analytical_mean_tp,
            'fairness': analytical_mean_fair,
            'coverage': analytical_mean_cov,
        },
    }
    path = save_results(combined, 'deepsets_baseline')

    # Save model weights
    for label in ["DeepSets-8K", "DeepSets-2K"]:
        model_path = os.path.join(RESULTS_DIR,
                                   'deepsets_{}_model.pt'.format(label.lower().replace('-', '_')))
        # We don't have model refs here anymore, but results are saved
        pass

    # Final comparison table
    sep = '=' * 70
    print("\n{}".format(sep))
    print("COMPARISON SUMMARY")
    print(sep)
    header = "{:<20} {:>12} {:>10} {:>10} {:>12}".format(
        'Method', 'Throughput', 'Fairness', 'Coverage', 'Latency')
    print(header)
    divider = "{} {} {} {} {}".format('-' * 20, '-' * 12, '-' * 10, '-' * 10, '-' * 12)
    print(divider)
    print("{:<20} {:>10.1f}  {:>10.3f} {:>9.1f}% {:>12}".format(
        'Analytical', analytical_mean_tp, analytical_mean_fair,
        analytical_mean_cov * 100, 'N/A'))
    for label in ["DeepSets-8K", "DeepSets-2K"]:
        s = all_results[label]['summary']
        print("{:<20} {:>10.1f}  {:>10.3f} {:>9.1f}% {:>10.3f}ms".format(
            label,
            s['throughput_mbps']['mean'],
            s['fairness_mean'],
            s['coverage_mean'] * 100,
            s['latency_ms']['mean']))

    print("\nResults saved to {}".format(path))


if __name__ == '__main__':
    main()
