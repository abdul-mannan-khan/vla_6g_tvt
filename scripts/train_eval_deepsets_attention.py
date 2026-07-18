#!/usr/bin/env python3
"""
DeepSets with Attention Pooling — Reviewer Response Experiment

Replaces sum-pooling with multi-head attention pooling to test whether
the DeepSets collapse (68.5 Mbps) is due to the architecture or the
aggregation method. Uses a learned query that attends over per-user
embeddings, allowing the model to weight users by spatial importance.

Trains DeepSets-Attn-8K and DeepSets-Attn-2K variants.
"""

import sys
import os
import json
import time
import numpy as np

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

# Reuse feature extraction from original DeepSets
from train_eval_deepsets import (
    extract_per_user_features_from_raw,
    extract_global_features_from_raw,
    extract_features_for_scenario,
    FeatureNormalizer,
    DeepSetsDataset,
    deepsets_collate,
    load_training_data,
)


# ---------------------------------------------------------------------------
# Attention-Pooled DeepSets Architecture
# ---------------------------------------------------------------------------

class DeepSetsAttentionRelay(nn.Module):
    """DeepSets with multi-head attention pooling instead of sum-pooling."""

    def __init__(self, per_user_dim=5, global_dim=9, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        # Per-user encoder (same as original)
        self.phi = nn.Sequential(
            nn.Linear(per_user_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        # Attention pooling: learned query attends over user embeddings
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Decoder (same as original)
        decoder_input_dim = embed_dim + global_dim  # 137
        self.rho = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, user_features_list, global_features):
        batch_size = global_features.size(0)
        device = global_features.device
        pooled = torch.zeros(batch_size, self.embed_dim, device=device)

        for i in range(batch_size):
            user_feats = user_features_list[i]  # (num_users, 5)
            encoded = self.phi(user_feats)  # (num_users, 128)
            encoded = encoded.unsqueeze(0)  # (1, num_users, 128)
            query = self.query.to(device)  # (1, 1, 128)
            attn_out, _ = self.attn(query, encoded, encoded)  # (1, 1, 128)
            pooled[i] = self.layer_norm(attn_out.squeeze(0)).squeeze(0)

        combined = torch.cat([global_features, pooled], dim=1)
        return self.rho(combined)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deepsets_attn(per_user_train, global_train, targets_train,
                        per_user_val, global_val, targets_val,
                        normalizer, epochs=200, batch_size=64, lr=1e-3,
                        label="DeepSets-Attn"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {label} on {device}")

    model = DeepSetsAttentionRelay(per_user_dim=5, global_dim=9).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

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
            print(f"  Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}, "
                  f"val={val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if no_improve >= patience and epoch >= 60:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Best validation loss: {best_val_loss:.6f}")
    return model, n_params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_deepsets_attn(model, scenarios, normalizer, label="DeepSets-Attn"):
    device = next(model.parameters()).device
    model.eval()
    results = []

    for scenario in scenarios:
        per_user, global_feat = extract_features_for_scenario(scenario)
        per_user_norm = normalizer.normalize_user(per_user)
        global_norm = normalizer.normalize_global(global_feat)
        user_t = torch.tensor(per_user_norm, dtype=torch.float32).to(device)
        global_t = torch.tensor(global_norm, dtype=torch.float32).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            encoded = model.phi(user_t).unsqueeze(0)
            query = model.query.to(device)
            attn_out, attn_weights = model.attn(query, encoded, encoded)
            pooled = model.layer_norm(attn_out.squeeze(0))
            combined = torch.cat([global_t, pooled.squeeze(0)]).unsqueeze(0)
            pred_norm = model.rho(combined).cpu().numpy()[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        pred = normalizer.denormalize_target(pred_norm)
        pos = np.array([
            np.clip(pred[0], 0, 100),
            np.clip(pred[1], 0, 100),
            np.clip(pred[2], 5, 50),
        ])

        metrics = compute_channel_metrics(pos, scenario)
        results.append({
            'scenario_id': scenario.id,
            'predicted_position': pos.tolist(),
            'throughput': metrics['total_throughput'],
            'fairness': metrics['fairness'],
            'coverage_rate': metrics['coverage_rate'],
            'inference_ms': inference_ms,
            'composite_score': composite_score(metrics),
        })

    tp = np.array([r['throughput'] for r in results])
    fair = np.array([r['fairness'] for r in results])
    cov = np.array([r['coverage_rate'] for r in results])
    lat = np.array([r['inference_ms'] for r in results])

    summary = {
        'throughput_mbps': {
            'mean': float(np.mean(tp)),
            'std': float(np.std(tp)),
            'ci95': float(1.96 * np.std(tp, ddof=1) / np.sqrt(len(tp))),
        },
        'fairness_mean': float(np.mean(fair)),
        'coverage_mean': float(np.mean(cov)),
        'latency_ms': {
            'mean': float(np.mean(lat)),
            'p50': float(np.percentile(lat, 50)),
            'p95': float(np.percentile(lat, 95)),
        },
    }
    return summary, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(max_samples, label, scenarios):
    sep = '=' * 70
    print(f"\n{sep}")
    print(f"  {label} ({max_samples} training samples)")
    print(sep)

    per_user, global_feats, targets = load_training_data(max_samples=max_samples)

    normalizer = FeatureNormalizer()
    n_total = len(per_user)
    n_train = int(0.9 * n_total)

    normalizer.fit(per_user[:n_train], global_feats[:n_train], targets[:n_train])

    print(f"Train: {n_train}, Val: {n_total - n_train}")

    t0 = time.time()
    model, n_params = train_deepsets_attn(
        per_user[:n_train], global_feats[:n_train], targets[:n_train],
        per_user[n_train:], global_feats[n_train:], targets[n_train:],
        normalizer=normalizer, epochs=200, batch_size=64, lr=1e-3, label=label)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    print(f"\nEvaluating {label} on 100 canonical scenarios...")
    summary, per_scenario = evaluate_deepsets_attn(model, scenarios, normalizer, label)

    return summary, per_scenario, train_time, n_params


def main():
    print("DeepSets with Attention Pooling — Baseline Experiment")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    scenarios = generate_scenarios(100)

    # Analytical baseline for reference
    analytical_tp = []
    for s in scenarios:
        pos = get_position_analytical(s)
        m = compute_channel_metrics(pos, s)
        analytical_tp.append(m['total_throughput'])
    analytical_mean = float(np.mean(analytical_tp))
    print(f"\nAnalytical baseline: {analytical_mean:.1f} Mbps")

    all_results = {}

    for max_samples, label in [(8000, "DeepSets-Attn-8K"), (2000, "DeepSets-Attn-2K")]:
        summary, per_scenario, train_time, n_params = run_variant(
            max_samples, label, scenarios)

        vs_analytical = (summary['throughput_mbps']['mean'] - analytical_mean) / analytical_mean * 100

        all_results[label] = {
            'method': label.lower().replace('-', '_'),
            'architecture': 'DeepSets: phi(5->64->128) + attention-pool(4 heads) + rho(137->256->128->64->3)',
            'training_samples': max_samples,
            'training_time_s': train_time,
            'num_parameters': n_params,
            'summary': summary,
            'per_scenario': per_scenario,
        }

        sep = '=' * 60
        print(f"\n{sep}")
        print(f"{label} RESULTS")
        print(sep)
        print(f"  Parameters:  {n_params:,}")
        print(f"  Train time:  {train_time:.1f}s")
        print(f"  Throughput:  {summary['throughput_mbps']['mean']:.1f} "
              f"+/- {summary['throughput_mbps']['ci95']:.1f} Mbps")
        print(f"  Fairness:    {summary['fairness_mean']:.3f}")
        print(f"  Coverage:    {summary['coverage_mean'] * 100:.1f}%")
        print(f"  Latency:     {summary['latency_ms']['mean']:.3f} ms")
        print(f"  vs Analytical: {vs_analytical:+.1f}%")

    # Save
    combined = {
        'experiment': 'deepsets_attention_baseline',
        'description': 'DeepSets with multi-head attention pooling (4 heads)',
        'variants': all_results,
        'analytical_baseline_throughput': analytical_mean,
    }
    save_results(combined, 'deepsets_attention')

    # Summary
    sep = '=' * 70
    print(f"\n{sep}")
    print("COMPARISON: Sum-Pool vs Attention-Pool DeepSets")
    print(sep)
    print(f"{'Method':<25s} {'Throughput':>12s} {'Fairness':>10s} {'Coverage':>10s}")
    print("-" * 60)
    print(f"{'Analytical':<25s} {analytical_mean:>10.1f}  {'0.815':>10s} {'32.8%':>10s}")
    print(f"{'DeepSets-8K (sum)':<25s} {'68.5':>10s}  {'1.000':>10s} {'10.1%':>10s}")
    for label in ["DeepSets-Attn-8K", "DeepSets-Attn-2K"]:
        s = all_results[label]['summary']
        print(f"{label:<25s} {s['throughput_mbps']['mean']:>10.1f}  "
              f"{s['fairness_mean']:>10.3f} {s['coverage_mean']*100:>9.1f}%")


if __name__ == '__main__':
    main()
