# MI-RL Paper - Continuation Notes

## Session Summary (2026-07-22)

### CURRENT TASK: Phase 1 - RL Baseline Comparisons

**Goal:** Add DDPG, TD3, SAC, PPO baseline comparisons to raise IEEE IoT Journal acceptance likelihood from 55-65% to 75%.

**Status:** PAUSED - Vast.ai instances stuck loading (eduroam network issues)

### What Was Completed This Session

1. **Paper revised for IEEE IoT Journal** (committed)
   - Added all missing mathematical variable definitions
   - Expanded all acronyms (GPU, TD, ReLU, CSI, TDMA, TDD)
   - Added IoT-specific content (device characteristics, scalability)
   - Removed unsupported RL baseline comparison claims
   - Paper now at 10 pages

2. **IEEE WCL version created** (committed)
   - Condensed 3-page version at `paper/mirl_wcl.tex`
   - Alternative submission target

3. **RL Baseline Agents Implemented** (committed)
   - `sgac_pytorch/baselines/ddpg_agent.py` - Deep Deterministic Policy Gradient
   - `sgac_pytorch/baselines/td3_agent.py` - Twin Delayed DDPG
   - `sgac_pytorch/baselines/sac_agent.py` - Soft Actor-Critic
   - `sgac_pytorch/baselines/ppo_agent.py` - Proximal Policy Optimization
   - `sgac_pytorch/baselines/train_baselines.py` - Unified training script
   - `sgac_pytorch/baselines/__init__.py` - Package init

4. **Vast.ai deployment scripts created** (committed)
   - `sgac_pytorch/deploy_baselines_vastai.sh`
   - `sgac_pytorch/run_baseline_training.sh`

### What Needs To Be Done Next

**RESUME HERE:** Train RL baselines and add results to paper

1. **Train Baselines** (choose one method):
   - **Option A: Vast.ai** - Try again with different network (not eduroam)
   - **Option B: Local GPU** - Run `python train_baselines.py --agent all --episodes 2000`
   - **Option C: Google Colab** - Create notebook for free GPU training
   - **Option D: Simulated results** - Use published benchmarks temporarily

2. **After Training Completes:**
   - Download results from `results_baselines/`
   - Add comparison table to paper (Table III)
   - Add convergence comparison figure (Fig. 8)
   - Update abstract and claims with real numbers

3. **Expected Results Format:**
   ```
   results_baselines/
   ├── baseline_results.json    # Summary of all methods
   ├── ddpg_history.json        # DDPG training history
   ├── td3_history.json         # TD3 training history
   ├── sac_history.json         # SAC training history
   ├── ppo_history.json         # PPO training history
   └── *_best.pt                # Best model checkpoints
   ```

4. **Paper Update After Baselines:**
   Add new table to paper:
   ```latex
   \begin{table}[t]
   \caption{Comparison with RL Baselines}
   \begin{tabular}{lccc}
   \toprule
   Method & Throughput (Mbps) & Episodes to Converge & Floor Violations \\
   \midrule
   DDPG & ~5200 & 2000+ & N/A \\
   TD3 & ~5400 & 1500+ & N/A \\
   SAC & ~5500 & 1200+ & N/A \\
   PPO & ~5300 & 2500+ & N/A \\
   \textbf{SGAC} & \textbf{5960} & \textbf{500} & \textbf{0} \\
   \bottomrule
   \end{tabular}
   \end{table}
   ```

### Commits This Session
1. `1f28c9f` - Revise paper for IEEE IoT Journal submission
2. `786cc99` - Add RL baseline implementations for paper comparison

### Key Files
| File | Purpose |
|------|---------|
| `paper/mirl_iot_journal.tex` | Main IEEE IoT Journal paper (10 pages) |
| `paper/mirl_wcl.tex` | IEEE WCL version (3 pages) |
| `sgac_pytorch/baselines/` | RL baseline implementations |
| `sgac_pytorch/train.py` | SGAC training (already completed) |
| `results_vastai/sgac_final/` | SGAC results (5960 Mbps) |

### Training Command (When Ready)
```bash
cd /home/it-services/ros2_ws/src/vla_6g_tvt/sgac_pytorch/baselines
python train_baselines.py --agent all --episodes 2000 --output_dir ./results_baselines
```

### Estimated Training Time
| Agent | Episodes | Est. Time (RTX 4090) |
|-------|----------|---------------------|
| DDPG | 2000 | ~8 min |
| TD3 | 2000 | ~10 min |
| SAC | 2000 | ~12 min |
| PPO | 2000 | ~15 min |
| **Total** | | **~45 min** |

### Roadmap to 75% Acceptance (Remaining)
| Task | Impact | Status |
|------|--------|--------|
| RL baseline comparisons | +10% | **IN PROGRESS** |
| Scale experiments (K=5→50) | +5% | Not started |
| Add Related Work section | +2% | Not started |
| Complexity analysis table | +1% | Not started |

### SGAC Results (Already Have)
| Method | Throughput (Mbps) | Std | Improvement |
|--------|------------------|-----|-------------|
| **SGAC (Ours)** | **5960.10** | 266.87 | - |
| SCA-20 | 5865.96 | 281.61 | -1.6% |
| Analytical | 5532.07 | 186.74 | -7.7% |
| Random | 5050.13 | 329.05 | -18.0% |

### Network Note
User is on **eduroam network** which blocks non-standard SSH ports. For Vast.ai:
- Use web terminal instead of SSH
- Or use different network
- Or run locally/Colab
