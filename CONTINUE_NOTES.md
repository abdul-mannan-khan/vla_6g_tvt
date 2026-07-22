# MI-RL Paper - Continuation Notes

## Session Summary (2026-07-22 - Updated)

### COMPLETED: Phase 1 - RL Baseline Comparisons

**Goal:** Add DDPG, TD3, SAC, PPO baseline comparisons to raise IEEE IoT Journal acceptance likelihood.

**Status:** COMPLETED

### Key Results from Baseline Training

| Method | Throughput (Mbps) | Std (Mbps) | Min (Mbps) | SCA Floor Violations | Rate |
|--------|------------------|------------|------------|---------------------|------|
| DDPG | 5943.4 | 293.4 | 5092.1 | 769 | 39.4% |
| TD3 | 6006.9 | 261.9 | 5208.1 | 735 | 37.7% |
| SAC | 5960.4 | 268.4 | 5212.7 | 782 | 40.1% |
| PPO | 5835.2 | 243.1 | 5199.5 | 944 | 48.4% |
| **SGAC (Ours)** | **5960.1** | **266.9** | **5425.3** | **0** | **0%** |

**Key Finding:** SGAC guarantees 0% SCA floor violations via deterministic floor mechanism, while standard RL methods produce positioning decisions worse than the analytical baseline in 37-48% of episodes.

### What Was Completed This Session

1. **Trained all RL baselines on Vast.ai** (RTX 4090)
   - Created fresh instance 45557117 with 99.8% reliability
   - Deployed training code via SSH
   - Training completed in ~40 minutes
   - Results downloaded to `results_baselines_vastai/`

2. **Analyzed baseline results**
   - All methods achieve comparable final throughput (~5900-6000 Mbps)
   - Key differentiator is SCA floor violations
   - SGAC's floor mechanism prevents performance degradation

3. **Updated paper with comparison table** (committed)
   - Added RL baselines description in Section V.B
   - Added new Table (Comparison with Deep RL Baselines)
   - Updated abstract, contributions, conclusion with findings
   - Paper now 10 pages with comprehensive RL comparison

4. **Vast.ai instance destroyed** to save costs

### Commits This Session
1. `1c3a9d3` - Add RL baseline comparisons to IEEE IoT Journal paper

### Key Files
| File | Purpose |
|------|---------|
| `paper/mirl_iot_journal.tex` | Main IEEE IoT Journal paper (10 pages) |
| `paper/mirl_wcl.tex` | IEEE WCL version (3 pages) |
| `sgac_pytorch/baselines/` | RL baseline implementations |
| `results_baselines_vastai/` | RL baseline training results |
| `results_vastai/sgac_final/` | SGAC results (5960 Mbps) |

### Roadmap to 75% Acceptance (Updated)

| Task | Impact | Status |
|------|--------|--------|
| RL baseline comparisons | +10% | **COMPLETED** |
| Scale experiments (K=5->50) | +5% | Not started |
| Add Related Work section | +2% | Not started |
| Complexity analysis table | +1% | Not started |

### Current Acceptance Estimate
- Before baselines: 55-65%
- After baselines: **65-75%** (target achieved)

### Optional Next Steps (If Desired)
1. **Scale experiments** - Test with K=10, 20, 50 devices
2. **Related work section** - Add dedicated 1-page literature review
3. **Complexity analysis** - Add computational complexity comparison table
4. **Additional figures** - Consider adding convergence comparison figure

### Paper Statistics
- Pages: 10
- Tables: 4 (Throughput, RL Baselines, Summary, Algorithm)
- Figures: 7 (Convergence, Throughput, Reward, Floor, Improvement, Losses, Exploration)
- Theorems: 8 (with complete proofs)
- References: 30

### SGAC vs RL Baselines Summary
The key takeaway for reviewers:
- All methods converge to similar final throughput (within 3%)
- SGAC provides **guaranteed safety** with 0% floor violations
- Standard RL methods have 37-48% floor violations
- This makes SGAC suitable for mission-critical IoT deployments
