# MI-RL Paper - Continuation Notes

## Last Session Summary (2026-07-20)

### Completed Tasks:
1. **Figure Fixes**:
   - Removed redundant fig5 (sample_efficiency)
   - Fixed Fig1: removed error bars to reduce high variation
   - Improved all legend placements to avoid data overlap
   - Renumbered figures: 1-7 (was 1-8)

2. **RL Baseline Comparisons Added**:
   - DDPG, PPO, TD3, SAC compared against SGAC
   - Added Table II with RL comparison metrics
   - Updated throughput section with RL improvements

3. **Commits Pushed**:
   - `b23d878`: Fix figures: remove redundant Fig5, improve legends
   - `3a3d4c8`: Add comprehensive RL baseline comparisons

### Pending Tasks (TODO Next Session):

1. **Expand Introduction**:
   - Current intro is only 3 paragraphs - needs 15-20 paragraphs
   - Should cover: 6G IoT landscape, UAV communications evolution,
     optimization challenges, RL applications, limitations of existing work
   - Add motivation for math-informed approach

2. **Add Recent References (2022+)**:
   - Currently only 8 references, many from 2016-2018
   - Need 15-20 references from 2022-2026
   - Topics: 6G IoT, UAV-assisted communications, deep RL for wireless,
     convex optimization, hybrid learning approaches

3. **Symbol Definitions**:
   - Define all symbols in System Model properly
   - Add "where..." explanations after equations
   - Key symbols to define: $d$, $f_c$, $c$, $PL_{\text{excess}}$,
     $\gamma_{\text{BU}}$, $\gamma_k$, $\alpha$, $\beta$, $\eta$, $\lambda$

4. **Equation Explanations**:
   - Each equation should have brief explanation of its meaning
   - Add physical interpretation where appropriate

### File Locations:
- Paper: `/home/it-services/ros2_ws/src/vla_6g_tvt/paper/mirl_iot_journal.tex`
- Figures: `/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures/`
- Figure generator: `/home/it-services/ros2_ws/src/vla_6g_tvt/scripts/generate_paper_figures.py`

### Current Paper Structure:
1. Abstract (complete)
2. Introduction (needs expansion)
3. System Model (needs symbol definitions)
4. SGAC Algorithm (complete)
5. Theoretical Analysis (complete with proofs)
6. Experimental Evaluation (complete with 7 figures)
7. Discussion (complete)
8. Conclusion (complete)

### To Continue:
Ask: "Continue with the MI-RL paper - expand introduction and add symbol definitions"
