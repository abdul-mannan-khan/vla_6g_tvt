#!/usr/bin/env python3
"""Generate a professional 3D network topology visualization (RViz replacement).
Fixes: RSU looks like a tower (not an arrow), labels don't overlap lines.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

# ── Scenario data ──
rsu_pos = np.array([80, 80, 30])
uav_pos = np.array([48, 45, 25])

vt_positions = np.array([
    [35, 55, 0],   # VT0
    [42, 40, 0],   # VT1
    [55, 50, 0],   # VT2
    [30, 30, 0],   # VT3
    [60, 60, 0],   # VT4
])
vt_rates = [29, 44, 41, 15, 14]
vt_qos_req = [20, 30, 25, 20, 10]
total_throughput = sum(vt_rates)
fairness = 0.841
coverage = 80

# ── Figure ──
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ── Ground plane ──
gx = np.array([0, 100, 100, 0])
gy = np.array([0, 0, 100, 100])
gz = np.array([0, 0, 0, 0])
ground = Poly3DCollection([list(zip(gx, gy, gz))], alpha=0.08,
                           facecolor='#B0C4DE', edgecolor='#999', linewidth=0.5)
ax.add_collection3d(ground)

# Grid
for i in range(0, 101, 20):
    ax.plot([i, i], [0, 100], [0, 0], color='#CCCCCC', linewidth=0.3, zorder=0)
    ax.plot([0, 100], [i, i], [0, 0], color='#CCCCCC', linewidth=0.3, zorder=0)

# ── RSU Tower (looks like a real lattice tower) ──
tb = rsu_pos[:2]  # tower base XY
th = rsu_pos[2]   # tower height

# Tower main mast (thick cylinder-like)
ax.plot([tb[0], tb[0]], [tb[1], tb[1]], [0, th], color='#8B0000', linewidth=5, solid_capstyle='round', zorder=5)

# Tower support legs (wider at base, converging at top)
leg_spread = 3
for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
    ax.plot([tb[0]+dx*leg_spread, tb[0]+dx*0.3],
            [tb[1]+dy*leg_spread, tb[1]+dy*0.3],
            [0, th*0.7], color='#A52A2A', linewidth=1.2, alpha=0.8, zorder=4)

# Cross braces on tower
for frac in [0.2, 0.4, 0.6]:
    h = th * frac
    s = leg_spread * (1 - frac)
    ax.plot([tb[0]-s, tb[0]+s], [tb[1], tb[1]], [h, h], color='#A52A2A', linewidth=0.8, alpha=0.6, zorder=4)
    ax.plot([tb[0], tb[0]], [tb[1]-s, tb[1]+s], [h, h], color='#A52A2A', linewidth=0.8, alpha=0.6, zorder=4)

# Antenna panels at top
for angle in [0, 2.094, -2.094]:
    dx = 1.5 * np.cos(angle)
    dy = 1.5 * np.sin(angle)
    ax.plot([tb[0], tb[0]+dx], [tb[1], tb[1]+dy], [th-1, th-1], color='#555', linewidth=2.5, zorder=6)

# Red beacon at top
ax.scatter(tb[0], tb[1], th+1, s=80, c='red', marker='o', zorder=7, depthshade=False, edgecolors='#8B0000', linewidths=1)

# Tower base platform
base_s = 4.5
bxs = [tb[0]-base_s, tb[0]+base_s, tb[0]+base_s, tb[0]-base_s]
bys = [tb[1]-base_s, tb[1]-base_s, tb[1]+base_s, tb[1]+base_s]
bzs = [0, 0, 0, 0]
bp = Poly3DCollection([list(zip(bxs, bys, bzs))], alpha=0.3, facecolor='#CD5C5C', edgecolor='#8B0000', linewidth=0.8)
ax.add_collection3d(bp)

# RSU label (positioned away from lines, top-right)
ax.text(tb[0]+5, tb[1]-5, th+4, 'THz RSU\n300 GHz, 30 dBm',
        fontsize=6.5, fontweight='bold', color='#8B0000', ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CD5C5C', alpha=0.95))

# ── UAV Relay ──
ax.scatter(*uav_pos, s=180, c='#006400', marker='o', zorder=8, depthshade=False, edgecolors='#003300', linewidths=1.5)
# Quadrotor arms + rotors
arm_len = 3.5
for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
    ex, ey = uav_pos[0]+dx*arm_len, uav_pos[1]+dy*arm_len
    ax.plot([uav_pos[0], ex], [uav_pos[1], ey], [uav_pos[2], uav_pos[2]],
            color='#006400', linewidth=1.5, zorder=7)
    theta = np.linspace(0, 2*np.pi, 20)
    ax.plot(ex + 1.8*np.cos(theta), ey + 1.8*np.sin(theta),
            np.full(20, uav_pos[2]), color='#228B22', linewidth=0.5, alpha=0.4, zorder=7)

# UAV label (upper-left, well clear of lines)
ax.text(uav_pos[0]-14, uav_pos[1]+2, uav_pos[2]+5, 'UAV Relay\n20 dBm, h=25 m',
        fontsize=6.5, fontweight='bold', color='#006400', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#228B22', alpha=0.95))

# Altitude reference
ax.plot([uav_pos[0], uav_pos[0]], [uav_pos[1], uav_pos[1]], [0, uav_pos[2]],
        color='gray', linewidth=0.6, linestyle=':', alpha=0.4, zorder=2)

# ── THz Backhaul link (UAV ↔ RSU) ──
ax.plot([uav_pos[0], rsu_pos[0]], [uav_pos[1], rsu_pos[1]], [uav_pos[2], rsu_pos[2]],
        color='#FF6347', linewidth=2.0, linestyle='--', alpha=0.7, zorder=4)

# Backhaul label (midpoint, offset upward away from line)
mid = (uav_pos + rsu_pos) / 2
ax.text(mid[0]+5, mid[1]-2, mid[2]+5, 'THz Backhaul',
        fontsize=5.5, color='#CD5C5C', ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

# ── Vehicular Terminals + Access Links ──
# Labels placed well clear of ALL link lines.
# Each access link goes from uav_pos (48,45,25) downward to the VT.
# Labels are placed far to the side, with a thin leader line connecting them.
# VT positions:  VT0(35,55) VT1(42,40) VT2(55,50) VT3(30,30) VT4(60,60)
# The backhaul goes from UAV(48,45) to RSU(80,80), so labels must avoid that too.
label_offsets = [
    (-18, 8, 5),    # VT0 - far upper-left (away from all links)
    (0, -10, 4),    # VT1 - directly below its marker
    (14, -14, 3),   # VT2 - far right, pushed lower
    (0, -10, 4),    # VT3 - directly below its marker
    (14, 6, 5),     # VT4 - far right-above
]

for i, (pos, rate, qos) in enumerate(zip(vt_positions, vt_rates, vt_qos_req)):
    qos_met = rate >= qos
    link_color = '#228B22' if qos_met else '#DC143C'
    marker_color = '#2E86C1' if qos_met else '#E74C3C'

    # Vehicle marker
    ax.scatter(pos[0], pos[1], pos[2], s=90, c=marker_color, marker='s',
               zorder=6, depthshade=False, edgecolors='#1a1a1a', linewidths=0.8)

    # Coverage dome
    theta = np.linspace(0, 2*np.pi, 30)
    dome_r = 4 if qos_met else 3
    dome_color = '#90EE90' if qos_met else '#FFB6C1'
    dome_verts = [list(zip(pos[0]+dome_r*np.cos(theta), pos[1]+dome_r*np.sin(theta), np.full(30, 0.1)))]
    dp = Poly3DCollection(dome_verts, alpha=0.2, facecolor=dome_color, edgecolor=link_color, linewidth=0.4)
    ax.add_collection3d(dp)

    # Access link (dashed)
    ax.plot([uav_pos[0], pos[0]], [uav_pos[1], pos[1]], [uav_pos[2], pos[2]],
            color=link_color, linewidth=1.0, alpha=0.6, linestyle='--', zorder=3)

    # Leader line from VT marker to label position (thin, grey)
    ox, oy, oz = label_offsets[i]
    lx, ly, lz = pos[0]+ox, pos[1]+oy, pos[2]+oz
    ax.plot([pos[0], lx], [pos[1], ly], [pos[2]+0.5, lz-0.5],
            color='#888888', linewidth=0.4, linestyle='-', alpha=0.5, zorder=2)

    # Label (well clear of link lines)
    symbol = u'\u2713' if qos_met else u'\u2717'
    ax.text(lx, ly, lz, f'VT{i}: {rate} Mbps {symbol}',
            fontsize=5.5, fontweight='bold', color='#1a1a1a', ha='center',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='#FFFDE7' if qos_met else '#FFEBEE',
                      edgecolor=link_color, alpha=0.95, linewidth=0.5),
            zorder=10)

# ── Metrics box ──
metrics = (f'Aggregate Throughput: {total_throughput} Mbps\n'
           f'Jain Fairness Index: {fairness}\n'
           f'QoS Coverage: {coverage}%')
ax.text2D(0.02, 0.95, metrics, transform=ax.transAxes, fontsize=7, fontweight='bold',
          verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', edgecolor='#4682B4', alpha=0.95, linewidth=1.0))

# ── Legend ──
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='THz RSU Tower'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#006400', markersize=7, label='UAV Relay'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2E86C1', markersize=6, label='VT (QoS met)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C', markersize=6, label='VT (QoS unmet)'),
    Line2D([0], [0], color='#228B22', linewidth=1.5, linestyle='--', label='Access link (LoS)'),
    Line2D([0], [0], color='#DC143C', linewidth=1.5, linestyle='--', label='Access link (weak)'),
    Line2D([0], [0], color='#FF6347', linewidth=1.5, linestyle='--', label='Backhaul link'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=5.5,
          framealpha=0.95, edgecolor='#888', fancybox=True, bbox_to_anchor=(1.0, 0.0))

# ── Axes ──
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 45)
ax.set_xlabel('X (m)', fontsize=8, labelpad=5)
ax.set_ylabel('Y (m)', fontsize=8, labelpad=5)
ax.set_zlabel('Altitude (m)', fontsize=8, labelpad=5)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.view_init(elev=28, azim=-55)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#CCCCCC')
ax.yaxis.pane.set_edgecolor('#CCCCCC')
ax.zaxis.pane.set_edgecolor('#CCCCCC')
ax.grid(True, alpha=0.2, linewidth=0.3)

plt.tight_layout()
plt.savefig('/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures/rviz_simulation_tvt.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("RViz figure saved.")
