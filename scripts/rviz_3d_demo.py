#!/usr/bin/env python3
"""
VLA-6G 3D Demo Node — Standalone RViz visualization

Publishes MarkerArray + Odometry showing UAV cycling through 5 positioning methods.
No dependencies on other running nodes.
"""

import math
import random
import time

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA


# ---------------------------------------------------------------------------
# Channel model (inlined from channel_optimizer.py)
# ---------------------------------------------------------------------------
class ChannelModel:
    def __init__(self, frequency_ghz=300.0, bandwidth_ghz=10.0,
                 bs_power_dbm=30.0, uav_power_dbm=20.0):
        self.frequency_ghz = frequency_ghz
        self.bandwidth_ghz = bandwidth_ghz
        self.bs_power_dbm = bs_power_dbm
        self.uav_power_dbm = uav_power_dbm

    def calc_snr(self, distance, power_dbm):
        d_km = max(distance / 1000.0, 0.001)
        fspl = 20 * np.log10(d_km) + 20 * np.log10(self.frequency_ghz) + 92.45
        absorption = 10.0 * d_km
        path_loss = fspl + absorption
        noise = -174 + 10 * np.log10(self.bandwidth_ghz * 1e9) + 10
        return power_dbm - path_loss - noise

    def compute_metrics(self, uav_pos, bs_position, user_positions, user_requirements=None):
        d_bs_uav = np.linalg.norm(uav_pos - bs_position)
        snr_bs_uav = self.calc_snr(d_bs_uav, self.bs_power_dbm)
        user_rates, user_snrs, users_covered = [], [], 0
        for i, up in enumerate(user_positions):
            d = np.linalg.norm(uav_pos - up)
            snr_uu = self.calc_snr(d, self.uav_power_dbm)
            eff = min(snr_bs_uav, snr_uu)
            lin = 10 ** (eff / 10)
            rate = self.bandwidth_ghz * 1000 * np.log2(1 + max(lin, 0.001))
            rate = min(rate, 10000)
            user_rates.append(rate)
            user_snrs.append(snr_uu)
            req = user_requirements[i] if user_requirements else 25.0
            if rate >= req:
                users_covered += 1
        n = len(user_rates)
        s = sum(user_rates)
        s2 = sum(r**2 for r in user_rates)
        fairness = (s**2) / (n * s2) if s2 > 0 else 0
        return {
            'total_throughput': s,
            'user_rates': user_rates,
            'user_snrs': user_snrs,
            'average_rate': np.mean(user_rates) if user_rates else 0,
            'min_rate': min(user_rates) if user_rates else 0,
            'fairness': fairness,
            'coverage_rate': users_covered / n if n > 0 else 0,
        }


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------
def generate_scenario(seed=42):
    rng = random.Random(seed)
    bs = np.array([0.0, 0.0, 30.0])
    num_users = 5
    users = [np.array([rng.uniform(20, 80), rng.uniform(20, 80), 1.0])
             for _ in range(num_users)]
    reqs = [rng.uniform(10, 50) for _ in range(num_users)]
    return bs, users, reqs


def analytical_position(bs, users):
    cx = np.mean([u[0] for u in users])
    cy = np.mean([u[1] for u in users])
    bx, by = bs[0], bs[1]
    x = 0.6 * cx + 0.4 * bx
    y = 0.6 * cy + 0.4 * by
    return np.array([x, y, 25.0])


def optimize_position(bs, users, reqs):
    """Simple grid search optimisation (no scipy needed)."""
    ch = ChannelModel()
    best_score, best_pos = -1e9, np.array([50.0, 50.0, 25.0])
    for x in np.linspace(5, 95, 20):
        for y in np.linspace(5, 95, 20):
            for z in [15, 20, 25, 30, 35]:
                pos = np.array([x, y, z])
                m = ch.compute_metrics(pos, bs, users, reqs)
                sc = 0.6 * m['total_throughput'] + 0.3 * m['fairness'] * 1000 + 0.1 * m['coverage_rate'] * 1000
                if sc > best_score:
                    best_score, best_pos = sc, pos.copy()
    return best_pos


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------
METHODS = ['Random', 'Static', 'Analytical', 'VLA', 'Optimized']
METHOD_COLORS = {
    'Random':     (1.0, 0.3, 0.3),
    'Static':     (0.6, 0.6, 0.6),
    'Analytical': (0.3, 0.8, 1.0),
    'VLA':        (0.2, 1.0, 0.4),
    'Optimized':  (1.0, 0.85, 0.0),
}


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------
def _rgba(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _pt(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


def _header(stamp):
    h = Header()
    h.stamp = stamp
    h.frame_id = 'world'
    return h


def _make_marker(stamp, ns, mid, mtype, pos, scale, color, lifetime=0.0):
    m = Marker()
    m.header = _header(stamp)
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.position = _pt(*pos)
    m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    m.scale = Vector3(x=float(scale[0]), y=float(scale[1]), z=float(scale[2]))
    m.color = _rgba(*color)
    if lifetime > 0:
        m.lifetime.sec = int(lifetime)
        m.lifetime.nanosec = int((lifetime % 1) * 1e9)
    return m


# ---------------------------------------------------------------------------
# Demo node
# ---------------------------------------------------------------------------
class RViz3DDemoNode(Node):
    def __init__(self):
        super().__init__('rviz_3d_demo')
        self.pub_markers = self.create_publisher(MarkerArray, '/vla_demo/markers', 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom_world', 10)

        # Scenario
        self.bs, self.users, self.reqs = generate_scenario()
        self.channel = ChannelModel()

        # Pre-compute positions per method
        rng = random.Random(99)
        self.positions = {
            'Random': np.array([rng.uniform(10, 90), rng.uniform(10, 90), rng.uniform(15, 35)]),
            'Static': np.array([50.0, 50.0, 25.0]),
            'Analytical': analytical_position(self.bs, self.users),
            'VLA': analytical_position(self.bs, self.users) + np.array([2.0, -1.5, 1.0]),
            'Optimized': optimize_position(self.bs, self.users, self.reqs),
        }

        self.method_idx = 0
        self.phase_start = time.time()
        self.phase_duration = 4.0
        self.trail = []
        self.uav_pos = self.positions['Random'].copy()

        self.timer = self.create_timer(1.0 / 15.0, self.tick)  # 15Hz for smooth without flicker
        self.get_logger().info('RViz 3D demo started — cycling through 5 methods')

    # ------------------------------------------------------------------
    def tick(self):
        now = self.get_clock().now().to_msg()
        elapsed = time.time() - self.phase_start
        method = METHODS[self.method_idx]

        # Transition
        if elapsed >= self.phase_duration:
            self.method_idx = (self.method_idx + 1) % len(METHODS)
            self.phase_start = time.time()
            method = METHODS[self.method_idx]
            self.trail.clear()

        # Smooth move toward target
        target = self.positions[method]
        alpha = min(1.0, elapsed / 1.5)
        self.uav_pos = self.uav_pos + (target - self.uav_pos) * 0.08

        self.trail.append(self.uav_pos.copy())
        if len(self.trail) > 300:
            self.trail.pop(0)

        metrics = self.channel.compute_metrics(self.uav_pos, self.bs, self.users, self.reqs)

        ma = MarkerArray()

        # Use fixed sequential IDs — no DELETEALL to avoid flicker
        mid = [0]

        def nid():
            mid[0] += 1
            return mid[0]

        # --- BS tower ---
        m = _make_marker(now, 'bs', nid(), Marker.CYLINDER,
                         (self.bs[0], self.bs[1], self.bs[2] / 2),
                         (2.0, 2.0, self.bs[2]), (0.8, 0.1, 0.1))
        ma.markers.append(m)
        # BS antenna cone
        m = _make_marker(now, 'bs', nid(), Marker.MESH_RESOURCE,
                         (self.bs[0], self.bs[1], self.bs[2] + 1.5),
                         (3.0, 3.0, 3.0), (1.0, 0.2, 0.2))
        m.type = Marker.CYLINDER
        m.scale = Vector3(x=1.0, y=1.0, z=3.0)
        ma.markers.append(m)

        # --- Users ---
        for i, up in enumerate(self.users):
            m = _make_marker(now, 'users', nid(), Marker.CYLINDER,
                             (up[0], up[1], 0.75), (1.5, 1.5, 1.5), (0.9, 0.9, 0.9))
            ma.markers.append(m)
            # label
            m = _make_marker(now, 'users', nid(), Marker.TEXT_VIEW_FACING,
                             (up[0], up[1], 3.0), (0.1, 0.1, 1.5), (1.0, 1.0, 1.0))
            rate = metrics['user_rates'][i]
            m.text = f'U{i}:  {rate:.0f} Mbps'
            ma.markers.append(m)

        # --- UAV drone shape (80% of previous size) ---
        mc = METHOD_COLORS[method]
        ux, uy, uz = self.uav_pos
        # Central body
        m = _make_marker(now, 'uav', nid(), Marker.CUBE,
                         (ux, uy, uz), (4.0, 4.0, 1.0), (*mc, 1.0))
        ma.markers.append(m)
        # 4 arms + rotors
        arm_offsets = [(3.6, 3.6), (3.6, -3.6), (-3.6, 3.6), (-3.6, -3.6)]
        for dx, dy in arm_offsets:
            angle = math.atan2(dy, dx)
            length = math.sqrt(dx**2 + dy**2)
            arm = _make_marker(now, 'uav', nid(), Marker.CUBE,
                               (ux + dx * 0.5, uy + dy * 0.5, uz),
                               (length, 0.35, 0.35), (0.5, 0.5, 0.5))
            arm.pose.orientation = Quaternion(
                x=0.0, y=0.0,
                z=float(math.sin(angle / 2)),
                w=float(math.cos(angle / 2)))
            ma.markers.append(arm)
            rotor = _make_marker(now, 'uav', nid(), Marker.CYLINDER,
                                 (ux + dx, uy + dy, uz + 0.25),
                                 (2.8, 2.8, 0.16), (*mc, 0.6))
            ma.markers.append(rotor)
        # glow
        m = _make_marker(now, 'uav', nid(), Marker.SPHERE,
                         (ux, uy, uz), (10.0, 10.0, 5.0), (*mc, 0.12))
        ma.markers.append(m)

        # --- Link beams ---
        # BS -> UAV
        beam = _make_marker(now, 'links', nid(), Marker.LINE_STRIP,
                            (0, 0, 0), (0.3, 0.0, 0.0), (1.0, 0.4, 0.4, 0.6))
        beam.points = [_pt(*self.bs), _pt(*self.uav_pos)]
        ma.markers.append(beam)

        # UAV -> users (colored by rate)
        max_rate = max(metrics['user_rates']) if metrics['user_rates'] else 1
        for i, up in enumerate(self.users):
            rate = metrics['user_rates'][i]
            t = min(rate / max(max_rate, 1), 1.0)
            r, g, b = (1.0 - t, t, 0.2)
            beam = _make_marker(now, 'links', nid(), Marker.LINE_STRIP,
                                (0, 0, 0), (0.2, 0.0, 0.0), (r, g, b, 0.5))
            beam.points = [_pt(*self.uav_pos), _pt(up[0], up[1], up[2])]
            ma.markers.append(beam)

        # --- Trail ---
        if len(self.trail) > 2:
            trail_m = _make_marker(now, 'trail', nid(), Marker.LINE_STRIP,
                                   (0, 0, 0), (0.15, 0.0, 0.0), (*mc, 0.5))
            trail_m.points = [_pt(*p) for p in self.trail]
            ma.markers.append(trail_m)

        # --- Coverage dome per user ---
        for i, up in enumerate(self.users):
            rate = metrics['user_rates'][i]
            req = self.reqs[i]
            covered = rate >= req
            col = (0.2, 1.0, 0.3, 0.08) if covered else (1.0, 0.2, 0.2, 0.08)
            dome = _make_marker(now, 'coverage', nid(), Marker.SPHERE,
                                (up[0], up[1], 1.0), (8.0, 8.0, 4.0), col)
            ma.markers.append(dome)

        # --- HUD: 4 lines, fixed IDs, left-aligned ---
        # RViz TEXT_VIEW_FACING is center-aligned. To fake left-align,
        # shift X left for shorter lines based on character count difference.
        # Approximate: each char ~ 0.6 * font_size in world units.
        lines = [
            (f'Method: {method}', 3.2, mc + (1.0,)),
            (f'Throughput: {metrics["total_throughput"]:.0f} Mbps', 2.5, (1.0, 1.0, 1.0, 1.0)),
            (f'Fairness: {metrics["fairness"]:.3f}', 2.5, (1.0, 1.0, 1.0, 1.0)),
            (f'Coverage: {metrics["coverage_rate"]:.0%}', 2.5, (1.0, 1.0, 1.0, 1.0)),
        ]
        # Approximate world-space width: each char ~ 0.6 * font_size
        def _tw(t, s):
            return len(t) * 0.6 * s
        max_width = max(_tw(t, s) for t, s, _ in lines)
        z_positions = [52.0, 45.0, 38.0, 31.0]

        for i, (text, sz, col) in enumerate(lines):
            this_width = _tw(text, sz)
            # Shift shorter lines left so left edges align
            x_offset = -(max_width - this_width) / 2.0
            m = _make_marker(now, 'hud', 100 + i, Marker.TEXT_VIEW_FACING,
                             (50.0 + x_offset, -25.0, z_positions[i]),
                             (0.1, 0.1, sz), col)
            m.text = text
            ma.markers.append(m)

        self.pub_markers.publish(ma)

        # --- Odometry ---
        odom = Odometry()
        odom.header = _header(now)
        odom.child_frame_id = 'uav'
        odom.pose.pose.position = _pt(*self.uav_pos)
        odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.pub_odom.publish(odom)


def main():
    rclpy.init()
    node = RViz3DDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
