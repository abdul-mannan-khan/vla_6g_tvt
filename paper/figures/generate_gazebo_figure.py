#!/usr/bin/env python3
"""Generate a professional 3D urban LAWN simulation figure using PyVista/VTK.

Replaces the Gazebo screenshot approach (EGL/NVIDIA rendering issues).
Creates a simulation-quality 3D scene with:
  - Urban environment: buildings, roads with markings, sidewalks, trees
  - Vehicular terminals (cars) on roads
  - UAV relay drone hovering (enlarged, with glow sphere)
  - RSU tower with antennas
  - THz communication links (backhaul + access)
  - Blue sky background, proper lighting
"""

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

# ── Create plotter ──
pl = pv.Plotter(off_screen=True, window_size=[1920, 1080])
pl.set_background('skyblue', top='white')

# ── Ground plane (grass) ──
ground = pv.Plane(center=(0, 0, -0.05), direction=(0, 0, 1),
                  i_size=150, j_size=150, i_resolution=1, j_resolution=1)
pl.add_mesh(ground, color='#4a7c3f', ambient=0.3, diffuse=0.7)

# ── Roads ──
road_x = pv.Box(bounds=(-60, 60, -5, 5, 0, 0.02))
pl.add_mesh(road_x, color='#2a2a2a', ambient=0.2, diffuse=0.8)

road_y = pv.Box(bounds=(-5, 5, -60, 60, 0, 0.02))
pl.add_mesh(road_y, color='#2a2a2a', ambient=0.2, diffuse=0.8)

# Center lines (yellow)
cl_x = pv.Box(bounds=(-60, 60, -0.1, 0.1, 0.03, 0.04))
pl.add_mesh(cl_x, color='yellow', ambient=0.4)
cl_y = pv.Box(bounds=(-0.1, 0.1, -60, 60, 0.03, 0.04))
pl.add_mesh(cl_y, color='yellow', ambient=0.4)

# Lane markings (dashed white)
for offset in [-3.0, 3.0]:
    for start in range(-55, 55, 8):
        dash = pv.Box(bounds=(start, start+4, offset-0.08, offset+0.08, 0.03, 0.04))
        pl.add_mesh(dash, color='white', ambient=0.3)

for offset in [-3.0, 3.0]:
    for start in range(-55, 55, 8):
        dash = pv.Box(bounds=(offset-0.08, offset+0.08, start, start+4, 0.03, 0.04))
        pl.add_mesh(dash, color='white', ambient=0.3)

# Sidewalks
for y_off in [6.5, -6.5]:
    sw = pv.Box(bounds=(-60, 60, y_off-1.2, y_off+1.2, 0, 0.15))
    pl.add_mesh(sw, color='#b0b0b0', ambient=0.3, diffuse=0.7)
for x_off in [6.5, -6.5]:
    sw = pv.Box(bounds=(x_off-1.2, x_off+1.2, -60, 60, 0, 0.15))
    pl.add_mesh(sw, color='#b0b0b0', ambient=0.3, diffuse=0.7)


def add_building(pl, cx, cy, w, d, h, color):
    """Add a building with window stripes."""
    import matplotlib.colors as mc
    body = pv.Box(bounds=(cx-w/2, cx+w/2, cy-d/2, cy+d/2, 0, h))
    pl.add_mesh(body, color=color, ambient=0.2, diffuse=0.7, specular=0.1)
    # Roof
    roof = pv.Box(bounds=(cx-w/2-0.1, cx+w/2+0.1, cy-d/2-0.1, cy+d/2+0.1, h, h+0.3))
    r, g, b = mc.to_rgb(color)
    roof_c = (max(0, r-0.08), max(0, g-0.08), max(0, b-0.08))
    pl.add_mesh(roof, color=roof_c, ambient=0.2, diffuse=0.6)
    # Window stripes
    for frac in np.linspace(0.2, 0.85, 4):
        wh = h * frac
        stripe = pv.Box(bounds=(cx-w/2-0.05, cx+w/2+0.05,
                                cy-d/2-0.05, cy+d/2+0.05,
                                wh, wh+h*0.08))
        pl.add_mesh(stripe, color='#8ec8e8', ambient=0.3, diffuse=0.5,
                    specular=0.4, opacity=0.7)


# Buildings (b1 moved right so drone has clear green/sky background)
add_building(pl, 30, 22, 12, 12, 25, '#b0c0d0')    # moved right, away from drone
add_building(pl, -18, 18, 14, 10, 17, '#c0a090')    # light tan
add_building(pl, 18, -22, 16, 12, 12, '#909080')    # shifted south
add_building(pl, -22, -22, 14, 14, 10, '#a0a0a0')   # shifted further out
add_building(pl, -30, 16, 9, 9, 40, '#8098b0')      # tall tower left
add_building(pl, 30, -18, 12, 10, 22, '#906050')     # shifted right
add_building(pl, 38, 32, 10, 10, 18, '#708090')
add_building(pl, -35, -28, 8, 12, 14, '#806050')


def add_tree(pl, x, y, trunk_h=4.0, canopy_r=2.2):
    """Add a tree (trunk cylinder + canopy sphere)."""
    trunk = pv.Cylinder(center=(x, y, trunk_h/2), direction=(0, 0, 1),
                        radius=0.22, height=trunk_h, resolution=12)
    pl.add_mesh(trunk, color='#5c3a1e', ambient=0.3, diffuse=0.7)
    canopy = pv.Sphere(radius=canopy_r, center=(x, y, trunk_h + canopy_r * 0.7))
    pl.add_mesh(canopy, color='#2d6b1e', ambient=0.25, diffuse=0.75)


# Trees
for x in [-30, -15, 5, 20, 35]:
    add_tree(pl, x, 8.5, 4.0, 2.0)
for x in [-25, -10, 10, 25]:
    add_tree(pl, x, -8.5, 3.8, 1.9)
for y in [15, 25, 35]:
    add_tree(pl, 8.5, y, 4.2, 2.1)


def add_car(pl, x, y, yaw_deg, color, suv=False):
    """Add a car (body + cabin + wheels)."""
    import math
    ca = math.cos(math.radians(yaw_deg))
    sa = math.sin(math.radians(yaw_deg))

    bh = 0.65 if not suv else 0.85
    bl = 4.2 if not suv else 4.8
    bw = 1.8 if not suv else 1.9

    # Body
    body = pv.Box(bounds=(-bl/2, bl/2, -bw/2, bw/2, 0.25, 0.25+bh))
    body.rotate_z(yaw_deg, inplace=True)
    body.translate([x, y, 0], inplace=True)
    pl.add_mesh(body, color=color, ambient=0.2, diffuse=0.7, specular=0.2)

    # Cabin (glass)
    ch = 0.55 if not suv else 0.65
    cabin = pv.Box(bounds=(-1.0, 0.8, -bw/2*0.85, bw/2*0.85, 0.25+bh, 0.25+bh+ch))
    cabin.rotate_z(yaw_deg, inplace=True)
    cabin.translate([x, y, 0], inplace=True)
    pl.add_mesh(cabin, color='#a0c8e8', ambient=0.3, diffuse=0.5,
                specular=0.5, opacity=0.75)

    # Wheels
    wr = 0.28 if not suv else 0.32
    for lx, ly in [(1.2, 0.95), (1.2, -0.95), (-1.2, 0.95), (-1.2, -0.95)]:
        wx = x + lx * ca - ly * sa
        wy = y + lx * sa + ly * ca
        wheel = pv.Cylinder(center=(wx, wy, wr), direction=(0, 0, 1),
                            radius=wr, height=0.2, resolution=16)
        pl.add_mesh(wheel, color='#1a1a1a', ambient=0.2, diffuse=0.6)


# Cars
add_car(pl, -15, 2.5, 0, '#1a3580')
add_car(pl, 5, -2.5, 180, '#b01515')
add_car(pl, 20, 2.5, 0, '#a0a0a0')
add_car(pl, -25, -2.5, 180, '#1a6b1a', suv=True)
add_car(pl, 2, 15, 90, '#c86020')


# ── RSU Tower ──
tower_x, tower_y = 8, 12
tower_h = 30

mast = pv.Cylinder(center=(tower_x, tower_y, tower_h/2), direction=(0, 0, 1),
                   radius=0.4, height=tower_h, resolution=16)
pl.add_mesh(mast, color='#cc2020', ambient=0.2, diffuse=0.7, specular=0.1)

# Support legs
for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
    leg = pv.Line(
        [tower_x + dx*2.5, tower_y + dy*2.5, 0],
        [tower_x + dx*0.5, tower_y + dy*0.5, tower_h*0.65]
    ).tube(radius=0.12)
    pl.add_mesh(leg, color='#a03030', ambient=0.2, diffuse=0.7)

# Cross braces
for frac in [0.2, 0.4, 0.6]:
    h = tower_h * frac
    s = 2.5 * (1 - frac)
    for ddx, ddy in [(1, 0), (0, 1)]:
        brace = pv.Line(
            [tower_x - ddx*s, tower_y - ddy*s, h],
            [tower_x + ddx*s, tower_y + ddy*s, h]
        ).tube(radius=0.08)
        pl.add_mesh(brace, color='#a03030', ambient=0.2, diffuse=0.7)

# Antenna panels
for angle in [0, 2.094, -2.094]:
    adx = 1.2 * np.cos(angle)
    ady = 1.2 * np.sin(angle)
    panel = pv.Box(bounds=(-0.08, 0.08, -0.4, 0.4, -0.75, 0.75))
    panel.translate([tower_x + adx, tower_y + ady, tower_h - 1], inplace=True)
    pl.add_mesh(panel, color='#666666', ambient=0.2, diffuse=0.7, specular=0.2)

# Beacon
beacon = pv.Sphere(radius=0.3, center=(tower_x, tower_y, tower_h + 0.5))
pl.add_mesh(beacon, color='red', ambient=0.5, diffuse=0.5)

# Tower base
tbase = pv.Box(bounds=(tower_x-1.5, tower_x+1.5, tower_y-1.5, tower_y+1.5, 0, 0.3))
pl.add_mesh(tbase, color='#666666', ambient=0.2, diffuse=0.7)


# ── UAV Drone (bright orange, shifted left-upper at 45 deg) ──
uav_pos = np.array([-18.0, -4.0, 38.0])

# Fuselage - bright orange, enlarged
fuselage = pv.Box(bounds=(-1.0, 1.0, -1.0, 1.0, -0.3, 0.3))
fuselage.translate(uav_pos, inplace=True)
pl.add_mesh(fuselage, color='#ff6600', ambient=0.35, diffuse=0.65, specular=0.2)

# Arms along cardinal directions (X/Y axes) so all 4 arms are clearly
# visible from the 45-degree isometric camera angle.
arm_len = 3.5
arm_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # N, E, S, W
for dx, dy in arm_dirs:
    arm_end = uav_pos + np.array([dx*arm_len, dy*arm_len, 0])
    arm_tube = pv.Line(uav_pos, arm_end).tube(radius=0.15)
    pl.add_mesh(arm_tube, color='#cc5500', ambient=0.3, diffuse=0.7)

    motor = pv.Cylinder(center=arm_end + [0, 0, 0.15], direction=(0, 0, 1),
                        radius=0.3, height=0.35, resolution=16)
    pl.add_mesh(motor, color='#ff8800', ambient=0.3, diffuse=0.7)

    prop = pv.Disc(center=arm_end + [0, 0, 0.35], normal=(0, 0, 1),
                   inner=0.05, outer=1.4, r_res=1, c_res=30)
    pl.add_mesh(prop, color='#00ddff', ambient=0.4, diffuse=0.5,
                opacity=0.5, specular=0.2)

# Landing gear (wider for visibility)
for dy_off in [0.8, -0.8]:
    leg = pv.Cylinder(center=uav_pos + [0, dy_off, -0.5], direction=(0, 0, 1),
                      radius=0.06, height=0.7, resolution=8)
    pl.add_mesh(leg, color='#cc5500', ambient=0.3, diffuse=0.7)
    skid = pv.Cylinder(center=uav_pos + [0, dy_off, -0.85], direction=(1, 0, 0),
                       radius=0.05, height=1.5, resolution=8)
    pl.add_mesh(skid, color='#cc5500', ambient=0.3, diffuse=0.7)

# Glow sphere
glow = pv.Sphere(radius=3.5, center=uav_pos)
pl.add_mesh(glow, color='#ffaa00', ambient=0.25, diffuse=0.4,
            opacity=0.12, specular=0.0)


# ── Communication Links ──
rsu_top = np.array([float(tower_x), float(tower_y), float(tower_h)])

# ── Helper: dashed 3D line (series of short tube segments with gaps) ──
def add_dashed_line(pl, p0, p1, color, radius=0.1, dash_len=1.5, gap_len=1.0,
                    ambient=0.3, diffuse=0.5, opacity=0.7):
    """Draw a dashed line from p0 to p1 as discrete tube segments."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    direction = p1 - p0
    total_len = np.linalg.norm(direction)
    if total_len < 1e-6:
        return
    unit = direction / total_len
    cycle = dash_len + gap_len
    t = 0.0
    while t < total_len:
        seg_end = min(t + dash_len, total_len)
        seg = pv.Line(p0 + unit * t, p0 + unit * seg_end).tube(radius=radius)
        pl.add_mesh(seg, color=color, ambient=ambient, diffuse=diffuse, opacity=opacity)
        t += cycle


# THz Backhaul (UAV ↔ RSU) — dashed
add_dashed_line(pl, uav_pos, rsu_top, color='#ff5533', radius=0.12,
                dash_len=1.8, gap_len=1.2, opacity=0.7)

# Access links — dashed
car_positions = [
    np.array([-15.0, 2.5, 1.0]),
    np.array([5.0, -2.5, 1.0]),
    np.array([20.0, 2.5, 1.0]),
    np.array([-25.0, -2.5, 1.0]),
    np.array([2.0, 15.0, 1.0]),
]
for i, cp in enumerate(car_positions):
    lc = '#22aa22' if i < 3 else '#cc2222'
    add_dashed_line(pl, uav_pos, cp, color=lc, radius=0.06,
                    dash_len=1.2, gap_len=0.8, opacity=0.55)

# ── Light poles ──
for lp_x in [-25, 10, 25]:
    pole = pv.Cylinder(center=(lp_x, 7.5, 3.5), direction=(0, 0, 1),
                       radius=0.08, height=7, resolution=8)
    pl.add_mesh(pole, color='#666666', ambient=0.2, diffuse=0.7)
    lamp = pv.Sphere(radius=0.2, center=(lp_x, 7.0, 7.0))
    pl.add_mesh(lamp, color='#ffee99', ambient=0.5, diffuse=0.5)


# ── Lighting ──
pl.add_light(pv.Light(position=(50, -50, 80), focal_point=(0, 0, 0),
                       color='white', intensity=1.0))
pl.add_light(pv.Light(position=(-30, 30, 60), focal_point=(0, 0, 10),
                       color='#dde8ff', intensity=0.4))

# ── Camera (isometric view — high elevation so roads, cars, drone all visible) ──
pl.camera_position = [
    (-68, -68, 75),   # camera position (slightly higher to show full tower)
    (0, 0, 8),        # focal point
    (0, 0, 1),        # view up
]

# ── Render ──
out_path = '/home/it-services/ros2_ws/src/vla_6g_tvt/paper/figures/gazebo_simulation_tvt.png'
pl.screenshot(out_path, transparent_background=False, return_img=False)
pl.close()
print(f"Simulation figure saved to {out_path}")
