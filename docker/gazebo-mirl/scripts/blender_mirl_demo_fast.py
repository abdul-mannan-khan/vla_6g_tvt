#!/usr/bin/env python3
"""
MI-RL 6G UAV Relay Demo - Fast Blender Animation (Workbench)

Uses Workbench renderer for fast headless rendering.

Run with: blender --background --python blender_mirl_demo_fast.py

Output: /root/videos/mi_rl_blender_demo.mp4
"""

import bpy
import math

OUTPUT_PATH = "/root/videos/mi_rl_blender_demo.mp4"

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Scene settings
scene = bpy.context.scene
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 300  # 10 seconds
scene.render.film_transparent = False

# Use Workbench for fast rendering
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.color_type = 'MATERIAL'
scene.display.shading.show_shadows = True

# Output settings
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
scene.render.ffmpeg.audio_codec = 'NONE'
scene.render.filepath = OUTPUT_PATH

# World background
world = bpy.data.worlds.new("World")
scene.world = world
scene.world.color = (0.4, 0.6, 0.9)

# ============== HELPER FUNCTIONS ==============

def add_color(obj, color):
    """Add viewport color to object."""
    mat = bpy.data.materials.new(name=f"{obj.name}_mat")
    mat.diffuse_color = (*color, 1.0)
    obj.data.materials.append(mat)
    obj.color = (*color, 1.0)

def create_building(name, location, size, color):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    building = bpy.context.active_object
    building.name = name
    building.scale = size
    add_color(building, color)
    return building

def create_ground():
    bpy.ops.mesh.primitive_plane_add(size=200, location=(50, 50, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    add_color(ground, (0.3, 0.35, 0.25))

    # Roads
    bpy.ops.mesh.primitive_cube_add(size=1, location=(50, 50, 0.05))
    road = bpy.context.active_object
    road.name = "Road_EW"
    road.scale = (150, 12, 0.1)
    add_color(road, (0.15, 0.15, 0.18))

    bpy.ops.mesh.primitive_cube_add(size=1, location=(50, 50, 0.05))
    road2 = bpy.context.active_object
    road2.name = "Road_NS"
    road2.scale = (12, 150, 0.1)
    add_color(road2, (0.15, 0.15, 0.18))

def create_base_station(location):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=10, location=location)
    tower = bpy.context.active_object
    tower.name = "BS_Tower"
    add_color(tower, (0.7, 0.2, 0.2))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(location[0], location[1], location[2]+5.5))
    beacon = bpy.context.active_object
    beacon.name = "Beacon"
    add_color(beacon, (1.0, 0.2, 0.1))
    return tower

def create_uav(name, location):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    body = bpy.context.active_object
    body.name = name
    body.scale = (2.5, 2.5, 0.6)
    add_color(body, (0.3, 0.3, 0.35))

    # Propellers
    for dx, dy in [(1.8, 1.8), (1.8, -1.8), (-1.8, 1.8), (-1.8, -1.8)]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.9, depth=0.1,
            location=(location[0]+dx, location[1]+dy, location[2]+0.4))
        prop = bpy.context.active_object
        prop.parent = body
        add_color(prop, (0.2, 0.2, 0.25))
    return body

def create_user(name, location, color):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.6, depth=2.2, location=(location[0], location[1], location[2]+1.1))
    body = bpy.context.active_object
    body.name = name
    add_color(body, color)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(location[0], location[1], location[2]+2.5))
    head = bpy.context.active_object
    head.parent = body
    add_color(head, (0.9, 0.75, 0.6))
    return body

def create_signal_beam(name, start, end, color):
    dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    mid = ((start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=length, location=mid)
    beam = bpy.context.active_object
    beam.name = name

    direction = (dx/length, dy/length, dz/length)
    beam.rotation_euler = (
        math.acos(direction[2]),
        0,
        math.atan2(direction[1], direction[0]) + math.pi/2
    )
    add_color(beam, color)
    return beam

# ============== CREATE SCENE ==============

print("Creating scene...")

create_ground()

# Buildings
buildings = [
    ("Tower_NE", (80, 80, 25), (16, 16, 50), (0.25, 0.45, 0.8)),
    ("Tower_NW", (15, 85, 20), (18, 14, 40), (0.85, 0.87, 0.9)),
    ("Tower_SE", (85, 15, 22), (16, 16, 44), (0.15, 0.3, 0.6)),
    ("Tower_SW", (20, 20, 17), (14, 14, 34), (0.5, 0.35, 0.65)),
]
for name, loc, size, color in buildings:
    create_building(name, loc, size, color)

# Base station
bs = create_base_station((80, 80, 55))

# UAV
uav = create_uav("UAV_Relay", (50, 50, 35))

# Users
users = [
    ("User_1", (38, 58, 0), (0.9, 0.2, 0.15)),
    ("User_2", (75, 40, 0), (0.15, 0.8, 0.25)),
    ("User_3", (42, 32, 0), (0.95, 0.6, 0.1)),
    ("User_4", (60, 72, 0), (0.95, 0.9, 0.15)),
    ("User_5", (88, 60, 0), (0.9, 0.15, 0.7)),
]
user_objs = []
for name, loc, color in users:
    user_objs.append(create_user(name, loc, color))

# Signal beams (static at optimal position)
uav_pos = (52, 54, 35)
bs_pos = (80, 80, 60)
beam_bs = create_signal_beam("Beam_BS_UAV", bs_pos, uav_pos, (0.2, 1.0, 0.3))
for i, (name, loc, color) in enumerate(users):
    create_signal_beam(f"Beam_{i}", uav_pos, (loc[0], loc[1], 2.5), (0.3, 0.8, 1.0))

# ============== LIGHTING ==============

print("Setting up lighting...")

bpy.ops.object.light_add(type='SUN', location=(100, 100, 100))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 5
sun.rotation_euler = (math.radians(45), math.radians(30), 0)

bpy.ops.object.light_add(type='SUN', location=(-50, -50, 80))
fill = bpy.context.active_object
fill.name = "Fill"
fill.data.energy = 2

# ============== CAMERA ==============

print("Setting up camera...")

bpy.ops.object.camera_add(location=(-30, -30, 60))
camera = bpy.context.active_object
camera.name = "MainCamera"
camera.rotation_euler = (math.radians(60), 0, math.radians(-45))
scene.camera = camera

# Camera tracks UAV
bpy.ops.object.constraint_add(type='TRACK_TO')
camera.constraints["Track To"].target = uav
camera.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
camera.constraints["Track To"].up_axis = 'UP_Y'

# ============== ANIMATION ==============

print("Creating animation...")

# UAV movement: random -> optimization -> optimal
uav_keyframes = [
    (1, (20, 20, 35)),       # Start
    (50, (35, 80, 40)),      # Random 1
    (100, (85, 25, 38)),     # Random 2
    (150, (25, 60, 42)),     # Random 3
    (180, (52, 54, 36)),     # Converging
    (220, (52, 54, 35)),     # Optimal
    (300, (52, 54, 35)),     # Hold
]

for frame, loc in uav_keyframes:
    uav.location = loc
    uav.keyframe_insert(data_path="location", frame=frame)

# Camera orbit
camera.location = (-30, -30, 60)
camera.keyframe_insert(data_path="location", frame=1)
camera.location = (-50, 30, 65)
camera.keyframe_insert(data_path="location", frame=300)

# ============== RENDER ==============

print(f"Rendering to {OUTPUT_PATH}...")
print(f"Frames: {scene.frame_start} to {scene.frame_end}")

bpy.ops.render.render(animation=True)

print("Complete!")
print(f"Video: {OUTPUT_PATH}")
