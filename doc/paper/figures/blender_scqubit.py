"""
blender_scqubit.py — Renders the scqubit (Case Study B) 3D scene in Blender.

Run: blender --background --python blender_scqubit.py -- out.png
"""
import bpy
import bmesh
import math
import os
import sys

# =========================================================
# Config
# =========================================================
OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "blender_scqubit.png")
if "--" in sys.argv:
    user_args = sys.argv[sys.argv.index("--") + 1:]
    if user_args:
        OUTFILE = user_args[0]

# =========================================================
# Reset scene
# =========================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Engine + rendering settings
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.render.resolution_x = 2000
scene.render.resolution_y = 1600
scene.cycles.samples = 256
scene.view_settings.view_transform = "Filmic"
scene.view_settings.look = "Medium High Contrast"
scene.render.film_transparent = False
scene.render.resolution_percentage = 100
scene.render.filepath = OUTFILE
scene.render.image_settings.file_format = "PNG"

# Enable GPU
prefs = bpy.context.preferences
cprefs = prefs.addons["cycles"].preferences
cprefs.compute_device_type = "OPTIX"
for dev in cprefs.devices:
    dev.use = True

# World background: warm cream
world = bpy.data.worlds.new("World") if not bpy.data.worlds else bpy.data.worlds[0]
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.98, 0.95, 0.88, 1.0)
bg.inputs["Strength"].default_value = 0.35

# =========================================================
# Materials
# =========================================================
def mat_principled(name, color, metallic=0.0, roughness=0.5, transmission=0.0,
                   emission=None, emission_strength=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    # remove existing
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    try:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    except Exception:
        pass
    if emission:
        try:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission_strength
        except Exception:
            pass
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m

mat_sapphire = mat_principled("sapphire", (0.55, 0.67, 0.85), metallic=0.15,
                               roughness=0.08)
mat_gold     = mat_principled("gold", (0.95, 0.70, 0.25), metallic=1.0,
                               roughness=0.18)
mat_gold_dk  = mat_principled("gold_dk", (0.65, 0.45, 0.12), metallic=0.9,
                               roughness=0.40)
mat_red_jj   = mat_principled("red_jj", (0.85, 0.18, 0.15), metallic=0.2,
                               roughness=0.35,
                               emission=(0.9, 0.25, 0.2), emission_strength=0.8)
mat_purple   = mat_principled("purple", (0.55, 0.22, 0.62), metallic=0.8,
                               roughness=0.3)
mat_green_m  = mat_principled("green_m", (0.15, 0.55, 0.30), metallic=0.85,
                               roughness=0.30)
mat_navy     = mat_principled("navy", (0.20, 0.28, 0.55), metallic=0.6,
                               roughness=0.40)
mat_ground   = mat_principled("ground", (0.92, 0.88, 0.78), metallic=0.0,
                               roughness=0.9)

# =========================================================
# Ground + stage plane
# =========================================================
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.5))
ground = bpy.context.active_object
ground.data.materials.append(mat_ground)

# =========================================================
# Sapphire substrate (chip)
# =========================================================
CHIP_W = 12.0   # x
CHIP_D = 8.0    # y
CHIP_H = 0.35   # z (thickness)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, CHIP_H/2))
chip = bpy.context.active_object
chip.scale = (CHIP_W/2, CHIP_D/2, CHIP_H/2)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
# bevel corners for soft edges
bpy.ops.object.modifier_add(type="BEVEL")
chip.modifiers["Bevel"].width = 0.02
chip.modifiers["Bevel"].segments = 3
chip.data.materials.append(mat_sapphire)

# =========================================================
# Transmon cross (x-mon): two orthogonal gold pads on chip surface
# =========================================================
ARM_LEN  = 2.8
ARM_WID  = 1.00
ARM_THK  = 0.12
# Shift the whole transmon toward -x so there's room for SQUID + extras on +x
TRANSMON_CX = -1.2
# Horizontal arm (along x)
bpy.ops.mesh.primitive_cube_add(size=1,
    location=(TRANSMON_CX, 0, CHIP_H + ARM_THK/2))
arm_x = bpy.context.active_object
arm_x.scale = (ARM_LEN, ARM_WID/2, ARM_THK/2)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
arm_x.data.materials.append(mat_gold)
# Vertical arm (along y)
bpy.ops.mesh.primitive_cube_add(size=1,
    location=(TRANSMON_CX, 0, CHIP_H + ARM_THK/2))
arm_y = bpy.context.active_object
arm_y.scale = (ARM_WID/2, ARM_LEN, ARM_THK/2)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
arm_y.data.materials.append(mat_gold)

# =========================================================
# SQUID loop at end of +x arm
# =========================================================
SQ_CX = TRANSMON_CX + ARM_LEN + 1.80
SQ_CY = 0.0
SQ_CZ = CHIP_H + ARM_THK + 0.06
SQ_W  = 2.20  # outer width
SQ_H  = 1.50  # outer height
SQ_TH = 0.18  # wall thickness
SQ_Z  = 0.14

# Outer rectangle (ring)
def add_ring_square(cx, cy, cz, w, h, th, z_height, material):
    """Build a rectangular ring by subtracting inner box from outer."""
    # outer
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    outer = bpy.context.active_object
    outer.scale = (w/2, h/2, z_height/2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # inner
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    inner = bpy.context.active_object
    inner.scale = ((w-2*th)/2, (h-2*th)/2, z_height)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # boolean difference
    outer.select_set(True)
    bpy.context.view_layer.objects.active = outer
    mod = outer.modifiers.new("subtract", type="BOOLEAN")
    mod.object = inner
    mod.operation = "DIFFERENCE"
    bpy.ops.object.modifier_apply(modifier="subtract")
    bpy.data.objects.remove(inner, do_unlink=True)
    outer.data.materials.append(material)
    return outer

sq_ring = add_ring_square(SQ_CX, SQ_CY, SQ_CZ, SQ_W, SQ_H, SQ_TH, SQ_Z, mat_gold)

# Two Josephson junctions — wider red pads in the middle of top/bottom segments
jj_w = 0.30
jj_z = SQ_Z + 0.04
for y_sign in (+1, -1):
    y = y_sign * (SQ_H/2 - SQ_TH/2)
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX, y, SQ_CZ + SQ_Z/2 + 0.02))
    jj = bpy.context.active_object
    jj.scale = (jj_w/2, SQ_TH*0.62, jj_z/2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    jj.data.materials.append(mat_red_jj)

# Connect SQUID to arm with two parallel narrow wires (more realistic)
bridge_x_start = TRANSMON_CX + ARM_LEN
bridge_x_end   = SQ_CX - SQ_W/2
for y_sign in (+1, -1):
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=((bridge_x_start + bridge_x_end)/2,
                  y_sign * (SQ_H/2 - SQ_TH/2),
                  CHIP_H + ARM_THK/2))
    bridge = bpy.context.active_object
    bridge.scale = ((bridge_x_end - bridge_x_start)/2, SQ_TH/2, ARM_THK/2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bridge.data.materials.append(mat_gold)

# =========================================================
# Flux-bias line (purple): a U-shaped wire coming from the bottom right,
# passing close to the SQUID loop
# =========================================================
def add_wire(points, radius, material, bevel_depth=None):
    # create bezier curve
    curve_data = bpy.data.curves.new("wire", type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points)-1)
    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = p
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    curve_data.bevel_depth = bevel_depth or radius
    curve_data.bevel_resolution = 6
    obj = bpy.data.objects.new("wire", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj

# Flux-bias line (purple): a U-bend hugging the +x side of the SQUID loop
add_wire(
    [(CHIP_W/2 - 0.1, -3.2, CHIP_H + 0.10),
     (SQ_CX + SQ_W/2 + 0.30, -0.8, CHIP_H + 0.18),
     (SQ_CX + SQ_W/2 + 0.30, -0.3, SQ_CZ + 0.15),
     (SQ_CX + SQ_W/2 + 0.30,  0.3, SQ_CZ + 0.15),
     (SQ_CX + SQ_W/2 + 0.30,  0.8, CHIP_H + 0.18),
     (CHIP_W/2 - 0.1,  3.2, CHIP_H + 0.10)],
    radius=0.10, material=mat_purple)

# =========================================================
# Magnetic-field line (navy): small dashed-looking torus through the SQUID loop
# =========================================================
# A translucent navy arrow (cylinder + cone) pointing UP through the SQUID loop,
# representing the external magnetic flux Phi_ext threading the loop.
arrow_len = 1.8
bpy.ops.mesh.primitive_cylinder_add(radius=0.055,
    depth=arrow_len, location=(SQ_CX, 0, SQ_CZ + arrow_len/2 + 0.05),
    vertices=32)
bpy.context.active_object.data.materials.append(mat_navy)
bpy.ops.mesh.primitive_cone_add(radius1=0.18, radius2=0.0, depth=0.30,
    location=(SQ_CX, 0, SQ_CZ + arrow_len + 0.20), vertices=32)
bpy.context.active_object.data.materials.append(mat_navy)
# Small "phi" label sphere as a hint marker (actually a glow sphere)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12,
    location=(SQ_CX + 0.35, 0, SQ_CZ + arrow_len + 0.25),
    segments=24, ring_count=12)
bpy.context.active_object.data.materials.append(
    mat_principled("phi_glow", (0.25, 0.35, 0.70), metallic=0.0, roughness=0.3,
                   emission=(0.2, 0.35, 0.85), emission_strength=2.5))

# =========================================================
# Readout resonator (green meander) laid out as a proper curve
# on the -x side of the chip, taking a meander path.
# =========================================================
mx0 = TRANSMON_CX - ARM_LEN - 0.6
pts = []
dx = -0.5; dy = 0.5
n_meanders = 4
x = mx0
y = 0
pts.append((x, y, CHIP_H + ARM_THK/2 + 0.03))
for i in range(n_meanders):
    pts.append((x,       y + dy, CHIP_H + ARM_THK/2 + 0.03))
    pts.append((x + dx,  y + dy, CHIP_H + ARM_THK/2 + 0.03))
    pts.append((x + dx,  y - dy, CHIP_H + ARM_THK/2 + 0.03))
    pts.append((x + 2*dx, y - dy, CHIP_H + ARM_THK/2 + 0.03))
    x += 2*dx
pts.append((x, y, CHIP_H + ARM_THK/2 + 0.03))
pts.append((x - 0.8, y, CHIP_H + ARM_THK/2 + 0.03))  # exit

# polyline (not bezier) via curve
curve_data = bpy.data.curves.new("meander", type="CURVE")
curve_data.dimensions = "3D"
spline = curve_data.splines.new("POLY")
spline.points.add(len(pts)-1)
for i, (px, py, pz) in enumerate(pts):
    spline.points[i].co = (px, py, pz, 1.0)
curve_data.bevel_depth = 0.07
curve_data.bevel_resolution = 6
obj = bpy.data.objects.new("meander", curve_data)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(mat_green_m)

# =========================================================
# Camera
# =========================================================
bpy.ops.object.camera_add(location=(0, -10, 6.5),
                          rotation=(math.radians(58), 0, math.radians(-10)))
cam = bpy.context.active_object
cam.data.lens = 55
cam.data.dof.use_dof = False
scene.camera = cam

# Subtle aim: track to chip center (slight bias toward SQUID)
bpy.ops.object.empty_add(type="PLAIN_AXES",
                         location=(0.5, 0, CHIP_H + 0.3))
target = bpy.context.active_object
tc = cam.constraints.new(type="TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"

# =========================================================
# Lighting: three-point
# =========================================================
# Key light (warm) from upper-right
bpy.ops.object.light_add(type="AREA", location=(6, -6, 12))
key = bpy.context.active_object
key.data.energy = 1800
key.data.size = 8
key.data.color = (1.0, 0.95, 0.85)
# Fill (cool) from upper-left
bpy.ops.object.light_add(type="AREA", location=(-10, -4, 8))
fill = bpy.context.active_object
fill.data.energy = 900
fill.data.size = 10
fill.data.color = (0.80, 0.88, 1.0)
# Back rim from behind
bpy.ops.object.light_add(type="AREA", location=(2, 12, 8))
rim = bpy.context.active_object
rim.data.energy = 1200
rim.data.size = 8
rim.data.color = (0.92, 0.90, 1.0)
# Fill-up bounce from below
bpy.ops.object.light_add(type="AREA", location=(0, 0, -2))
bounce = bpy.context.active_object
bounce.data.energy = 300
bounce.data.size = 15
bounce.data.color = (0.98, 0.95, 0.88)
bounce.rotation_euler = (math.pi, 0, 0)

# =========================================================
# Render
# =========================================================
print(f"Rendering to {OUTFILE}")
bpy.ops.render.render(write_still=True)
print(f"Saved: {OUTFILE}")
