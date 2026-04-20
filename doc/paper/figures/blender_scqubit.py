"""
blender_scqubit.py — Case Study B 3D hero scene in Blender + Cycles (GPU).

Produces a high-quality photorealistic render of the superconducting-qubit
flux sensor: sapphire substrate, x-mon transmon cross, SQUID loop with two
visible Josephson junctions, purple flux-bias line, green meandering readout
resonator, and a navy flux arrow threading the SQUID loop.

Output: blender_scqubit.png  (transparent background for compositing).

Run:  blender --background --python blender_scqubit.py -- out.png
"""
import bpy
import math
import os
import sys

# -------------------------------------------------------------------
OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "blender_scqubit.png")
if "--" in sys.argv:
    user_args = sys.argv[sys.argv.index("--") + 1:]
    if user_args:
        OUTFILE = user_args[0]

# -------------------------------------------------------------------
# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 512
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 8
scene.render.resolution_x = 2400
scene.render.resolution_y = 1800
scene.render.resolution_percentage = 100
scene.render.filepath = OUTFILE
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.view_settings.view_transform = "Filmic"
scene.view_settings.look = "Medium High Contrast"

prefs = bpy.context.preferences
cprefs = prefs.addons["cycles"].preferences
cprefs.compute_device_type = "OPTIX"
for dev in cprefs.devices:
    dev.use = True

# World
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
nt = world.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.97, 0.93, 0.85, 1.0)
bg.inputs["Strength"].default_value = 0.20   # less ambient to preserve shadows
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

# -------------------------------------------------------------------
# Materials
def mat_principled(name, color, metallic=0.0, roughness=0.5,
                   transmission=0.0, coat=0.0, emission=None,
                   emission_strength=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    try: bsdf.inputs["Transmission Weight"].default_value = transmission
    except Exception: pass
    try: bsdf.inputs["Coat Weight"].default_value = coat
    except Exception: pass
    if emission:
        try:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission_strength
        except Exception: pass
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m

mat_sapphire = mat_principled("sapphire", (0.24, 0.36, 0.58),
                               metallic=0.0, roughness=0.85, coat=0.0)
mat_gold     = mat_principled("gold",     (0.95, 0.70, 0.24),
                               metallic=1.0,  roughness=0.18)
mat_red_jj   = mat_principled("red_jj",   (0.88, 0.20, 0.17),
                               metallic=0.3, roughness=0.28,
                               emission=(0.95, 0.30, 0.25),
                               emission_strength=3.0)
mat_purple   = mat_principled("purple",   (0.55, 0.22, 0.62),
                               metallic=0.85, roughness=0.25)
mat_green    = mat_principled("green",    (0.15, 0.55, 0.30),
                               metallic=0.85, roughness=0.25)
mat_navy     = mat_principled("navy",     (0.22, 0.32, 0.65),
                               metallic=0.4, roughness=0.35,
                               emission=(0.30, 0.45, 0.90),
                               emission_strength=1.2)

# -------------------------------------------------------------------
# Convention: all primitive_cube_add use size=1 -> edge-length 1, so setting
# scale=(Wx, Wy, Wz) gives FULL edge-lengths (Wx, Wy, Wz) and half-extents
# (Wx/2, Wy/2, Wz/2).  Positions are centers.
#
# Sapphire substrate
CHIP_W, CHIP_D, CHIP_H = 12.0, 8.0, 0.30
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, CHIP_H/2))
chip = bpy.context.active_object
chip.scale = (CHIP_W, CHIP_D, CHIP_H)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bpy.ops.object.modifier_add(type="BEVEL")
chip.modifiers["Bevel"].width = 0.02
chip.modifiers["Bevel"].segments = 3
chip.data.materials.append(mat_sapphire)

# -------------------------------------------------------------------
# Transmon x-mon cross
TRANSMON_CX = -2.0
ARM_LEN = 2.6
ARM_WID = 1.0
ARM_THK = 0.10
Z_METAL = CHIP_H + ARM_THK/2   # metal bottom flush with chip top (true contact)
# Use a single cross mesh built by unioning two rectangles so there's no
# z-fight seam at the center (avoids the "black square" artifact).
bpy.ops.mesh.primitive_cube_add(size=1, location=(TRANSMON_CX, 0, Z_METAL))
arm_x = bpy.context.active_object
arm_x.name = "arm_x"
arm_x.scale = (ARM_LEN, ARM_WID, ARM_THK)   # FULL edge lengths
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

bpy.ops.mesh.primitive_cube_add(size=1, location=(TRANSMON_CX, 0, Z_METAL))
arm_y = bpy.context.active_object
arm_y.name = "arm_y"
arm_y.scale = (ARM_WID, ARM_LEN, ARM_THK)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Boolean union
bpy.context.view_layer.objects.active = arm_x
mod = arm_x.modifiers.new("union", type="BOOLEAN")
mod.object = arm_y
mod.operation = "UNION"
bpy.ops.object.modifier_apply(modifier="union")
bpy.data.objects.remove(arm_y, do_unlink=True)
bpy.ops.object.modifier_add(type="BEVEL")
arm_x.modifiers["Bevel"].width = 0.01
arm_x.data.materials.append(mat_gold)

# -------------------------------------------------------------------
# SQUID loop (hollow rectangle)
SQ_CX = TRANSMON_CX + ARM_LEN/2 + 1.8   # gap to arm tip = 1.8 - SQ_W/2
SQ_W, SQ_H, SQ_TH = 2.0, 1.4, 0.18
SQ_Z = 0.12
Z_SQ  = CHIP_H + SQ_Z/2        # SQUID loop bottom flush with chip top

def add_ring_square(cx, cy, cz, w, h, th, z_h, material):
    """w, h, z_h are FULL edge lengths of the outer box.  th = wall thickness."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    outer = bpy.context.active_object
    outer.scale = (w, h, z_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    inner = bpy.context.active_object
    inner.scale = (w - 2*th, h - 2*th, z_h * 1.2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    outer.select_set(True)
    bpy.context.view_layer.objects.active = outer
    mod = outer.modifiers.new("sub", type="BOOLEAN")
    mod.object = inner
    mod.operation = "DIFFERENCE"
    bpy.ops.object.modifier_apply(modifier="sub")
    bpy.data.objects.remove(inner, do_unlink=True)
    outer.data.materials.append(material)
    return outer

squid = add_ring_square(SQ_CX, 0, Z_SQ, SQ_W, SQ_H, SQ_TH, SQ_Z, mat_gold)

# Two gold bridges from +x cross-arm tip to the SQUID's left edge.
# Slight overlap into both arm and SQUID so joints look continuous.
bx0 = TRANSMON_CX + ARM_LEN/2 - 0.05  # small overlap into arm
bx1 = SQ_CX - SQ_W/2 + 0.05           # small overlap into SQUID outer
for y_sign in (+1, -1):
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=((bx0 + bx1)/2,
                  y_sign*(SQ_H/2 - SQ_TH/2),
                  Z_METAL))
    b = bpy.context.active_object
    b.scale = (bx1 - bx0, SQ_TH, ARM_THK)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    b.data.materials.append(mat_gold)

# Two Josephson junctions (red pads inserted into top/bottom SQUID segments)
for y_sign in (+1, -1):
    y = y_sign * (SQ_H/2 - SQ_TH/2)
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX, y, Z_SQ + SQ_Z/2 + 0.01))
    jj = bpy.context.active_object
    jj.scale = (0.30, SQ_TH * 1.05, SQ_Z * 0.6)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    jj.data.materials.append(mat_red_jj)

# -------------------------------------------------------------------
# Purple flux-bias line: a U-bend hugging the +x side of the SQUID, visibly
# ELEVATED above the chip (so camera sees it).
def add_wire_poly(points, radius, material):
    curve_data = bpy.data.curves.new("wire", type="CURVE")
    curve_data.dimensions = "3D"
    sp = curve_data.splines.new("BEZIER")
    sp.bezier_points.add(len(points)-1)
    for i, (px, py, pz) in enumerate(points):
        bp = sp.bezier_points[i]
        bp.co = (px, py, pz)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 8
    obj = bpy.data.objects.new("wire", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj

PURPLE_ELEV = 0.55  # how high the bias line lifts above the chip
add_wire_poly(
    [(CHIP_W/2 + 0.8, -3.5, CHIP_H + PURPLE_ELEV),
     (CHIP_W/2 - 0.8, -2.4, CHIP_H + PURPLE_ELEV*0.9),
     (SQ_CX + SQ_W/2 + 0.35, -0.6, CHIP_H + PURPLE_ELEV*0.6),
     (SQ_CX + SQ_W/2 + 0.35, -0.15, Z_SQ + SQ_Z + 0.02),
     (SQ_CX + SQ_W/2 + 0.35,  0.15, Z_SQ + SQ_Z + 0.02),
     (SQ_CX + SQ_W/2 + 0.35,  0.6, CHIP_H + PURPLE_ELEV*0.6),
     (CHIP_W/2 - 0.8,  2.4, CHIP_H + PURPLE_ELEV*0.9),
     (CHIP_W/2 + 0.8,  3.5, CHIP_H + PURPLE_ELEV)],
    radius=0.10, material=mat_purple)

# -------------------------------------------------------------------
# Navy flux arrow (vertical through the SQUID)
ARROW_LEN = 2.0
bpy.ops.mesh.primitive_cylinder_add(radius=0.07,
    depth=ARROW_LEN, location=(SQ_CX, 0, Z_SQ + SQ_Z + ARROW_LEN/2 + 0.10),
    vertices=48)
bpy.context.active_object.data.materials.append(mat_navy)
bpy.ops.mesh.primitive_cone_add(radius1=0.22, radius2=0.0, depth=0.35,
    location=(SQ_CX, 0, Z_SQ + SQ_Z + ARROW_LEN + 0.27), vertices=48)
bpy.context.active_object.data.materials.append(mat_navy)

# -------------------------------------------------------------------
# Readout resonator: green meandering line clearly on the -x arm side of chip
mx_start = TRANSMON_CX - ARM_LEN - 0.35
pts = [(mx_start, 0, Z_METAL + 0.03)]
dx = 0.32
dy = 0.55
n_meanders = 4
x = mx_start
y = 0
for i in range(n_meanders):
    pts.append((x,      y + dy, Z_METAL + 0.03))
    x -= dx
    pts.append((x,      y + dy, Z_METAL + 0.03))
    pts.append((x,      y - dy, Z_METAL + 0.03))
    x -= dx
    pts.append((x,      y - dy, Z_METAL + 0.03))
pts.append((x, 0, Z_METAL + 0.03))
pts.append((x - 0.35, 0, Z_METAL + 0.03))

curve_data = bpy.data.curves.new("meander", type="CURVE")
curve_data.dimensions = "3D"
sp = curve_data.splines.new("POLY")
sp.points.add(len(pts)-1)
for i, (px, py, pz) in enumerate(pts):
    sp.points[i].co = (px, py, pz, 1.0)
curve_data.bevel_depth = 0.075
curve_data.bevel_resolution = 8
obj = bpy.data.objects.new("meander", curve_data)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(mat_green)

# Small green pad at chip-edge exit
bpy.ops.mesh.primitive_cube_add(size=1,
    location=(x - 0.7, 0, Z_METAL + 0.02))
pad = bpy.context.active_object
pad.scale = (0.12, 0.20, ARM_THK/2)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
pad.data.materials.append(mat_green)

# -------------------------------------------------------------------
# xy-control line on +y side of chip (blue short wire entering the chip)
mat_blue = mat_principled("blue_xy", (0.20, 0.40, 0.85), metallic=0.8,
                           roughness=0.25)
add_wire_poly(
    [(TRANSMON_CX - 0.2, CHIP_D/2 + 1.0, CHIP_H + 0.55),
     (TRANSMON_CX - 0.2, ARM_LEN + 0.3,  CHIP_H + 0.40),
     (TRANSMON_CX - 0.2, ARM_LEN + 0.05, Z_METAL + 0.02)],
    radius=0.06, material=mat_blue)

# -------------------------------------------------------------------
# Camera
bpy.ops.object.camera_add(location=(-1.0, -12.5, 6.0),
                          rotation=(math.radians(62), 0, math.radians(-5)))
cam = bpy.context.active_object
cam.data.lens = 62
cam.data.dof.use_dof = False
scene.camera = cam

bpy.ops.object.empty_add(type="PLAIN_AXES",
                         location=(-0.2, 0, CHIP_H + 0.35))
target = bpy.context.active_object
tc = cam.constraints.new(type="TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"

# -------------------------------------------------------------------
# Lighting: 3-point + bottom bounce
bpy.ops.object.light_add(type="AREA", location=(6, -4, 12))
key = bpy.context.active_object
key.data.energy = 2200
key.data.size = 8
key.data.color = (1.0, 0.96, 0.88)
key.rotation_euler = (math.radians(35), math.radians(25), 0)

bpy.ops.object.light_add(type="AREA", location=(-10, -3, 9))
fill = bpy.context.active_object
fill.data.energy = 1100
fill.data.size = 12
fill.data.color = (0.80, 0.88, 1.0)
fill.rotation_euler = (math.radians(40), math.radians(-25), 0)

bpy.ops.object.light_add(type="AREA", location=(2, 10, 10))
rim = bpy.context.active_object
rim.data.energy = 1500
rim.data.size = 8
rim.data.color = (0.95, 0.92, 1.0)
rim.rotation_euler = (math.radians(-30), 0, 0)

bpy.ops.object.light_add(type="AREA", location=(0, 0, -3))
bounce = bpy.context.active_object
bounce.data.energy = 500
bounce.data.size = 18
bounce.data.color = (1.0, 0.96, 0.88)
bounce.rotation_euler = (math.pi, 0, 0)

# Subtle sunlight for highlights
bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
sun = bpy.context.active_object
sun.data.energy = 2.0
sun.data.angle = math.radians(4)  # narrower angle -> sharper contact shadows
sun.rotation_euler = (math.radians(25), math.radians(15), 0)

# -------------------------------------------------------------------
print(f"Rendering to {OUTFILE}")
bpy.ops.render.render(write_still=True)
print(f"Saved: {OUTFILE}")
