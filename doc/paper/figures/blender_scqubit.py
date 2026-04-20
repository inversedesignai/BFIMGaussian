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
scene.render.film_transparent = True  # composite on white in the final overlay
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
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)     # pure white
bg.inputs["Strength"].default_value = 0.30
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
# Bright orange-red for magnetic flux lines (contrasts with blue sapphire)
mat_flux     = mat_principled("flux",     (0.88, 0.22, 0.10),
                               metallic=0.0, roughness=0.28,
                               emission=(0.96, 0.30, 0.15),
                               emission_strength=0.8)
# Amber insulating layer for JJ barrier (between two gold pads)
mat_jj_barrier = mat_principled("jj_barrier", (0.92, 0.32, 0.14),
                                 metallic=0.0, roughness=0.35,
                                 emission=(1.0, 0.42, 0.18),
                                 emission_strength=0.9)

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

# Two Josephson junctions: realistic overlap-barrier-overlap structure.
#   - Two gold pads overlap each other at a thin amber barrier layer.
#   - Pads protrude slightly above/below the SQUID loop surface, so the
#     JJ stack is clearly distinguishable from the ring.
#
# Layout: for each JJ on the top/bottom SQUID segment, build three parts:
#   * Left pad (gold, short stub into the loop wall)
#   * Right pad (gold, short stub into the loop wall, overlapping the left)
#   * Barrier (thin amber layer sandwiched between the two pads)
for y_sign in (+1, -1):
    y = y_sign * (SQ_H/2 - SQ_TH/2)
    # Pad dimensions
    pad_w = 0.22    # width of each pad along the loop direction
    pad_d = SQ_TH * 1.50   # into/out of loop (wider than ring)
    pad_h = SQ_Z * 1.40
    barrier_w = 0.06
    overlap  = 0.02
    # left pad (slightly above)
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX - (pad_w - barrier_w)/2 - overlap,
                  y, Z_SQ + SQ_Z/2 + pad_h/2 * 0.35))
    lp = bpy.context.active_object
    lp.scale = (pad_w, pad_d, pad_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    lp.data.materials.append(mat_gold)
    # right pad (slightly lower)
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX + (pad_w - barrier_w)/2 + overlap,
                  y, Z_SQ + SQ_Z/2 + pad_h/2 * 0.15))
    rp = bpy.context.active_object
    rp.scale = (pad_w, pad_d, pad_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    rp.data.materials.append(mat_gold)
    # thin amber barrier (in the overlap region between the two pads)
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX, y, Z_SQ + SQ_Z/2 + pad_h/2 * 0.28))
    br = bpy.context.active_object
    br.scale = (barrier_w, pad_d * 1.02, pad_h * 0.9)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    br.data.materials.append(mat_jj_barrier)

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

# Flux-bias line: a thin flat-on-chip trace (purple) routed from two chip-edge
# bond pads, hugging the +x side of the SQUID loop at chip-surface height.
# The trace is a RIBBON (flat rectangular wire), not a floating tube.

BIAS_Z = CHIP_H + 0.02            # trace sits on chip surface
BIAS_HALFWIDTH = 0.09
BIAS_THK = 0.04
PAD_SIZE = 0.55                   # square bond pads at chip edges
PAD_ZH   = 0.06

# Two bond pads at the +x chip edge (y = ±3.4)
for y_pad in (-3.2, 3.2):
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(CHIP_W/2 - PAD_SIZE/2 - 0.2, y_pad, BIAS_Z + PAD_ZH/2))
    pad = bpy.context.active_object
    pad.scale = (PAD_SIZE, PAD_SIZE, PAD_ZH)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pad.data.materials.append(mat_purple)

# Trace segments: lay out as a ribbon following a polyline of waypoints, but
# built from rectangular slabs at the chip-surface height.
# Waypoints in (x, y):
bias_path = [
    (CHIP_W/2 - 0.5, -3.2),
    (SQ_CX + SQ_W/2 + 0.30, -2.0),
    (SQ_CX + SQ_W/2 + 0.30, -0.35),   # approaches SQUID from bottom-right
    (SQ_CX + SQ_W/2 + 0.30,  0.35),   # passes around SQUID +x side
    (SQ_CX + SQ_W/2 + 0.30,  2.0),
    (CHIP_W/2 - 0.5,  3.2),
]
for i in range(len(bias_path) - 1):
    (x0, y0), (x1, y1) = bias_path[i], bias_path[i+1]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, BIAS_Z + BIAS_THK/2),
                                    rotation=(0, 0, angle))
    seg = bpy.context.active_object
    seg.scale = (length, 2*BIAS_HALFWIDTH, BIAS_THK)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    seg.data.materials.append(mat_purple)
# Bevel to smooth corners slightly (separate pass per object; skip for speed)

# -------------------------------------------------------------------
# Magnetic flux lines threading the SQUID loop.
# We draw a set of 5 parallel vertical tubes spaced across the loop
# interior, each with an up-arrow cap.  Each tube has a gentle S-curve
# near the SQUID plane to visualise the field bending as it passes through
# the loop (dipole-like).
FLUX_TUBE_R   = 0.035
FLUX_Z_TOP    = Z_SQ + SQ_Z + 2.3
FLUX_Z_BOT    = Z_SQ - 1.6
# Spread positions across the SQUID loop interior in BOTH x and y so the
# tubes are individually visible from the camera (which looks from -y).
interior_x_half = SQ_W/2 - SQ_TH - 0.12
interior_y_half = SQ_H/2 - SQ_TH - 0.12
# 5 lines along x (visible row in camera view)
flux_positions_x = [
    SQ_CX - 0.70*interior_x_half,
    SQ_CX - 0.35*interior_x_half,
    SQ_CX,
    SQ_CX + 0.35*interior_x_half,
    SQ_CX + 0.70*interior_x_half,
]
# offset y slightly for perspective depth (a shallow V pattern)
flux_positions_y = [-0.3*interior_y_half,
                    -0.15*interior_y_half,
                     0.0,
                    -0.15*interior_y_half,
                    -0.3*interior_y_half]

def add_flux_tube(cx, cy):
    curve_data = bpy.data.curves.new("flux_line", type="CURVE")
    curve_data.dimensions = "3D"
    sp = curve_data.splines.new("BEZIER")
    # three anchors: top, plane, bottom — with slight curvature in x
    pts3 = [
        (cx, cy, FLUX_Z_TOP),
        (cx, cy, Z_SQ + SQ_Z/2),
        (cx, cy, FLUX_Z_BOT),
    ]
    sp.bezier_points.add(len(pts3)-1)
    for i, p in enumerate(pts3):
        bp = sp.bezier_points[i]
        bp.co = p
        bp.handle_left_type  = "AUTO"
        bp.handle_right_type = "AUTO"
    curve_data.bevel_depth = FLUX_TUBE_R
    curve_data.bevel_resolution = 8
    obj = bpy.data.objects.new("flux_line", curve_data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat_flux)

for (cx, cy) in zip(flux_positions_x, flux_positions_y):
    add_flux_tube(cx, cy)
    # Upward arrow cap at the top of each line
    bpy.ops.mesh.primitive_cone_add(radius1=0.085, radius2=0.0, depth=0.22,
        location=(cx, cy, FLUX_Z_TOP + 0.10), vertices=28)
    bpy.context.active_object.data.materials.append(mat_flux)

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

# (Removed xy-control line — not needed for this figure)

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
