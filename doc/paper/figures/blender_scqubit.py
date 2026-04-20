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

# Eevee engine with emission-only shading = cartoon / flat editorial look
# (no PBR reflections, so no mirror-image of flux lines on the gold).
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 2400
scene.render.resolution_y = 1800
scene.render.resolution_percentage = 100
scene.render.filepath = OUTFILE
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"    # linear, no Filmic tone map
scene.view_settings.look = "None"

# Freestyle: black outlines for cartoon edge definition
scene.render.use_freestyle = True
scene.render.line_thickness_mode = "ABSOLUTE"
scene.render.line_thickness = 1.2
try:
    vl = scene.view_layers[0]
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    # Wipe default and add a single comprehensive lineset
    while len(fs.linesets) > 0:
        fs.linesets.remove(fs.linesets[-1])
    ls = fs.linesets.new("outline")
    ls.select_silhouette   = True
    ls.select_crease       = True
    ls.select_border       = True
    ls.select_contour      = True
    ls.select_edge_mark    = True
    ls.select_external_contour = True
    ls.select_material_boundary = True
    ls.select_suggestive_contour = False   # suggestive contours clutter; disable
    fs.crease_angle = math.radians(120)    # only outline sharp-ish creases
    ls.select_by_visibility = True
    ls.visibility = "VISIBLE"
    if ls.linestyle is None:
        lstyle = bpy.data.linestyles.new("outline_style")
        ls.linestyle = lstyle
    ls.linestyle.color = (0.05, 0.07, 0.12)
    ls.linestyle.thickness = 1.2
except Exception as e:
    print(f"Freestyle setup warning: {e}")

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
# Flat (toon/editorial) materials: mix emission with a tiny diffuse
# component so that shading gradients still read softly, but the surface
# does not reflect other geometry.  No PBR, no mirror-image of flux
# lines on gold.
def mat_principled(name, color, metallic=0.0, roughness=0.5,
                   transmission=0.0, coat=0.0, emission=None,
                   emission_strength=0.0, alpha=1.0):
    # All params except `color` and `alpha` are ignored in flat mode.
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    # Pure Emission with a touch of diffuse, mixed by an AO-ish factor
    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (*color, 1.0)
    emis.inputs["Strength"].default_value = 1.0
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    # slightly darker diffuse tint so shading gradient stays subtle
    diff.inputs["Color"].default_value = (
        color[0]*0.72, color[1]*0.72, color[2]*0.72, 1.0)
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.18   # mostly emission
    nt.links.new(emis.outputs["Emission"], mix.inputs[1])
    nt.links.new(diff.outputs["BSDF"],     mix.inputs[2])
    if alpha < 1.0:
        trans = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix2 = nt.nodes.new("ShaderNodeMixShader")
        mix2.inputs["Fac"].default_value = alpha
        nt.links.new(trans.outputs["BSDF"], mix2.inputs[1])
        nt.links.new(mix.outputs["Shader"], mix2.inputs[2])
        nt.links.new(mix2.outputs["Shader"], out.inputs["Surface"])
        m.blend_method = "BLEND"
    else:
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
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
# Deep/dark blue JJ marker (user request)
mat_jj_barrier = mat_principled("jj_barrier", (0.08, 0.12, 0.50),
                                 metallic=0.0, roughness=0.35)

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

# Two Josephson junctions, each depicted as a proper 3-element overlap:
#   (a) the SQUID ring is BROKEN at the JJ position — a small rectangular
#       gap cut out of the ring's top/bottom segment.
#   (b) two gold electrodes extend across the gap, each attached to one
#       side of the ring, overlapping at the center.
#   (c) the insulating barrier (dark blue) sits flat at the overlap,
#       visible as a thin stripe separating the two electrode tabs.
# All flat (same z as ring top), no bumps, no cast shadows on the ring.

GAP_W       = 0.28              # ring break width in x (along the ring)
ELECTRODE_W = 0.20              # each electrode tab extends this far
BARRIER_W   = 0.06              # insulator width at the overlap
Z_TOP_RING  = Z_SQ + SQ_Z/2
ELEC_Z      = Z_TOP_RING + 0.002
BARRIER_Z   = Z_TOP_RING + 0.010
ELEC_THK    = 0.018
BARRIER_THK = 0.014

for y_sign in (+1, -1):
    y_center = y_sign * (SQ_H/2 - SQ_TH/2)

    # (a) CUT a gap out of the top/bottom ring segment.
    # Build a small subtraction cube at (SQ_CX, y_center, Z_TOP_RING) of size
    # (GAP_W, SQ_TH*1.2, SQ_Z*1.2) and boolean-difference it from the SQUID ring.
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX, y_center, Z_TOP_RING))
    gap = bpy.context.active_object
    gap.scale = (GAP_W, SQ_TH * 1.40, SQ_Z * 1.40)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.context.view_layer.objects.active = squid
    mod = squid.modifiers.new(f"jj_gap_{y_sign}", type="BOOLEAN")
    mod.object = gap
    mod.operation = "DIFFERENCE"
    bpy.ops.object.modifier_apply(modifier=f"jj_gap_{y_sign}")
    bpy.data.objects.remove(gap, do_unlink=True)

    # (b) Two gold electrode tabs, one on each side of the gap, overlapping.
    # Each tab is ELECTRODE_W wide; they overlap by ELECTRODE_W - GAP_W/2 + 0.02.
    left_cx  = SQ_CX - (GAP_W/2) + ELECTRODE_W/2 - 0.01
    right_cx = SQ_CX + (GAP_W/2) - ELECTRODE_W/2 + 0.01
    for ecx in (left_cx, right_cx):
        bpy.ops.mesh.primitive_cube_add(size=1,
            location=(ecx, y_center, ELEC_Z))
        tab = bpy.context.active_object
        tab.scale = (ELECTRODE_W, SQ_TH * 0.92, ELEC_THK)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        tab.data.materials.append(mat_gold)

    # (c) Dark-blue insulating barrier at the overlap (center of the gap)
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(SQ_CX, y_center, BARRIER_Z))
    barrier = bpy.context.active_object
    barrier.scale = (BARRIER_W, SQ_TH * 1.02, BARRIER_THK)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    barrier.data.materials.append(mat_jj_barrier)

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

# Flux-bias line: a thin flat trace (purple) on the chip surface, drawn
# as a SMOOTH BEZIER CURVE with a rectangular cross-section (bevel object).
# This replaces angular straight-segment slabs whose sharp corners read as
# ugly kinks; bezier AUTO handles give continuous-tangent bends.

BIAS_Z = CHIP_H + 0.025            # trace sits on chip surface
BIAS_HALFW = 0.085
BIAS_THK = 0.04
PAD_SIZE = 0.55
PAD_ZH   = 0.06

# Two square bond pads at the +x chip edge
for y_pad in (-3.3, 3.3):
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(CHIP_W/2 - PAD_SIZE/2 - 0.15, y_pad, BIAS_Z + PAD_ZH/2))
    pad = bpy.context.active_object
    pad.scale = (PAD_SIZE, PAD_SIZE, PAD_ZH)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pad.data.materials.append(mat_purple)

# Bezier waypoints routed from -y pad -> around SQUID +x side -> +y pad
bias_waypoints = [
    (CHIP_W/2 - 0.55, -3.3, BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.80, -2.20, BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.30, -1.00, BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.30,  0.00, BIAS_Z + BIAS_THK/2),  # hugs SQUID side
    (SQ_CX + SQ_W/2 + 0.30,  1.00, BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.80,  2.20, BIAS_Z + BIAS_THK/2),
    (CHIP_W/2 - 0.55,  3.3, BIAS_Z + BIAS_THK/2),
]

# Build a rectangular BEVEL profile (flat ribbon: width 2*BIAS_HALFW, thickness BIAS_THK)
# Method: create a small rectangular curve as the bevel object for the main bezier.
bev_curve_data = bpy.data.curves.new("bias_bevel", type="CURVE")
bev_curve_data.dimensions = "3D"
sp_b = bev_curve_data.splines.new("POLY")
# rectangle in local XY of the bevel, traced counterclockwise
rect_pts = [( BIAS_HALFW, -BIAS_THK/2),
            ( BIAS_HALFW,  BIAS_THK/2),
            (-BIAS_HALFW,  BIAS_THK/2),
            (-BIAS_HALFW, -BIAS_THK/2),
            ( BIAS_HALFW, -BIAS_THK/2)]
sp_b.points.add(len(rect_pts)-1)
for i, (u, v) in enumerate(rect_pts):
    sp_b.points[i].co = (u, v, 0.0, 1.0)
bev_obj = bpy.data.objects.new("bias_bevel", bev_curve_data)
bpy.context.collection.objects.link(bev_obj)

# Main bezier path
main_curve = bpy.data.curves.new("bias_trace", type="CURVE")
main_curve.dimensions = "3D"
sp_m = main_curve.splines.new("BEZIER")
sp_m.bezier_points.add(len(bias_waypoints)-1)
for i, (x, y, z) in enumerate(bias_waypoints):
    bp = sp_m.bezier_points[i]
    bp.co = (x, y, z)
    bp.handle_left_type  = "AUTO"
    bp.handle_right_type = "AUTO"
main_curve.bevel_mode   = "OBJECT"
main_curve.bevel_object = bev_obj
trace_obj = bpy.data.objects.new("bias_trace", main_curve)
bpy.context.collection.objects.link(trace_obj)
trace_obj.data.materials.append(mat_purple)

# -------------------------------------------------------------------
# External magnetic flux through the SQUID: uniform far-field
# approximation — three parallel straight vertical arrows pointing DOWN
# through the loop interior.  This is the standard textbook symbol for a
# nearly-uniform flux density threading a small loop (valid when the
# bias source is far compared to the loop size).  Straight + parallel
# conveys uniform B without misleading curvature.
plane_z = Z_SQ + SQ_Z/2
interior_x_half = SQ_W/2 - SQ_TH - 0.15
flux_arrow_xs = [SQ_CX - 0.55*interior_x_half,
                 SQ_CX,
                 SQ_CX + 0.55*interior_x_half]
ARROW_TOP_Z   = plane_z + 2.4          # top of shaft (high above loop)
ARROW_TIP_Z   = plane_z + 0.15         # cone tip just above SQUID top
CONE_HEIGHT   = 0.40
SHAFT_RADIUS  = 0.055

for ax_x in flux_arrow_xs:
    shaft_bot = ARROW_TIP_Z + CONE_HEIGHT       # where shaft meets cone base
    shaft_top = ARROW_TOP_Z
    shaft_mid = (shaft_top + shaft_bot) / 2
    shaft_h   = shaft_top - shaft_bot
    bpy.ops.mesh.primitive_cylinder_add(radius=SHAFT_RADIUS,
        depth=shaft_h, location=(ax_x, 0, shaft_mid), vertices=28)
    bpy.context.active_object.data.materials.append(mat_flux)
    # cone tip pointing DOWN (base at shaft bottom, tip at ARROW_TIP_Z)
    bpy.ops.mesh.primitive_cone_add(radius1=SHAFT_RADIUS * 2.8, radius2=0.0,
        depth=CONE_HEIGHT,
        location=(ax_x, 0, ARROW_TIP_Z + CONE_HEIGHT/2),
        rotation=(math.pi, 0, 0), vertices=28)
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
