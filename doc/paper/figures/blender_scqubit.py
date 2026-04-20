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
scene.render.line_thickness = 2.2
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
    fs.crease_angle = math.radians(60)     # tighter crease angle so the upper
                                            # edges of the cross/arm rectangles
                                            # read as outlined creases.
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
mat_gold     = mat_principled("gold",     (0.72, 0.52, 0.14),
                               metallic=1.0,  roughness=0.25)
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
# Colorful JJ stack (metals stay gold; barrier + marker are DEEP BLUE):
#   - bright electrum-gold electrode tabs (distinct from the dull-gold ring)
#   - deep-navy AlOx barrier
#   - emissive mid-blue marker dot on top
mat_jj_tab     = mat_principled("jj_tab",     (1.00, 0.85, 0.32),
                                 metallic=1.0, roughness=0.18)
mat_jj_barrier = mat_principled("jj_barrier", (0.06, 0.12, 0.45),
                                 metallic=0.0, roughness=0.35)
mat_jj_marker  = mat_principled("jj_marker",  (0.18, 0.36, 0.85),
                                 metallic=0.0, roughness=0.30,
                                 emission=(0.22, 0.42, 0.95),
                                 emission_strength=2.8)

# -------------------------------------------------------------------
# Convention: all primitive_cube_add use size=1 -> edge-length 1, so setting
# scale=(Wx, Wy, Wz) gives FULL edge-lengths (Wx, Wy, Wz) and half-extents
# (Wx/2, Wy/2, Wz/2).  Positions are centers.
#
# Sapphire substrate (trimmed for tighter framing)
CHIP_W, CHIP_D, CHIP_H = 9.5, 5.0, 0.30
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, CHIP_H/2))
chip = bpy.context.active_object
chip.scale = (CHIP_W, CHIP_D, CHIP_H)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bpy.ops.object.modifier_add(type="BEVEL")
chip.modifiers["Bevel"].width = 0.02
chip.modifiers["Bevel"].segments = 3
chip.data.materials.append(mat_sapphire)

# -------------------------------------------------------------------
# Transmon x-mon cross (shifted right to minimize empty space on the -x side)
TRANSMON_CX = -1.5
ARM_LEN = 2.2
ARM_WID = 0.95
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
# No bevel on the cross: sharp 90° creases match the SQUID ring so
# Freestyle produces consistent single-line outlines on all metal.
arm_x.data.materials.append(mat_gold)

# -------------------------------------------------------------------
# SQUID loop (hollow rectangle)
SQ_CX = TRANSMON_CX + ARM_LEN/2 + 1.8   # gap to arm tip = 1.8 - SQ_W/2
SQ_W, SQ_H, SQ_TH = 2.0, 1.4, 0.18
SQ_Z = ARM_THK               # match the cross-arm thickness so the ring
                              # reads as the same metal layer as the rails
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
# We BOOLEAN-UNION each bridge into the SQUID ring with the EXACT solver
# and then weld duplicate vertices, so the bridge/ring edges merge into
# a single continuous polyline (otherwise Freestyle can abruptly end an
# edge at the junction).
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
    bpy.context.view_layer.objects.active = squid
    mod = squid.modifiers.new(f"bridge_{y_sign}", type="BOOLEAN")
    mod.object = b
    mod.operation = "UNION"
    mod.solver = "EXACT"
    mod.use_self = True
    bpy.ops.object.modifier_apply(modifier=f"bridge_{y_sign}")
    bpy.data.objects.remove(b, do_unlink=True)

# Weld duplicate/collinear vertices in the SQUID+bridges composite mesh
bpy.ops.object.select_all(action='DESELECT')
squid.select_set(True)
bpy.context.view_layer.objects.active = squid
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=1e-5)
# Dissolve degenerate geometry + limit-dissolve so collinear edges merge
bpy.ops.mesh.dissolve_degenerate(threshold=1e-5)
bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(2.0))
bpy.ops.object.mode_set(mode='OBJECT')

# Two Josephson junctions, each depicted as a proper 3-element overlap:
#   (a) the SQUID ring is BROKEN at the JJ position — a small rectangular
#       gap cut out of the ring's top/bottom segment.
#   (b) two gold electrodes extend across the gap, each attached to one
#       side of the ring, overlapping at the center.
#   (c) the insulating barrier (dark blue) sits flat at the overlap,
#       visible as a thin stripe separating the two electrode tabs.
# All flat (same z as ring top), no bumps, no cast shadows on the ring.

# JJs intentionally omitted — strip markers will be added in post.
# Leaving the SQUID ring as a clean unbroken rectangle.

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

PAD_Y_FLUX = 2.00    # reduced from 3.3 to fit the smaller 5.0-deep chip
PAD_X_FLUX = CHIP_W/2 - PAD_SIZE/2 - 0.15

# Two square bond pads at the +x chip edge
for y_pad in (-PAD_Y_FLUX, PAD_Y_FLUX):
    bpy.ops.mesh.primitive_cube_add(size=1,
        location=(PAD_X_FLUX, y_pad, BIAS_Z + PAD_ZH/2))
    pad = bpy.context.active_object
    pad.scale = (PAD_SIZE, PAD_SIZE, PAD_ZH)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pad.data.materials.append(mat_purple)

# Bezier waypoints routed from -y pad -> around SQUID +x side -> +y pad
bias_waypoints = [
    (PAD_X_FLUX,              -PAD_Y_FLUX,     BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.75,   -1.35,           BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.30,   -0.70,           BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.30,    0.00,           BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.30,    0.70,           BIAS_Z + BIAS_THK/2),
    (SQ_CX + SQ_W/2 + 0.75,    1.35,           BIAS_Z + BIAS_THK/2),
    (PAD_X_FLUX,               PAD_Y_FLUX,     BIAS_Z + BIAS_THK/2),
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
# External magnetic flux threading the SQUID loop.
# Rather than 3 big arrows in a line, sample the uniform field on an
# 8-arrow HEXAGONAL STAGGERED grid (3 rows of 3-2-3) covering the loop
# interior.  Staggered rows avoid any row-or-column alignment while still
# reading as "many parallel arrows = uniform B field threading the loop".
plane_z         = Z_SQ + SQ_Z/2
interior_x_half = SQ_W/2 - SQ_TH - 0.15
interior_y_half = SQ_H/2 - SQ_TH - 0.10
# Hex pattern (positions normalized to [-1,1] then scaled):
#   row A (bottom):   3 arrows at x ∈ {-0.70, 0.00, +0.70}, y = -0.70
#   row B (middle):   2 arrows at x ∈ {-0.36, +0.36}, y = 0
#   row C (top):      3 arrows at x ∈ {-0.70, 0.00, +0.70}, y = +0.70
hex_pattern = [
    (-0.70, -0.70), ( 0.00, -0.70), ( 0.70, -0.70),
    (-0.36,  0.00),                  ( 0.36,  0.00),
    (-0.70,  0.70), ( 0.00,  0.70), ( 0.70,  0.70),
]
flux_arrow_positions = [
    (SQ_CX + nx * interior_x_half, ny * interior_y_half)
    for (nx, ny) in hex_pattern
]
# Arrows point UPWARDS (tip at top) to indicate positive flux threading
# the loop in the +z direction.  Small and compact so they don't dominate
# the frame.
ARROW_BASE_Z = plane_z + 0.07           # shaft starts just above loop top
ARROW_TOP_Z  = plane_z + 0.80           # overall tip height
CONE_HEIGHT  = 0.17
CONE_RADIUS  = 0.062
SHAFT_RADIUS = 0.022

for (ax_x, ax_y) in flux_arrow_positions:
    shaft_bot = ARROW_BASE_Z
    shaft_top = ARROW_TOP_Z - CONE_HEIGHT     # shaft ends at cone base
    shaft_mid = 0.5 * (shaft_top + shaft_bot)
    shaft_h   = shaft_top - shaft_bot
    bpy.ops.mesh.primitive_cylinder_add(radius=SHAFT_RADIUS,
        depth=shaft_h, location=(ax_x, ax_y, shaft_mid), vertices=18)
    bpy.context.active_object.data.materials.append(mat_flux)
    # Cone with tip pointing UP (tip at ARROW_TOP_Z, base at shaft_top)
    bpy.ops.mesh.primitive_cone_add(radius1=CONE_RADIUS, radius2=0.0,
        depth=CONE_HEIGHT,
        location=(ax_x, ax_y, shaft_top + CONE_HEIGHT/2),
        rotation=(0, 0, 0), vertices=18)
    bpy.context.active_object.data.materials.append(mat_flux)

# -------------------------------------------------------------------
# Readout resonator: green meandering line on the -x arm side (compact
# version to fit the trimmed substrate).
mx_start = TRANSMON_CX - ARM_LEN/2 - 0.30
pts = [(mx_start, 0, Z_METAL + 0.03)]
dx = 0.22
dy = 0.55
n_meanders = 2
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

# Small green bond pad at chip-edge exit of the readout line
READOUT_PAD_X = x - 0.6
READOUT_PAD_Z = Z_METAL + 0.02
bpy.ops.mesh.primitive_cube_add(size=1,
    location=(READOUT_PAD_X, 0, READOUT_PAD_Z))
readout_pad = bpy.context.active_object
readout_pad.scale = (0.28, 0.32, ARM_THK)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
readout_pad.data.materials.append(mat_green)

# -------------------------------------------------------------------
# EXTERNAL CIRCUIT — off-chip coax connectors + bond wires.
# Each on-chip bond pad (2 flux + 1 readout) connects to a short coax
# stub sitting just past the chip edge, via a thin gold arc ("bond wire").
# This signals that the device is driven and read out by external
# instrumentation, rather than appearing as an isolated island.

mat_coax_jacket = mat_principled("coax_jacket", (0.18, 0.19, 0.22),
                                  metallic=0.0, roughness=0.55)
mat_coax_core   = mat_principled("coax_core",   (0.95, 0.78, 0.30),
                                  metallic=1.0, roughness=0.25)
mat_bondwire    = mat_principled("bondwire",    (0.96, 0.78, 0.35),
                                  metallic=1.0, roughness=0.22)

def add_coax_stub(center, length=1.20, jacket_r=0.15, core_r=0.05, axis="x"):
    """Horizontal coax stub: outer dark jacket + exposed gold inner core at
    the chip-facing end.  `axis` = 'x' means length runs along x-axis."""
    cx, cy, cz = center
    # Jacket
    if axis == "x":
        rot = (0, math.radians(90), 0)
    else:
        rot = (math.radians(90), 0, 0)
    bpy.ops.mesh.primitive_cylinder_add(radius=jacket_r, depth=length,
                                         location=(cx, cy, cz),
                                         rotation=rot, vertices=28)
    bpy.context.active_object.data.materials.append(mat_coax_jacket)
    # Inner core sticking out the chip-facing end (shorter, fatter tip)
    if axis == "x":
        core_dir = 1.0 if cx > 0 else -1.0
        core_len = 0.45
        core_loc = (cx - core_dir * (length/2 + core_len/2 - 0.02),
                    cy, cz)
        core_rot = rot
    else:
        core_len = 0.45
        core_loc = (cx, cy, cz)
        core_rot = rot
    bpy.ops.mesh.primitive_cylinder_add(radius=core_r, depth=core_len,
                                         location=core_loc,
                                         rotation=core_rot, vertices=20)
    bpy.context.active_object.data.materials.append(mat_coax_core)
    return core_loc

def add_bondwire(p0, p1, arc_height=0.55, radius=0.022, material=None):
    """Thin gold bezier arc from p0 -> (mid lifted by arc_height) -> p1."""
    midx = 0.5 * (p0[0] + p1[0])
    midy = 0.5 * (p0[1] + p1[1])
    midz = 0.5 * (p0[2] + p1[2]) + arc_height
    curve = bpy.data.curves.new("bond", type="CURVE")
    curve.dimensions = "3D"
    sp = curve.splines.new("BEZIER")
    sp.bezier_points.add(2)
    for i, p in enumerate([p0, (midx, midy, midz), p1]):
        bp = sp.bezier_points[i]
        bp.co = p
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    curve.bevel_depth = radius
    curve.bevel_resolution = 8
    obj = bpy.data.objects.new("bond", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material if material is not None else mat_bondwire)
    return obj

# Flux-bias coax stubs (two of them, sitting at +x past the chip edge)
FLUX_COAX_X = CHIP_W/2 + 0.90
FLUX_COAX_Z = CHIP_H + 0.35
for y_pad in (-PAD_Y_FLUX, PAD_Y_FLUX):
    core_end = add_coax_stub(
        center=(FLUX_COAX_X, y_pad, FLUX_COAX_Z),
        length=1.10, jacket_r=0.16, core_r=0.045, axis="x")
    # core_end returned for reference but we use the chip-facing tip
    pad_top = (PAD_X_FLUX, y_pad, BIAS_Z + PAD_ZH + 0.005)
    coax_tip = (FLUX_COAX_X - 0.55 - 0.22, y_pad, FLUX_COAX_Z)
    add_bondwire(pad_top, coax_tip, arc_height=0.55, radius=0.026)

# Readout coax stub (one, at -x past the chip edge)
READOUT_COAX_X = -CHIP_W/2 - 0.90
READOUT_COAX_Z = CHIP_H + 0.35
core_end = add_coax_stub(
    center=(READOUT_COAX_X, 0, READOUT_COAX_Z),
    length=1.10, jacket_r=0.16, core_r=0.045, axis="x")
readout_tip = (READOUT_PAD_X + 0.05, 0, READOUT_PAD_Z + 0.05)
readout_coax_tip = (READOUT_COAX_X + 0.55 + 0.22, 0, READOUT_COAX_Z)
add_bondwire(readout_tip, readout_coax_tip, arc_height=0.50, radius=0.026)

# -------------------------------------------------------------------
# Camera — framed on the full device incl. external coax stubs
bpy.ops.object.camera_add(location=(0.0, -9.2, 5.0),
                          rotation=(math.radians(60), 0, math.radians(-3)))
cam = bpy.context.active_object
cam.data.lens = 55
cam.data.dof.use_dof = False
scene.camera = cam

# Target the midpoint between the transmon cross and the SQUID loop, at
# approximately the chip surface height.
TARGET_X = 0.5 * (TRANSMON_CX + SQ_CX)
bpy.ops.object.empty_add(type="PLAIN_AXES",
                         location=(TARGET_X, 0, CHIP_H + 0.30))
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
