"""blender_bloch.py — Bloch sphere for the Ramsey measurement inset."""
import bpy
import math
import os
import sys

OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "blender_bloch.png")
if "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]
    if args: OUTFILE = args[0]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.render.resolution_x = 1200
scene.render.resolution_y = 1200
scene.render.resolution_percentage = 100
scene.render.filepath = OUTFILE
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = True
scene.view_settings.view_transform = "Filmic"
scene.view_settings.look = "Medium High Contrast"

prefs = bpy.context.preferences
cprefs = prefs.addons["cycles"].preferences
cprefs.compute_device_type = "OPTIX"
for dev in cprefs.devices: dev.use = True

# transparent world
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
nt = world.node_tree
for n in list(nt.nodes): nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputWorld")
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.97, 0.93, 0.85, 1.0)
bg.inputs["Strength"].default_value = 0.40
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def mat(name, color, metallic=0, roughness=0.4, transmission=0, alpha=1,
        emission=None, estrength=0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    try: bsdf.inputs["Transmission Weight"].default_value = transmission
    except Exception: pass
    if emission:
        try:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1)
            bsdf.inputs["Emission Strength"].default_value = estrength
        except Exception: pass
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m

mat_sphere = mat("sph", (0.85, 0.90, 1.0), metallic=0, roughness=0.10,
                 transmission=0.92, alpha=0.32)
mat_axis   = mat("axis", (0.1, 0.12, 0.18), metallic=0.2, roughness=0.5)
mat_red    = mat("red", (0.88, 0.22, 0.18), metallic=0.2, roughness=0.3,
                 emission=(0.9, 0.3, 0.25), estrength=3)
mat_blue   = mat("blue", (0.22, 0.40, 0.85), metallic=0.3, roughness=0.25,
                 emission=(0.3, 0.45, 0.95), estrength=3)
mat_arc    = mat("arc", (0.88, 0.22, 0.18), metallic=0.2, roughness=0.25,
                 emission=(0.9, 0.3, 0.25), estrength=4)

# Sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0,0,0),
                                      segments=96, ring_count=48)
sph = bpy.context.active_object
bpy.ops.object.shade_smooth()
sph.data.materials.append(mat_sphere)

# Equator ring (thin torus)
bpy.ops.mesh.primitive_torus_add(major_radius=1.001, minor_radius=0.008,
                                  location=(0,0,0))
torus = bpy.context.active_object
torus.data.materials.append(mat_axis)

# Meridian (x-z great circle)
bpy.ops.mesh.primitive_torus_add(major_radius=1.001, minor_radius=0.008,
                                  location=(0,0,0), rotation=(math.pi/2, 0, 0))
torus2 = bpy.context.active_object
torus2.data.materials.append(mat_axis)

# Axes: x, y, z
AXIS_L = 1.40
for (axis, color_mat) in [
    ((AXIS_L, 0, 0), mat_axis),
    ((0, AXIS_L, 0), mat_axis),
    ((0, 0, AXIS_L), mat_axis),
]:
    bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=AXIS_L,
        location=(axis[0]/2, axis[1]/2, axis[2]/2),
        rotation=(math.pi/2 if axis[1] > 0 else 0,
                  math.pi/2 if axis[0] > 0 else 0, 0),
        vertices=24)
    bpy.context.active_object.data.materials.append(color_mat)

# arrowheads
for (pos, rot) in [
    ((AXIS_L, 0, 0), (0, math.pi/2, 0)),
    ((0, AXIS_L, 0), (-math.pi/2, 0, 0)),
    ((0, 0, AXIS_L), (0, 0, 0)),
]:
    bpy.ops.mesh.primitive_cone_add(radius1=0.05, radius2=0, depth=0.12,
        location=pos, rotation=rot, vertices=24)
    bpy.context.active_object.data.materials.append(mat_axis)

# Initial state (|0>+|1>)/sqrt(2) — on +x axis (equator)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09,
    location=(1, 0, 0), segments=24, ring_count=12)
bpy.context.active_object.data.materials.append(mat_blue)

# Final state (after accumulated phase phi, equator rotation)
phi = 1.8  # some representative phase
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09,
    location=(math.cos(phi), math.sin(phi), 0), segments=24, ring_count=12)
bpy.context.active_object.data.materials.append(mat_red)

# Phase-accumulation red arc on the equator (from 0 to phi)
from bpy import context
phi_steps = 80
phi_vals = [phi * t/(phi_steps-1) for t in range(phi_steps)]
pts = [(math.cos(p)*1.01, math.sin(p)*1.01, 0.0) for p in phi_vals]

curve_data = bpy.data.curves.new("arc", type="CURVE")
curve_data.dimensions = "3D"
sp = curve_data.splines.new("POLY")
sp.points.add(len(pts)-1)
for i, (px, py, pz) in enumerate(pts):
    sp.points[i].co = (px, py, pz, 1.0)
curve_data.bevel_depth = 0.045
curve_data.bevel_resolution = 10
obj = bpy.data.objects.new("arc", curve_data)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(mat_arc)

# camera — view so the +x axis and phase-accumulation arc on the equator are in front
bpy.ops.object.camera_add(location=(3.1, -2.6, 2.2),
                          rotation=(math.radians(62), 0, math.radians(50)))
cam = bpy.context.active_object
cam.data.lens = 70
scene.camera = cam
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
target = bpy.context.active_object
tc = cam.constraints.new(type="TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"

# lights
bpy.ops.object.light_add(type="AREA", location=(3, -3, 4))
key = bpy.context.active_object
key.data.energy = 1200; key.data.size = 5
key.data.color = (1.0, 0.96, 0.88)

bpy.ops.object.light_add(type="AREA", location=(-4, -2, 4))
fill = bpy.context.active_object
fill.data.energy = 600; fill.data.size = 5
fill.data.color = (0.85, 0.90, 1.0)

bpy.ops.render.render(write_still=True)
print(f"Saved: {OUTFILE}")
