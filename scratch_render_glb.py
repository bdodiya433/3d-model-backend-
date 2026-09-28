import bpy
import mathutils
from pathlib import Path

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='scratch/hunyuan_turtle_test.glb')
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.filepath = str(Path('scratch_test_render_turtle.png').resolve())

objs = [o for o in scene.objects if o.type == 'MESH']
if objs:
    obj = objs[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    obj.location = (0, 0, 0)
    dim = max(obj.dimensions)
    dist = dim * 2.2
else:
    dist = 3.0

cam_data = bpy.data.cameras.new('Cam')
cam = bpy.data.objects.new('Cam', cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = (dist * 0.7, -dist * 0.9, dist * 0.9)
d = mathutils.Vector((0, 0, 0)) - cam.location
cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

light_data = bpy.data.lights.new('Light', 'SUN')
light_data.energy = 4.0
light = bpy.data.objects.new('Light', light_data)
light.location = (2, -2, 5)
scene.collection.objects.link(light)

bpy.ops.render.render(write_still=True)
print("RENDER_DONE")
