import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
glb_path = 'outputs/3b47236a-be3e-48fd-8f18-4be94c2ec188.glb'
bpy.ops.import_scene.gltf(filepath=glb_path)

cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

light_data = bpy.data.lights.new(name='Sun', type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new('Sun', light_data)
bpy.context.scene.collection.objects.link(light_obj)
light_obj.location = (2, -2, 4)

bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

cam_obj.location = (0, -2.5, 0)
cam_obj.rotation_euler = (1.5708, 0, 0)
bpy.context.scene.render.filepath = 'temp/render_front.png'
bpy.ops.render.render(write_still=True)

cam_obj.location = (0, 2.5, 0)
cam_obj.rotation_euler = (1.5708, 0, 3.14159)
bpy.context.scene.render.filepath = 'temp/render_back.png'
bpy.ops.render.render(write_still=True)

print('Rendered views successfully!')
