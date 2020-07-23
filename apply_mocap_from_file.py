import csv
import bpy
import bmesh
import math

#--------------------- PARAMETERS--------------------
bpy.context.scene.frame_set(0)
counter = 0
frame_interval = 10
file_path = 'D:/3D/Mermaid/output.csv'

neck_bone_power= 0.3
head_bone_power= 0.7

neutral_NER = 0.22
min_NER = 0.16
max_NER = 0.25

#----------------------------------------------------

bpy.context.scene.transform_orientation_slots[0].type = 'LOCAL'
object = bpy.data.objects['Armature']
bpy.context.view_layer.objects.active = object

old_type = bpy.context.area.type

bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.select_all(action='SELECT')
bpy.ops.pose.transforms_clear()
bpy.ops.pose.select_all(action='DESELECT')

neck_bone = object.pose.bones['Neck']
head_bone = object.pose.bones['Head']
outer_eyebrow_R = object.pose.bones['Eyebrow_Outer_R']
outer_eyebrow_L = object.pose.bones['Eyebrow_Outer_L']
inner_eyebrow_R = object.pose.bones['Eyebrow_Inner_R']
inner_eyebrow_L = object.pose.bones['Eyebrow_Inner_L']



if (neck_bone is  None or head_bone is None):
    print("Some Bones Don't Exist")
else:
    max_NER -= neutral_NER
    min_NER -= neutral_NER
     
    with open(file_path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            if row:
                for ob in bpy.data.objects["Armature"].data.bones:
                    ob.select = False
                
#-------------------------------------------------------------------------------------------------------------------------------
# Head Tracking
                neck_bone.rotation_mode = 'XYZ'
                head_bone.rotation_mode = 'XYZ'
                angle = float(row[0]) * 1.2
                
                neck_bone.rotation_euler.rotate_axis('Y', math.radians(angle) * neck_bone_power)
                head_bone.rotation_euler.rotate_axis('Y', math.radians(angle) * head_bone_power)
                
                angle = float(row[1]) * 1.2
                
                neck_bone.rotation_euler.rotate_axis('X', math.radians(angle) * neck_bone_power)
                head_bone.rotation_euler.rotate_axis('X', math.radians(angle) * head_bone_power)
                
                angle = float(row[2]) * 0.8
                
                neck_bone.rotation_euler.rotate_axis('Z', math.radians(angle) * neck_bone_power)
                head_bone.rotation_euler.rotate_axis('Z', math.radians(angle) * head_bone_power)
                
                neck_bone.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                head_bone.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
#--------------------------------------------------------------------------------------------------------------------------------------
# Eyebrows
         
                inner_eyebrow_R.rotation_mode = 'XYZ'
                inner_eyebrow_L.rotation_mode = 'XYZ'
                outer_eyebrow_R.rotation_mode = 'XYZ'
                outer_eyebrow_L.rotation_mode = 'XYZ'
                NER_L = float(row[3]) - neutral_NER
                NER_R = float(row[4]) - neutral_NER
                
                if NER_L >= 0:
                    inner_eyebrow_L.rotation_euler.rotate_axis('X', 0.25 * NER_L / max_NER)
                    outer_eyebrow_L.rotation_euler.rotate_axis('X', 0.20 * NER_L / max_NER)
                else:
                    inner_eyebrow_L.rotation_euler.rotate_axis('Z', -0.33 * NER_L / min_NER)
                    outer_eyebrow_L.rotation_euler.rotate_axis('X', 0.1 * NER_L / max_NER)
                    
                if NER_R >= 0:
                    inner_eyebrow_R.rotation_euler.rotate_axis('X', 0.25 * NER_R / max_NER)
                    outer_eyebrow_R.rotation_euler.rotate_axis('X', 0.20 * NER_R / max_NER)
                else:
                    inner_eyebrow_R.rotation_euler.rotate_axis('Z', 0.33 * NER_R / min_NER)
                    outer_eyebrow_R.rotation_euler.rotate_axis('X', 0.1 * NER_R / max_NER)
                    
                inner_eyebrow_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                inner_eyebrow_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                outer_eyebrow_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                outer_eyebrow_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
#-----------------------------------------------------------------------------------------------------------------------------------
                
                counter += 1
                bpy.context.scene.frame_set(counter * frame_interval)   
                
                bpy.ops.pose.select_all(action='SELECT')
                bpy.ops.pose.transforms_clear()
                bpy.ops.pose.select_all(action='DESELECT')
                
    
    bpy.ops.pose.select_all(action='SELECT')
    bpy.context.area.type = 'GRAPH_EDITOR'
    
    bpy.ops.graph.select_all(action='SELECT')
    bpy.ops.graph.smooth()
    bpy.ops.graph.smooth()
    
    bpy.context.area.type = old_type
    



 