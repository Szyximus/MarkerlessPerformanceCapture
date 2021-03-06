import csv
import bpy
import bmesh
import math

#--------------------- PARAMETERS--------------------
bpy.context.scene.frame_set(0)
counter = 0
frame_interval = 3
file_path = 'D:/3D/Mermaid/output.csv'

neck_bone_power= 0.3
head_bone_power= 0.7

pitch_adjustment = 2
neutral_NER = 0.15
min_NER = 0.10
max_NER = 0.23

neutral_EAR = 0.29
min_EAR = 0.13
max_EAR = 0.45

eyeY_adjustment = 0.1

eyeY_power = 3
eyeX_power = 8

eye_look_threshold = 0.19

neutral_OMAR = 0.30
min_OMAR = 0.2
max_OMAR = 0.65

min_IMAR = 0.2
max_IMAR = 0.6

neutral_N2MAR = 1.1
min_N2MAR = 0.9
max_N2MAR = 1.3

EAR_blink_threshold = 0.19

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

eye_R = object.pose.bones['Eye_R']
eye_L = object.pose.bones['Eye_L']

nose_R = object.pose.bones['Nose_R']
nose_L = object.pose.bones['Nose_L']

eyelid_bottom_L = object.pose.bones['EyeLid_Bottom_L']
eyelid_top_L = object.pose.bones['EyeLid_Top_L']
eyelid_bottom_R = object.pose.bones['EyeLid_Bottom_R']
eyelid_top_R = object.pose.bones['EyeLid_Top_R']
cheek_R = object.pose.bones['Cheek_R']
cheek_L = object.pose.bones['Cheek_L']

jaw = object.pose.bones['Jaw']

lip_top_R = object.pose.bones['LipTop_R']
lip_top_L = object.pose.bones['LipTop_L']
lip_top = object.pose.bones['LipTop']

lip_bottom_R = object.pose.bones['LipBottom_R']
lip_bottom_L = object.pose.bones['LipBottom_L']
lip_bottom = object.pose.bones['LipBottom']
chin = object.pose.bones['Chin']

lip_corner_L = object.pose.bones['LipCorner_L']
lip_corner_R = object.pose.bones['LipCorner_R']




if (neck_bone is  None or head_bone is None or outer_eyebrow_R is None or outer_eyebrow_L is None or inner_eyebrow_R is None 
    or inner_eyebrow_L is None or eye_R is None or eye_L is None):
        
    print("Some Bones Don't Exist")
else:     
    with open(file_path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            if row:
                for ob in bpy.data.objects["Armature"].data.bones:
                    ob.select = False
                    
                #bpy.ops.pose.select_all(action='SELECT')
                #bpy.ops.pose.transforms_clear()
                
#-------------------------------------------------------------------------------------------------------------------------------
# Head Tracking
                neck_bone.rotation_mode = 'XYZ'
                head_bone.rotation_mode = 'XYZ'
                angle = float(row[0]) * 1.2
                
                neck_bone.rotation_euler.rotate_axis('Y', math.radians(angle) * neck_bone_power)
                head_bone.rotation_euler.rotate_axis('Y', math.radians(angle) * head_bone_power)
                
                angle = float(row[1]) * 1
                
                neck_bone.rotation_euler.rotate_axis('X', math.radians(angle + pitch_adjustment) * neck_bone_power)
                head_bone.rotation_euler.rotate_axis('X', math.radians(angle + pitch_adjustment) * head_bone_power)
                
                angle = float(row[2]) * 0.8
                
                neck_bone.rotation_euler.rotate_axis('Z', math.radians(angle) * neck_bone_power)
                head_bone.rotation_euler.rotate_axis('Z', math.radians(angle) * head_bone_power)
                
                neck_bone.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                head_bone.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
#--------------------------------------------------------------------------------------------------------------------------------------
# Eye Tracking
                eye_L.rotation_mode = 'XYZ'
                eye_R.rotation_mode = 'XYZ'
                
                angleX = (float(row[7]) - 0.5) * 3
                angleY = (float(row[8]) - 0.5 + eyeY_adjustment) * 3
                blink = float(row[11])
                
                if blink <= EAR_blink_threshold:
                    angleX = 0
                    angleY = 0
                
                
                angleX = (abs(angleX) - eye_look_threshold) * eyeX_power 
                
                angleY = (abs(angleY) - eye_look_threshold) * eyeY_power
                if angleX < 0:
                   angleX = 0
                   
                if angleY < 0:
                   angleY = 0
                   
                angleX = angleX * math.copysign(1, (float(row[7]) - 0.5))
                angleY = angleY * math.copysign(1, (float(row[8]) - 0.5 + eyeY_adjustment) )             
                
                eye_L.rotation_euler.rotate_axis('Z', angleX)
                eye_R.rotation_euler.rotate_axis('Z', angleX)
                
                eye_L.rotation_euler.rotate_axis('X', -angleY)
                eye_R.rotation_euler.rotate_axis('X', -angleY)
                
                eye_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                eye_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)

#--------------------------------------------------------------------------------------------------------------------------------------
# Eyebrows
         
                inner_eyebrow_R.rotation_mode = 'XYZ'
                inner_eyebrow_L.rotation_mode = 'XYZ'
                outer_eyebrow_R.rotation_mode = 'XYZ'
                outer_eyebrow_L.rotation_mode = 'XYZ'
                NER_L = (float(row[3])*3 + float(row[4])) / 4
                NER_R = (float(row[4])*3 + float(row[3])) / 4
                
                normalisation_value = 1- ((abs(float(row[0])) + abs(float(row[1]))) / 20)
                
                if normalisation_value < 0:
                    normalisation_value = 0
                
                if (NER_L - neutral_NER) >= 0:
                    inner_eyebrow_L.rotation_euler.rotate_axis('X', (0.25 * ((NER_L - neutral_NER) / (max_NER - neutral_NER)) * normalisation_value ))
                    outer_eyebrow_L.rotation_euler.rotate_axis('X', (0.08 * ((NER_L - neutral_NER) / (max_NER - neutral_NER))  * normalisation_value ))
                else:
                    inner_eyebrow_L.rotation_euler.rotate_axis('Z', (-0.33 * ((NER_L - neutral_NER) / (min_NER - neutral_NER)) * normalisation_value ))
                    outer_eyebrow_L.rotation_euler.rotate_axis('X', (0.06 * ((NER_L - neutral_NER) / (max_NER - neutral_NER)) * normalisation_value ))
                    
                if (NER_R - neutral_NER) >= 0:
                    inner_eyebrow_R.rotation_euler.rotate_axis('X', (0.25 * ((NER_R - neutral_NER) / (max_NER - neutral_NER)) * normalisation_value ))
                    outer_eyebrow_R.rotation_euler.rotate_axis('X', (0.08 * ((NER_R - neutral_NER) / (max_NER - neutral_NER)) * normalisation_value ))
                else:
                    inner_eyebrow_R.rotation_euler.rotate_axis('Z', (0.33 * ((NER_R - neutral_NER) / (min_NER - neutral_NER)) * normalisation_value ))
                    outer_eyebrow_R.rotation_euler.rotate_axis('X', (0.06 * ((NER_R - neutral_NER) / (max_NER - neutral_NER)) * normalisation_value ))
                    
                inner_eyebrow_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                inner_eyebrow_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                outer_eyebrow_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                outer_eyebrow_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                

#--------------------------------------------------------------------------------------------------------------------------------------
# Eye Squinting
                
                eyelid_bottom_L.rotation_mode = 'XYZ'
                eyelid_top_L.rotation_mode = 'XYZ'
                eyelid_bottom_R.rotation_mode = 'XYZ'
                eyelid_top_R.rotation_mode = 'XYZ'
                cheek_L.rotation_mode = 'XYZ'
                cheek_R.rotation_mode = 'XYZ'
                                
                EAR_L = float(row[5])
                EAR_R = float(row[6])
                
                if (EAR_L - neutral_EAR) >= 0:
                    eyelid_top_L.rotation_euler.rotate_axis('X', (0.5 * ((EAR_L - neutral_EAR) / (max_EAR - neutral_EAR)) * normalisation_value ))
                    eyelid_bottom_L.rotation_euler.rotate_axis('X', 0)
                    cheek_L.rotation_euler.rotate_axis('X', 0)
                else:
                    eyelid_top_L.rotation_euler.rotate_axis('X', (-0.33 * ((EAR_L - neutral_EAR) / (min_EAR - neutral_EAR)) * normalisation_value ))
                    eyelid_bottom_L.rotation_euler.rotate_axis('X', (0.25 * ((EAR_L - neutral_EAR) / (min_EAR - neutral_EAR)) * normalisation_value ))
                    cheek_L.rotation_euler.rotate_axis('X', (0.05 * ((EAR_L - neutral_EAR) / (min_EAR - neutral_EAR)) * normalisation_value ))
                    
                if (EAR_R - neutral_EAR) >= 0:
                    eyelid_top_R.rotation_euler.rotate_axis('X', (0.5 * ((EAR_R - neutral_EAR) / (max_EAR - neutral_EAR)) * normalisation_value ))
                    eyelid_bottom_R.rotation_euler.rotate_axis('X', 0)
                    cheek_R.rotation_euler.rotate_axis('X', 0)
                else:
                    eyelid_top_R.rotation_euler.rotate_axis('X', (-0.33 * ((EAR_R - neutral_EAR) / (min_EAR - neutral_EAR)) * normalisation_value ))
                    eyelid_bottom_R.rotation_euler.rotate_axis('X', (0.25 * ((EAR_R - neutral_EAR) / (min_EAR - neutral_EAR)) * normalisation_value ))
                    cheek_R.rotation_euler.rotate_axis('X', (0.05 * ((EAR_R - neutral_EAR) / (min_EAR - neutral_EAR)) * normalisation_value ))
                    
                
                eyelid_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                eyelid_bottom_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                cheek_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                eyelid_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                eyelid_bottom_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                cheek_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                
#--------------------------------------------------------------------------------------------------------------------------------------
# Mouth opening & teeth showing

                IMAR = float(row[10])
                
                jaw.rotation_mode = 'XYZ'
                
                lip_top_R.rotation_mode = 'XYZ'
                lip_top_L.rotation_mode = 'XYZ'
                lip_top.rotation_mode = 'XYZ'
                
                lip_bottom_R.rotation_mode = 'XYZ'
                lip_bottom_L.rotation_mode = 'XYZ'
                lip_bottom.rotation_mode = 'XYZ'
                
                lip_corner_L.rotation_mode = 'XYZ'
                lip_corner_R.rotation_mode = 'XYZ'
                    
                jaw.rotation_euler.rotate_axis('X', (math.radians(5) - math.radians(5) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                
                lip_top.rotation_euler.rotate_axis('X', ( math.radians(4) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                lip_top_R.rotation_euler.rotate_axis('X', ( math.radians(20) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                lip_top_L.rotation_euler.rotate_axis('X', ( math.radians(20) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                
                lip_bottom.rotation_euler.rotate_axis('X', ( -math.radians(6) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                lip_bottom_R.rotation_euler.rotate_axis('X', ( -math.radians(15) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                lip_bottom_L.rotation_euler.rotate_axis('X', ( -math.radians(15) * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value ))
                
                
                
                jaw.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                lip_top.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)

                lip_bottom.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_bottom_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_bottom_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
#--------------------------------------------------------------------------------------------------------------------------------------
# Mouth Squash and Smile

                OMAR = float(row[9])
                
                lip_corner_R.rotation_mode = 'XYZ'
                lip_corner_L.rotation_mode = 'XYZ'
                
                if (OMAR - neutral_OMAR) <= 0:
                    lip_corner_R.rotation_euler.rotate_axis('X', (-0.3 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value +
                     0.25 * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value))
                     
                    lip_corner_L.rotation_euler.rotate_axis('X', (-0.3 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value +
                     0.25 * ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value))
                     
                    lip_corner_R.rotation_euler.rotate_axis('Z', (-0.01 * ((OMAR - neutral_OMAR) / (min_OMAR - neutral_OMAR)) * normalisation_value ))
                    lip_corner_L.rotation_euler.rotate_axis('Z', ( 0.01 * ((OMAR - neutral_OMAR) / (min_OMAR - neutral_OMAR)) * normalisation_value ))
                    
                    lip_top_R.rotation_euler.rotate_axis('X', ( - 0.025 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ) )
                    lip_top_L.rotation_euler.rotate_axis('X', ( - 0.025 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ) )
                    
                    lip_bottom.rotation_euler.rotate_axis('X', (- 0.025 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    
                    cheek_L.rotation_euler.rotate_axis('X', (- 0.1 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    cheek_R.rotation_euler.rotate_axis('X', (-0.1 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    
                    eyelid_bottom_L.rotation_euler.rotate_axis('X', (eyelid_bottom_L.matrix.to_euler().x - 0.05 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    eyelid_bottom_R.rotation_euler.rotate_axis('X', (eyelid_bottom_R.matrix.to_euler().x - 0.05 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    
                    eyelid_top_L.rotation_euler.rotate_axis('X', (eyelid_top_L.matrix.to_euler().x - 0.1 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    eyelid_top_R.rotation_euler.rotate_axis('X', (eyelid_top_R.matrix.to_euler().x - 0.1 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    
                    jaw.rotation_euler.rotate_axis('X', (-0.15 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value))
                    
                else:
                    lip_corner_R.rotation_euler.rotate_axis('Z', (0.25 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    lip_corner_L.rotation_euler.rotate_axis('Z', (-0.25 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    lip_top.rotation_euler.rotate_axis('X', (0.3 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    lip_bottom.rotation_euler.rotate_axis('X', (-0.3 * ((OMAR - neutral_OMAR) / (max_OMAR - neutral_OMAR)) * normalisation_value ))
                    
                    
                lip_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                lip_corner_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_corner_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                cheek_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                cheek_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                eyelid_bottom_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                eyelid_bottom_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                eyelid_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                eyelid_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                lip_bottom.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                jaw.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                lip_top.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_bottom.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)

#--------------------------------------------------------------------------------------------------------------------------------------
# Mouth Smile & Frown
                
                N2MAR_R = float(row[12])
                N2MAR_L = float(row[13])
                
                chin.rotation_mode = 'XYZ'
                
                nose_R.rotation_mode = 'XYZ'
                nose_L.rotation_mode = 'XYZ'
                
                if (N2MAR_R - neutral_N2MAR) >= 0:
                    
                    lip_corner_R.rotation_euler.rotate_axis('Z', 0)
                    
                    lip_corner_R.rotation_euler.rotate_axis('X', ( 0.25 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value * (OMAR - max_OMAR)/ (max_OMAR - min_OMAR)) +
                     0.35* ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value)
                    
                    chin.rotation_euler.rotate_axis('X', (-0.1 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value * ((OMAR - max_OMAR)/ (max_OMAR - min_OMAR))) -
                     0.12* ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value)
                    
                    nose_R.rotation_euler.rotate_axis('X', (-0.3 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value * ((OMAR - max_OMAR)/ (max_OMAR - min_OMAR))) -
                     0.35* ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value)
                else:
                    
                    lip_corner_R.rotation_euler.rotate_axis('X', (0.15 * ((N2MAR_R - neutral_N2MAR) / (min_N2MAR - neutral_N2MAR)) * normalisation_value ))
                
                    lip_top_R.rotation_euler.rotate_axis('X', ( - 0.025 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ) )
                
                    lip_bottom.rotation_euler.rotate_axis('X', (- 0.02 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    cheek_R.rotation_euler.rotate_axis('X', (- 0.07 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    eyelid_bottom_R.rotation_euler.rotate_axis('X', (- 0.05 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    lip_corner_R.rotation_euler.rotate_axis('Z', (0.02 * ((N2MAR_R - neutral_N2MAR) / (min_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    inner_eyebrow_R.rotation_euler.rotate_axis('X', (0.1 * ((N2MAR_R - neutral_N2MAR) / (min_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    chin.rotation_euler.rotate_axis('Z', 0)
                    
                    nose_R.rotation_euler.rotate_axis('X', 0)

                    
                if (N2MAR_L - neutral_N2MAR) >= 0:
                    
                    lip_corner_L.rotation_euler.rotate_axis('Z', 0)
                    
                    lip_corner_L.rotation_euler.rotate_axis('X', (0.25 * ((N2MAR_L - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value * (OMAR - max_OMAR)/ (max_OMAR - min_OMAR))+
                     0.35* ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value)
                
                    chin.rotation_euler.rotate_axis('X', (-0.1 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value * (OMAR - max_OMAR)/ (max_OMAR - min_OMAR))-
                     0.12* ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value)
                
                    nose_L.rotation_euler.rotate_axis('X', (-0.3 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value * ((OMAR - max_OMAR)/ (max_OMAR - min_OMAR)))-
                     0.35* ((IMAR - min_IMAR) / (max_IMAR - min_IMAR)) * normalisation_value)
                else:
                    
                    lip_corner_L.rotation_euler.rotate_axis('X', (0.15 * ((N2MAR_L - neutral_N2MAR) / (min_N2MAR - neutral_N2MAR)) * normalisation_value ))
                
                    lip_top_L.rotation_euler.rotate_axis('X', (- 0.025 * ((N2MAR_L - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ) )
                
                    lip_bottom.rotation_euler.rotate_axis('X', (- 0.02 * ((N2MAR_L - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    cheek_L.rotation_euler.rotate_axis('X', (-0.07 * ((N2MAR_L - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    eyelid_bottom_L.rotation_euler.rotate_axis('X', (-0.05 * ((N2MAR_R - neutral_N2MAR) / (max_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    lip_corner_L.rotation_euler.rotate_axis('Z', (-0.02 * ((N2MAR_R - neutral_N2MAR) / (min_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    inner_eyebrow_L.rotation_euler.rotate_axis('X', ( 0.1 * ((N2MAR_L - neutral_N2MAR) / (min_N2MAR - neutral_N2MAR)) * normalisation_value ))
                    
                    chin.rotation_euler.rotate_axis('Z', 0)
                    
                    nose_L.rotation_euler.rotate_axis('X', 0)
                    
                lip_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                lip_corner_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                lip_corner_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                cheek_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                cheek_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                eyelid_bottom_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                eyelid_bottom_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                inner_eyebrow_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                inner_eyebrow_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)  
                
                lip_bottom.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)       
                
                chin.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)   
                
                nose_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                nose_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)   
#-----------------------------------------------------------------------------------------------------------------------------------
                
                counter += 1
                bpy.context.scene.frame_set(counter * frame_interval)   
                
                bpy.ops.pose.select_all(action='SELECT')
                bpy.ops.pose.transforms_clear()
                bpy.ops.pose.select_all(action='DESELECT')
                
#--------------------------------------------------------------------------------------------------------------------------------------
# Blinking
    
    counter = 0
    bpy.context.scene.frame_set(counter * frame_interval)   
    
    with open(file_path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            if row:
                for ob in bpy.data.objects["Armature"].data.bones:
                    ob.select = False
                
                blink = float(row[11])
                
                if blink <= EAR_blink_threshold:
                    eyelid_top_R.rotation_euler.rotate_axis('X', -0.65)
                    eyelid_top_L.rotation_euler.rotate_axis('X', -0.65)
                    
                
                    eyelid_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                    eyelid_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                counter += 1
                bpy.context.scene.frame_set(counter * frame_interval)   
    
    bpy.ops.pose.select_all(action='SELECT')
    bpy.context.area.type = 'GRAPH_EDITOR'
      
    bpy.ops.graph.select_all(action='SELECT')
    bpy.ops.graph.smooth()
    
    #--------------------------------------------------------------------------------------------------------------------------------------
# Blinking After filtering
    
    counter = 0
    bpy.context.scene.frame_set(counter * frame_interval)   
    
    with open(file_path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            if row:
                for ob in bpy.data.objects["Armature"].data.bones:
                    ob.select = False
                
                blink = float(row[11])
                
                if blink <= EAR_blink_threshold:
                    eyelid_top_R.rotation_euler.rotate_axis('X', -0.65)
                    eyelid_top_L.rotation_euler.rotate_axis('X', -0.65)
                    
                
                    eyelid_top_L.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                    eyelid_top_R.keyframe_insert(data_path="rotation_euler" ,frame= counter * frame_interval)
                
                counter += 1
                bpy.context.scene.frame_set(counter * frame_interval)   
    
    
    bpy.context.area.type = old_type

#-----------------------------------------------------------------------------------------------------------------------------------