#Import required modules
import cv2
import dlib
import numpy as np
import os
import csv
import statistics
#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
#video_capture = cv2.VideoCapture('SourceShort4.mp4')
video_capture = cv2.VideoCapture('Source3.mp4')
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("Dataset/shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

#---------------------------------------------------
#Parameters:

frame_interval = 10
pitch_normalisation_value = 0.002
yaw_normalisation_value = 0.002
pupil_threshold = 50
EAR_blink_threshold = 0.15

#---------------------------------------------------

# POSE ESTIMATION:
# 3D model points for head tracking.
model_points = np.array([
    (0.0, 0.0, 0),  # Nose tip
    (0.0, -210.0, -100.0),  # Chin
    (-130.0, 90.0, -110.0),  # Left eye left corner
    (0, 90.0, -70.0),  # Nose Bridge
    (130.0, 90.0, -110.0),  # Right eye right corner
    (-180.0, -100.0, -280.0),  # Left Jaw
    (180.0, -100.0, -280.0)  # Right Jaw

])

# Camera internals for head tracking:

focal_length = video_capture.get(3)
center = (video_capture.get(3) / 2, video_capture.get(4))
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

pitch =[]
yaw = []
roll = []
NER_L = []
NER_R = []
EAR_L = []
EAR_R = []
EX = []
EY = []
IMAR = []
OMAR = []
N2MAR_L = []
N2MAR_R = []

head_output_buffer = [pitch,yaw,roll, NER_L, NER_R, EAR_L, EAR_R, EX, EY, IMAR, OMAR]
final_output = []



while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Histogram Equalisation
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1) #Detect the faces in the image


        if (detections):
            shape = predictor(clahe_image, detections[0]) #Get coordinates

        for i in range(0,68): #There are 69 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0,0,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame

        for i in range(0,16):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on Jaw

        for i in range(17,21):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on Eyebrow

        for i in range(22,26):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on Eyebrow

        for i in range(27,35):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on Nose

        cv2.line(frame, (shape.part(35).x, shape.part(35).y), (shape.part(30).x, shape.part(30).y), (0, 0, 0), thickness=1, lineType=8)  #Line on Nose
        cv2.line(frame, (shape.part(27).x, shape.part(27).y), (shape.part(31).x, shape.part(31).y), (0, 0, 0), thickness=1, lineType=8)  # Last line on Nose
        cv2.line(frame, (shape.part(27).x, shape.part(27).y), (shape.part(35).x, shape.part(35).y), (0, 0, 0), thickness=1, lineType=8)  # Last line on Nose

        for i in range(48,59):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on Mouth
        cv2.line(frame, (shape.part(59).x, shape.part(59).y), (shape.part(48).x, shape.part(48).y), (0, 0, 0), thickness=1, lineType=8)  # Line on Mouth

        for i in range(60,67):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on inner Mouth
        cv2.line(frame, (shape.part(67).x, shape.part(67).y), (shape.part(60).x, shape.part(60).y), (0, 0, 0), thickness=1, lineType=8)  # Line on inner Mouth

        for i in range(36,41):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on eye
        cv2.line(frame, (shape.part(41).x, shape.part(41).y), (shape.part(36).x, shape.part(36).y), (0, 0, 0), thickness=1, lineType=8)  # Line on eye

        for i in range(42,47):
            cv2.line(frame, (shape.part(i).x, shape.part(i).y), (shape.part(i+1).x, shape.part(i+1).y), (0,0,0),thickness=1,lineType = 8) #Line on eye
        cv2.line(frame, (shape.part(47).x, shape.part(47).y), (shape.part(42).x, shape.part(42).y), (0, 0, 0), thickness=1, lineType=8)  # Line on eye


#------------------------------------------------------------------------------------
        #Head Tracking:

        # 2D points for Head Tracking
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),  # Chin
            (shape.part(45).x, shape.part(45).y),  # Left eye left corner
            (shape.part(27).x, shape.part(27).y),  # Nose Bridge
            (shape.part(36).x, shape.part(36).y),  # Right eye right corner
            (shape.part(12).x, shape.part(12).y),  # Left Jaw
            (shape.part(4).x, shape.part(4).y)  # Right Jaw
        ], dtype="double")


        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)


        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)

        nose_tip = (int(image_points[0][0]), int(image_points[0][1]))
        nose_target = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.arrowedLine(frame, nose_tip, nose_target, (255, 0, 0), 1)

        rotation_matrix = np.zeros((3, 3))
        cv2.Rodrigues(rotation_vector,rotation_matrix,jacobian)

        projection_matrix = np.array(
            [[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2] , translation_vector[0]],
            [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], translation_vector[1]],
            [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2] , translation_vector[2]]], dtype="double"
        )

        euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        euler_angles = euler_angles[6]

        nose_bridge = np.array([shape.part(27).x, shape.part(27).y])


        yaw = -euler_angles[1][0]
        pitch = -(nose_tip[1] - nose_target[1]) * 0.05

        if abs(euler_angles[2][0] % 180) <= abs(180 - (euler_angles[2][0] % 180)):
            roll = (euler_angles[2][0] % 180)
        else:
            roll = -1*(180 - (euler_angles[2][0] % 180))

        roll = - roll



        cv2.putText(frame,
                    'yaw = '+ str(yaw), (32, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'pitch = ' + str(pitch), (32, 64), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'roll = ' + str(roll), (32, 96), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)

#--------------------------------------------------------------------------------------------
#   NER - normalized eyebrow ratios for eyebrow movement

        cv2.line(frame, (shape.part(17).x, shape.part(17).y), (shape.part(27).x, shape.part(27).y), (0, 255, 0),
                 thickness=1, lineType=8)  # Line between outer eyebrow and nose bridge
        cv2.line(frame, (shape.part(27).x, shape.part(27).y), (shape.part(26).x, shape.part(26).y), (0, 255, 0),
                 thickness=1, lineType=8)  # Line between outer eyebrow and nose bridge

        outer_brow1 = np.array([shape.part(17).x, shape.part(17).y])
        outer_brow2 = np.array([shape.part(26).x, shape.part(26).y])
        nose_bridge = np.array([shape.part(27).x, shape.part(27).y])

        p3 = np.array([shape.part(18).x, shape.part(18).y])
        d1 = np.linalg.norm(np.cross(outer_brow1 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow1 - nose_bridge)

        p3 = np.array([shape.part(19).x, shape.part(19).y])
        d2 = np.linalg.norm(np.cross(outer_brow1 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow1 - nose_bridge)

        p3 = np.array([shape.part(20).x, shape.part(20).y])
        d3 = np.linalg.norm(np.cross(outer_brow1 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow1 - nose_bridge)

        p3 = np.array([shape.part(21).x, shape.part(21).y])
        d4 = np.linalg.norm(np.cross(outer_brow1 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow1 - nose_bridge)


        ner_r =(d1 + d2 + d3 + d4) / (4 * np.linalg.norm(outer_brow1 - nose_bridge))
        ner_r = ner_r * (1 + (pitch * pitch_normalisation_value))



        p3 = np.array([shape.part(25).x, shape.part(25).y])
        d1 = np.linalg.norm(np.cross(outer_brow2 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow2 - nose_bridge)

        p3 = np.array([shape.part(24).x, shape.part(24).y])
        d2 = np.linalg.norm(np.cross(outer_brow2 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow2 - nose_bridge)

        p3 = np.array([shape.part(23).x, shape.part(23).y])
        d3 = np.linalg.norm(np.cross(outer_brow2 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow2 - nose_bridge)

        p3 = np.array([shape.part(22).x, shape.part(22).y])
        d4 = np.linalg.norm(np.cross(outer_brow2 - nose_bridge, nose_bridge - p3)) / np.linalg.norm(outer_brow2 - nose_bridge)

        ner_l = (d1 + d2 + d3 + d4) / (4 * np.linalg.norm(outer_brow2 - nose_bridge))
        ner_l = ner_l * (1 + (pitch * pitch_normalisation_value))

        cv2.putText(frame,
                    'NER_L = ' + str(ner_l), (32, 128), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'NER_R = ' + str(ner_r), (32, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)

#--------------------------------------------------------------------------------------------
# EAR -Eye Aspect Ratio for eyes closing

        p1 = np.array([shape.part(36).x, shape.part(36).y])
        p2 = np.array([shape.part(37).x, shape.part(37).y])
        p3 = np.array([shape.part(38).x, shape.part(38).y])
        p4 = np.array([shape.part(39).x, shape.part(39).y])
        p5 = np.array([shape.part(40).x, shape.part(40).y])
        p6 = np.array([shape.part(41).x, shape.part(41).y])

        ear_l = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))
        ear_l = ear_l * (1 - (np.abs(yaw)*yaw_normalisation_value))

        p1 = np.array([shape.part(42).x, shape.part(42).y])
        p2 = np.array([shape.part(43).x, shape.part(43).y])
        p3 = np.array([shape.part(44).x, shape.part(44).y])
        p4 = np.array([shape.part(45).x, shape.part(45).y])
        p5 = np.array([shape.part(46).x, shape.part(46).y])
        p6 = np.array([shape.part(47).x, shape.part(47).y])

        ear_r = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))
        ear_r = ear_r * (1 - (np.abs(yaw) * yaw_normalisation_value))

        cv2.putText(frame,
                    'EAR_L = ' + str(ear_l), (32, 192), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'EAR_R = ' + str(ear_r), (32, 224), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)

        # -------------------------------------------
        # Eye Tracking
        left_eye_region = np.array([(shape.part(42).x, shape.part(42).y),
                                    (shape.part(43).x, shape.part(43).y),
                                    (shape.part(44).x, shape.part(44).y),
                                    (shape.part(45).x, shape.part(45).y),
                                    (shape.part(46).x, shape.part(46).y),
                                    (shape.part(47).x, shape.part(47).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.ones((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        gray_eye = cv2.GaussianBlur(gray_eye, (3,3), 0)

        _, threshold_eye = cv2.threshold(gray_eye, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
        M = cv2.moments(threshold_eye)

        if (M["m10"] != 0) and ear_l > EAR_blink_threshold:
            cX_l = M["m10"] / M["m00"]
            cY_l = M["m01"] / M["m00"]
            cX_l = cX_l / (max_x - min_x)
            cY_l = cY_l / (max_y - min_y)
        else:
            cX_l = 0.5
            cY_l = 0.5

        right_eye_region = np.array([(shape.part(36).x, shape.part(36).y),
                                    (shape.part(37).x, shape.part(37).y),
                                    (shape.part(38).x, shape.part(38).y),
                                    (shape.part(39).x, shape.part(39).y),
                                    (shape.part(40).x, shape.part(40).y),
                                    (shape.part(41).x, shape.part(41).y)], np.int32)

        cv2.polylines(mask, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [right_eye_region], 255)
        right_eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = right_eye[min_y: max_y, min_x: max_x]
        gray_eye = cv2.GaussianBlur(gray_eye, (3, 3), 0)

        _, threshold_eye = cv2.threshold(gray_eye, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
        M = cv2.moments(threshold_eye)

        if (M["m10"] != 0) and ear_r > EAR_blink_threshold:
            cX_r = M["m10"] / M["m00"]
            cY_r = M["m01"] / M["m00"]
            cX_r = cX_r / (max_x - min_x)
            cY_r = cY_r / (max_y - min_y)
        else:
            cX_r = 0.5
            cY_r = 0.5

        eX = (cX_l + cX_r) / 2
        eY = min((cY_l + cY_r) / 2, 1)

        print(eY)
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        #cv2.circle(threshold_eye, (cX_l*5, cY_l*5), 2, (0, 0, 255), thickness=1)
        #cv2.imshow("Threshold", threshold_eye)

        cv2.putText(frame,
                    'EX = ' + str(eX), (32, 256), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'EY = ' + str(eY), (32, 288), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)

# --------------------------------------------------------------------------------------------
# OMAR - Outer Mouth Aspect Ratio for mouth stretching, smiling

        p1 = np.array([shape.part(48).x, shape.part(48).y])
        p2 = np.array([shape.part(49).x, shape.part(49).y])
        p3 = np.array([shape.part(50).x, shape.part(50).y])
        p4 = np.array([shape.part(51).x, shape.part(51).y])
        p5 = np.array([shape.part(52).x, shape.part(52).y])
        p6 = np.array([shape.part(53).x, shape.part(53).y])
        p7 = np.array([shape.part(54).x, shape.part(54).y])
        p8 = np.array([shape.part(55).x, shape.part(55).y])
        p9 = np.array([shape.part(56).x, shape.part(56).y])
        p10 = np.array([shape.part(57).x, shape.part(57).y])
        p11 = np.array([shape.part(58).x, shape.part(58).y])
        p12 = np.array([shape.part(59).x, shape.part(59).y])
        p13 = np.array([shape.part(61).x, shape.part(61).y])
        p14 = np.array([shape.part(62).x, shape.part(62).y])
        p15 = np.array([shape.part(63).x, shape.part(63).y])
        p16 = np.array([shape.part(65).x, shape.part(65).y])
        p17 = np.array([shape.part(66).x, shape.part(66).y])
        p18 = np.array([shape.part(67).x, shape.part(67).y])

        omar = (np.linalg.norm(p2 - p12) + np.linalg.norm(p3 - p13) + np.linalg.norm(p4 - p14) +
                np.linalg.norm(p5 - p15) + np.linalg.norm(p6 - p8) + np.linalg.norm(p16 - p9) +
                np.linalg.norm(p16 - p9) + np.linalg.norm(p17 - p10) + np.linalg.norm(p18 - p11)) /\
                (5* + np.linalg.norm(p1 - p7))

        omar = omar * (1 - (np.abs(yaw) * yaw_normalisation_value))

        cv2.putText(frame,
                    'OMAR = ' + str(omar), (32, 320), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)

# --------------------------------------------------------------------------------------------
# IMAR - Inner Mouth Aspect Ratio for showing teeth

        imar = (5 * (np.linalg.norm(p13 - p18) + np.linalg.norm(p14 - p17) + np.linalg.norm(p15 - p16))) /\
               (3 * (np.linalg.norm(p2 - p12) + np.linalg.norm(p3 - p11) + np.linalg.norm(p4 - p10) +
                np.linalg.norm(p5 - p9) + np.linalg.norm(p6 - p8)))

        cv2.putText(frame,
                    'IMAR = ' + str(imar), (32, 352), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------------------------
# N2MAR - Nose to Mouth Ratios
        p1 = np.array([shape.part(27).x, shape.part(27).y]) #nose bridge
        p2 = np.array([shape.part(31).x, shape.part(31).y]) #right nose wing
        p3 = np.array([shape.part(35).x, shape.part(35).y]) #left nose wing
        p4 = np.array([shape.part(32).x, shape.part(32).y]) #right nostril
        p5 = np.array([shape.part(52).x, shape.part(52).y])
        p6 = np.array([shape.part(53).x, shape.part(53).y])
        p7 = np.array([shape.part(54).x, shape.part(54).y])
        p8 = np.array([shape.part(55).x, shape.part(55).y])
        p9 = np.array([shape.part(56).x, shape.part(56).y])



# --------------------------------------------------------------------------------------------

        cv2.imshow("image", frame) #Display the frame

        head_output_buffer[0].append(yaw)
        head_output_buffer[1].append(pitch)
        head_output_buffer[2].append(roll)
        head_output_buffer[3].append(ner_l)
        head_output_buffer[4].append(ner_r)
        head_output_buffer[5].append(ear_l)
        head_output_buffer[6].append(ear_r)
        head_output_buffer[7].append(eX)
        head_output_buffer[8].append(eY)
        head_output_buffer[9].append(omar)
        head_output_buffer[10].append(imar)

        if (len(head_output_buffer[0]) >= frame_interval):
            print(str(statistics.median(head_output_buffer[0])) + ', ' + str(statistics.median(head_output_buffer[1])) + ', ' + str(statistics.median(head_output_buffer[2])) + ', ' + str(statistics.median(head_output_buffer[3])) + ', ' + str(statistics.median(head_output_buffer[4])))
            final_output.append([statistics.median(head_output_buffer[0]),
                                 statistics.median(head_output_buffer[1]),
                                 statistics.median(head_output_buffer[2]),
                                 statistics.median(head_output_buffer[3]),
                                 statistics.median(head_output_buffer[4]),
                                 statistics.median(head_output_buffer[5]),
                                 statistics.median(head_output_buffer[6]),
                                 statistics.median(head_output_buffer[7]),
                                 statistics.median(head_output_buffer[8]),
                                 statistics.median(head_output_buffer[9]),
                                 statistics.median(head_output_buffer[10])])
            for datum in head_output_buffer:
                datum.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
            break
    else:
        if (len(head_output_buffer[0]) > 0):
            #print(str(statistics.median(head_output_buffer[0])) + ', ' + str(statistics.median(head_output_buffer[1])) + ', ' + str(statistics.median(head_output_buffer[2])))
            final_output.append([statistics.median(head_output_buffer[0]),
                                 statistics.median(head_output_buffer[1]),
                                 statistics.median(head_output_buffer[2]),
                                 statistics.median(head_output_buffer[3]),
                                 statistics.median(head_output_buffer[4]),
                                 statistics.median(head_output_buffer[5]),
                                 statistics.median(head_output_buffer[6]),
                                 statistics.median(head_output_buffer[7]),
                                 statistics.median(head_output_buffer[8]),
                                 statistics.median(head_output_buffer[9]),
                                 statistics.median(head_output_buffer[10])])

        video_capture.release()
        cv2.destroyAllWindows()
with open('output.csv', mode='w') as employee_file:
    writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['yaw', 'pitch', 'roll', 'NER_L', 'NER_R', 'EAR_L', 'EAR_R', 'E_X', 'E_Y', 'OMAR', 'IMAR'])
    for row in final_output:
        writer.writerow(row)


