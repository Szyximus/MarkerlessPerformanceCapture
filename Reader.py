#Import required modules
import cv2
import dlib
import numpy as np
#Set up some required objects
video_capture = cv2.VideoCapture('Source.mp4') #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("Dataset/shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Histogram Equalisation
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image

    # POSE ESTIMATION:
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals

    focal_length = video_capture.get(3)
    center = (video_capture.get(3) / 2, video_capture.get(4))
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )


    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates

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



        #POSE ESTIMATION:

        # 2D points for pose estimation
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),  # Chin
            (shape.part(45).x, shape.part(45).y),  # Left eye left corner
            (shape.part(36).x, shape.part(36).y),  # Right eye right corner
            (shape.part(12).x, shape.part(12).y),  # Left Mouth corner
            (shape.part(4).x, shape.part(4).y)  # Right mouth corner
        ], dtype="double")

        print(
        "Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs)

        print(
        "Rotation Vector:\n {0}".format(rotation_vector))
        print(
        "Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

    cv2.imshow("image", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break

video_capture.release()
cv2.destroyAllWindows()