import cv2
import os
import PoseModule as pm
import numpy as np
import csv

detector = pm.PoseDetector()

folder_path = "C:/Users/gowri/OneDrive - Purdue University Fort Wayne/Documents/PFW related/Thesis/Stage 2 (Pose classification)/PoseEstimation_mediapipe/thesis-pose-estimation/Aug_Data"
folders = os.listdir(folder_path)

with open('pose_region_flip.csv', mode='w') as poseMoments_file:
    moments_writer = csv.writer(poseMoments_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    moments_writer.writerow(["bodyMajor","bodyAngle","bodyArea","lh1_x","lh1_y","lh1Major","lh1Angle","lh1Area","rh1_x","rh1_y","rh1Major","rh1Angle","rh1Area","lh2_x","lh2_y","lh2Major","lh2Angle","lh2Area","rh2_x","rh2_y","rh2Major","rh2Angle","rh2Area",'Pose'])
    # final_list = []
    # c = 0
    for pose in folders:
        print(pose)
        # temp = []
        path = folder_path + "/" + pose
        # count = 0
        for file in os.listdir(path):
            file_path = path + "/" + file
            print(file)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.resize(image, (350, 350), interpolation=cv2.INTER_AREA)
                img = detector.findPose(image, draw=False)
                lmList = detector.findPosition(img, draw=False)
                if len(lmList) != 0:
                    (row, image) = detector.regionFeatures(lmList, image, thickness=3)
                    row = row[2:]
                    row.append(pose)
                    print("done")
                    moments_writer.writerow(row)
                else:
                    print("list empty")
            else:
                print("image empty")




# with open('poselandmarks.csv', mode='w') as poseLandmark_file:
#     landmark_writer = csv.writer(poseLandmark_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     landmark_writer.writerow(["Landmarks", "class"])



