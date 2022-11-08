import cv2
import os
import PoseModule as pm
import csv
# from PIL import Image
# import numpy as np


# cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

folder_path = "C:/Users/gowri/OneDrive - Purdue University Fort Wayne/Documents/PFW related/Thesis/Stage 2 (Pose classification)/PoseEstimation_mediapipe/thesis-pose-estimation/Aug_Data"
folders = os.listdir(folder_path)
print(folders)

with open('poseLandmarks_flip.csv', mode='w') as poseLandmark_file:
    landmark_writer = csv.writer(poseLandmark_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    landmark_writer.writerow(["LWN","RWN","LWLS","RWRS","LWRE","RWLE",'LWRH','LELH','RWLH','RERH','LHRHMN','LWRW','LERE','LEA','REA','RSA','LSA','LHTA','RHTA','ETRI','WTRI','HTRI','Pose'])
    final_list = []
    c = 0
    for pose in folders:
        print(pose)
        temp = []
        path = folder_path + "/" + pose
        count = 0
        for file in os.listdir(path):
            file_path = path + "/" + file
            print(file)
            img = cv2.imread(file_path)
            if img is not None:
                # print("image not none", count)
                img = detector.findPose(img)
                lmList = detector.findPosition(img)
                if len(lmList) != 0:
                    # print("list not none",count)
                    featureList = detector.poseFeatures(lmList)
                    featureList.append(pose)
                    landmark_writer.writerow(featureList)
                    temp.append(featureList)
                    count += 1
                    print(count)
                else:
                    print("image empty")
        final_list.append(temp)
        c += 1


print(c)
# print(final_list)

# with open('poselandmarks_all.csv', mode='w') as poseLandmark_file:
#     landmark_writer = csv.writer(poseLandmark_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     landmark_writer.writerow([final_list])
