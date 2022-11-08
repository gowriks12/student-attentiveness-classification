import cv2
import mediapipe as mp
import numpy as np
from sklearn import preprocessing


class PoseDetector:

    def __init__(self,mode = False , complexity = 1, smooth = True, en_segmentation = False, sm_segmentation = True, det_conf = 0.7, tr_conf = 0.7):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.en_segmentation = en_segmentation
        self.sm_segmentation = sm_segmentation
        self.det_conf = det_conf
        self.tr_conf = tr_conf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.en_segmentation, self.sm_segmentation, self.det_conf, self.tr_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        # Finds position of only the upper body
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id <= 24:
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList

    def angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def kp_distance(self, kp1, kp2):
        dist = abs(((kp2[1] - kp1[1])**2 + (kp2[0] - kp1[0])**2)**0.5)
        return dist

    def poseFeatures(self, lmList):
        featureList = []
        NOSE = lmList[0][1:]
        LEFT_SHOULDER = lmList[11][1:]
        RIGHT_SHOULDER = lmList[12][1:]
        LEFT_ELBOW = lmList[13][1:]
        RIGHT_ELBOW = lmList[14][1:]
        LEFT_WRIST = lmList[15][1:]
        RIGHT_WRIST = lmList[16][1:]
        LEFT_HIP = lmList[23][1:]
        RIGHT_HIP = lmList[24][1:]
        LHRH_MID = [(LEFT_HIP[0]+RIGHT_HIP[0])/2, (LEFT_HIP[1]+RIGHT_HIP[1])/2]
        LSRS_MID = [(LEFT_SHOULDER[0] + RIGHT_SHOULDER[0]) / 2, (LEFT_SHOULDER[1] + RIGHT_SHOULDER[1]) / 2]
        LERE_MID = [(LEFT_ELBOW[0] + RIGHT_ELBOW[0]) / 2, (LEFT_ELBOW[1] + RIGHT_ELBOW[1]) / 2]
        LWRW_MID = [(LEFT_WRIST[0] + RIGHT_WRIST[0]) / 2, (LEFT_WRIST[1] + RIGHT_WRIST[1]) / 2]

        # FeaturesComputation

        # Distances
        LWN = self.kp_distance(LEFT_WRIST,NOSE)
        RWN = self.kp_distance(RIGHT_WRIST,NOSE)
        LWLS = self.kp_distance(LEFT_WRIST,LEFT_SHOULDER)
        RWRS = self.kp_distance(RIGHT_WRIST,RIGHT_SHOULDER)
        LWRE = self.kp_distance(LEFT_WRIST,RIGHT_ELBOW)
        RWLE = self.kp_distance(RIGHT_WRIST,LEFT_ELBOW)
        LWRH = self.kp_distance(LEFT_WRIST,RIGHT_HIP)
        LELH = self.kp_distance(LEFT_ELBOW,LEFT_HIP)
        RWLH = self.kp_distance(RIGHT_WRIST,LEFT_HIP)
        RERH = self.kp_distance(RIGHT_ELBOW,RIGHT_HIP)
        LHRHMN = self.kp_distance(LHRH_MID,NOSE)


        LERE = self.kp_distance(LEFT_ELBOW,RIGHT_ELBOW) # b
        LWRW = self.kp_distance(LEFT_WRIST, RIGHT_WRIST) # b
        LEREMN = self.kp_distance(LERE_MID, NOSE) # h
        LWRWMN = self.kp_distance(LWRW_MID,NOSE)  # h
        LHRHMW = self.kp_distance(LHRH_MID, LWRW_MID) #h

        # Angles
        LEA = self.angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        REA = self.angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        RSA = self.angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
        LSA = self.angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
        LHTA = self.angle(LEFT_SHOULDER, LSRS_MID, NOSE)
        RHTA = self.angle(NOSE, LSRS_MID, RIGHT_SHOULDER)

        # Triangle area
        ETRI = (LERE * LEREMN)/2
        WTRI = (LWRW * LWRWMN)/2
        HTRI = (LWRW * LHRHMW)/2  # LWRW LHRHM triangle area


        featureList.append(round(LWN, 4))
        featureList.append(round(RWN, 4))
        featureList.append(round(LWLS,4))
        featureList.append(round(RWRS,4))
        featureList.append(round(LWRE,4))
        featureList.append(round(RWLE,4))
        featureList.append(round(LWRH, 4))
        featureList.append(round(LELH, 4))
        featureList.append(round(RWLH, 4))
        featureList.append(round(RERH, 4))
        featureList.append(round(LHRHMN, 4))
        featureList.append(round(LEA, 4))
        featureList.append(round(REA, 4))
        featureList.append(round(RSA, 4))
        featureList.append(round(LSA, 4))
        featureList.append(round(LHTA, 4))
        featureList.append(round(RHTA, 4))
        featureList.append(round(ETRI, 4))
        featureList.append(round(WTRI, 4))
        featureList.append(round(HTRI, 4))

        return featureList


    def regionFeatures(self, lmList, img,thickness = 3,start = 0,end = 360):
        thickness = thickness
        start = start
        end = end

        # Nose to Hip region
        NOSE = lmList[0][1:]
        LEFT_HIP = lmList[23][1:]
        RIGHT_HIP = lmList[24][1:]
        LHRH_MID = [int((LEFT_HIP[0] + RIGHT_HIP[0]) / 2), int((LEFT_HIP[1] + RIGHT_HIP[1]) / 2)]
        a = int(self.kp_distance(NOSE, LHRH_MID))
        b = int((1 / 3) * a)

        area = round(3.142*a*b,3)

        dx = LHRH_MID[0] - NOSE[0]
        dy = LHRH_MID[1] - NOSE[1]
        m = round(np.rad2deg(np.arctan2(dy, dx)),3)

        bodyCentre = [int((NOSE[0] + LHRH_MID[0]) / 2), int((NOSE[1] + LHRH_MID[1]) / 2)]
        b_x = 0
        b_y = 0

        color = (0, 255, 0)
        axes = (a, b)
        bodyRegion = [b_x, b_y, a, m, area]
        # bodyRegion.append(centre,axes,m)
        img = cv2.ellipse(img, bodyCentre, axes, m, start, end, color, thickness)


        # Right shoulder to Elbow ellipse
        RIGHT_SHOULDER = lmList[12][1:]
        RIGHT_ELBOW = lmList[14][1:]
        a = int(self.kp_distance(RIGHT_SHOULDER, RIGHT_ELBOW))
        b = int((1 / 3) * a)

        area = round(3.142 * a * b,3)

        dx = RIGHT_ELBOW[0] - RIGHT_SHOULDER[0]
        dy = RIGHT_ELBOW[1] - RIGHT_SHOULDER[1]
        m = round(np.rad2deg(np.arctan2(dy, dx)),3)

        centre = [int((RIGHT_SHOULDER[0] + RIGHT_ELBOW[0]) / 2), int((RIGHT_SHOULDER[1] + RIGHT_ELBOW[1]) / 2)]
        rh1_x = bodyCentre[0] - centre[0]
        rh1_y = bodyCentre[1] - centre[1]

        # Blue color in BGR
        color = (255, 0, 0)
        axes = (a, b)

        rightHand1 = [rh1_x, rh1_y, a, m, area]
        # rightHand1.append()

        img = cv2.ellipse(img, centre, axes, m, start, end, color, thickness)

        # Left shoulder to Elbow ellipse
        LEFT_SHOULDER = lmList[11][1:]
        LEFT_ELBOW = lmList[13][1:]
        a = int(self.kp_distance(LEFT_ELBOW, LEFT_SHOULDER))
        b = int((1 / 3) * a)

        area = round(3.142 * a * b,3)

        dx = LEFT_ELBOW[0] - LEFT_SHOULDER[0]
        dy = LEFT_ELBOW[1] - LEFT_SHOULDER[1]
        m = round(np.rad2deg(np.arctan2(dy, dx)),3)

        centre = [int((LEFT_SHOULDER[0] + LEFT_ELBOW[0]) / 2), int((LEFT_SHOULDER[1] + LEFT_ELBOW[1]) / 2)]
        lh1_x = bodyCentre[0] - centre[0]
        lh1_y = bodyCentre[1] - centre[1]

        # Blue color in BGR
        color = (255, 0, 0)
        axes = (a, b)

        leftHand1 = [lh1_x, lh1_y, a, m, area]
        # leftHand1.append()
        img = cv2.ellipse(img, centre, axes, m, start, end, color, thickness)

        # Right elbow to wrist ellipse
        RIGHT_WRIST = lmList[16][1:]
        RIGHT_ELBOW = lmList[14][1:]
        a = int(self.kp_distance(RIGHT_WRIST, RIGHT_ELBOW))
        b = int((1 / 3) * a)

        area = round(3.142 * a * b,3)

        dx = RIGHT_WRIST[0] - RIGHT_ELBOW[0]
        dy = RIGHT_WRIST[1] - RIGHT_ELBOW[1]
        m = round(np.rad2deg(np.arctan2(dy, dx)),3)

        centre = [int((RIGHT_ELBOW[0] + RIGHT_WRIST[0]) / 2), int((RIGHT_ELBOW[1] + RIGHT_WRIST[1]) / 2)]
        rh2_x = bodyCentre[0] - centre[0]
        rh2_y = bodyCentre[1] - centre[1]

        # Blue color in BGR
        color = (0, 0, 255)
        axes = (a, b)

        rightHand2 = [rh2_x, rh2_y, a, m, area]
        # rightHand2.append()
        img = cv2.ellipse(img, centre, axes, m, start, end, color, thickness)

        # Left Elbow to wrist ellipse
        LEFT_WRIST = lmList[15][1:]
        LEFT_ELBOW = lmList[13][1:]
        a = int(self.kp_distance(LEFT_ELBOW, LEFT_WRIST))
        b = int((1 / 3) * a)

        area = round(3.142 * a * b,3)

        dx = LEFT_WRIST[0] - LEFT_ELBOW[0]
        dy = LEFT_WRIST[1] - LEFT_ELBOW[1]
        m = round(np.rad2deg(np.arctan2(dy, dx)),3)

        centre = [int((LEFT_ELBOW[0] + LEFT_WRIST[0]) / 2), int((LEFT_ELBOW[1] + LEFT_WRIST[1]) / 2)]
        lh2_x = bodyCentre[0] - centre[0]
        lh2_y = bodyCentre[1] - centre[1]

        # Blue color in BGR
        color = (0, 0, 255)
        axes = (a, b)

        leftHand2 = [lh2_x, lh2_y, a, m, area]
        # leftHand2.append()
        img = cv2.ellipse(img, centre, axes, m, start, end, color, thickness)
        regions = [bodyRegion, leftHand1, rightHand1, leftHand2,rightHand2]
        row = []
        for region in regions:
            for i in range(len(region)):
                row.append(region[i])

        return row, img

    def poseClassifier(self,image,model):
        loaded_model = model
        image = self.findPose(image, draw=False)
        lmList = self.findPosition(image, draw=False)
        if len(lmList) != 0:
            (row, image) = self.regionFeatures(lmList, image, thickness=3)
            row = row[2:]
            F = np.array([row])
            F.reshape(1, -1)
            # print(F.shape)
            # scaler = preprocessing.StandardScaler().fit(F)
            # row = scaler.transform(F)
            prediction = loaded_model.predict(F)
            probs = loaded_model.predict_proba(F)
            className = prediction[0]
            # confidence = max(probs)
            # print(confidence)
            # confi = round(max(confidence),2)

            return image, className, probs
        else:
            return (image, "null",None)



def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        # print(lmList)
        featureList = detector.poseFeatures(lmList)
        # print("Pose:", pose)
        print("Features", featureList)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
