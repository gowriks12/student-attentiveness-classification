import Augmentor
import os

folder_path = "C:/Users/gowri/OneDrive - Purdue University Fort Wayne/Documents/PFW related/Thesis/Stage 2 (Pose classification)/PoseEstimation_mediapipe/thesis-pose-estimation/Create_database/Hand on chin"
# path = os.path.join(folder_path , folder)
p = Augmentor.Pipeline(folder_path)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_random(probability=0.8)
p.flip_left_right(probability=0.5)
p.sample(10)

# folders = os.listdir(folder_path)
# for folder in folders:
#     path = os.path.join(folder_path , folder)
#     p = Augmentor.Pipeline(path)
#     p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
#     p.flip_random(probability=0.8)
#     p.flip_left_right(probability=0.5)
#     p.sample(10)