import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

def data_split(df):
    labels = df["Pose"].tolist()
    # Train test split
    X = df.drop(columns=['Pose'])
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    return X_train, X_test, y_train, y_test

pose_landmark = pd.read_csv('poseLandmarks_flip.csv')

r_X_train,r_X_test, r_y_train, r_y_test = data_split(pose_landmark)

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3,weights = 'distance', metric='euclidean'))])
knn_pipe.fit(r_X_train,r_y_train)
knn_y_pred = knn_pipe.predict(r_X_test)
print("knn pipe accu", knn_pipe.score(r_X_test,r_y_test))
print(confusion_matrix(r_y_test, knn_y_pred))
class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']

print(classification_report(r_y_test, knn_y_pred, target_names=class_names))


svm_pipe = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))])
svm_pipe.fit(r_X_train,r_y_train)
print("SVM pipe accu", svm_pipe.score(r_X_test,r_y_test))
svm_y_pred = svm_pipe.predict(r_X_test)
print(confusion_matrix(r_y_test, svm_y_pred))
class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']
print(classification_report(r_y_test, svm_y_pred, target_names=class_names))

rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, random_state=1))])
rf_pipe.fit(r_X_train,r_y_train)
print("RF pipe accu", rf_pipe.score(r_X_test,r_y_test))
rf_y_pred = rf_pipe.predict(r_X_test)
print(confusion_matrix(r_y_test, rf_y_pred))
class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']
print(classification_report(r_y_test, rf_y_pred, target_names=class_names))


# knn_land_pickle = open('knn_pipe_land_pickle.sav', 'wb')
# svm_land_pickle = open('svm_pipe_land_pickle.sav', 'wb')
# rf_land_pickle = open('rf_pipe_land_pickle.sav', 'wb')
# # knn_land = open('knnpickle_landmarks.sav', 'wb')
#
# # # source, destination
# pickle.dump(knn_pipe, knn_land_pickle)
# pickle.dump(svm_pipe, svm_land_pickle)
# pickle.dump(rf_pipe, rf_land_pickle)
# # pickle.dump(knn_land_model, knn_land)
