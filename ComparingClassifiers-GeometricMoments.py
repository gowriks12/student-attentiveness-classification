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
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def data_split(df):
    labels = df["Pose"].tolist()
    # Train test split
    X = df.drop(columns=['Pose'])
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    return X_train, X_test, y_train, y_test

pose_region = pd.read_csv('pose_region_flip.csv')

r_X_train,r_X_test, r_y_train, r_y_test = data_split(pose_region)

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3,weights = 'distance', metric='euclidean'))])
knn_pipe.fit(r_X_train,r_y_train)
knn_y_pred = knn_pipe.predict(r_X_test)
# print("knn pipe accu", knn_pipe.score(r_X_test,r_y_test))
# print(confusion_matrix(r_y_test, knn_y_pred))
class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']

# print(classification_report(r_y_test, knn_y_pred, target_names=class_names))


svm_pipe = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))])
svm_pipe.fit(r_X_train,r_y_train)
# print("SVM pipe accu", svm_pipe.score(r_X_test,r_y_test))
svm_y_pred = svm_pipe.predict(r_X_test)
# print(confusion_matrix(r_y_test, svm_y_pred))
class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']
# print(classification_report(r_y_test, svm_y_pred, target_names=class_names))

rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, random_state=1))])
rf_pipe.fit(r_X_train,r_y_train)
# print("RF pipe accu", rf_pipe.score(r_X_test,r_y_test))
rf_y_pred = rf_pipe.predict(r_X_test)
# print(confusion_matrix(r_y_test, rf_y_pred))
class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']
# print(classification_report(r_y_test, rf_y_pred, target_names=class_names))

# F1 Score table
# lis = [class_names, f1_score(r_y_test,k_pred, average=None), f1_score(r_y_test,s_pred, average=None),  f1_score(r_y_test,r_pred, average=None)]
f1 = pd.DataFrame(class_names,columns=['Class Name'])
f1['KNN F1 No PCA'] = f1_score(r_y_test,knn_y_pred, average=None)
f1['SVM F1 No PCA'] = f1_score(r_y_test,svm_y_pred, average=None)
f1['RF F1 No PCA'] = f1_score(r_y_test,rf_y_pred, average=None)
print(f1)
# knn_region_pickle = open('knn_pipe_pickle.sav', 'wb')
# svm_region_pickle = open('svm_pipe_pickle.sav', 'wb')
# rf_region_pickle = open('rf_pipe_pickle.sav', 'wb')
# # knn_land = open('knnpickle_landmarks.sav', 'wb')
#
# # # source, destination
# pickle.dump(knn_pipe, knn_region_pickle)
# pickle.dump(svm_pipe, svm_region_pickle)
# pickle.dump(rf_pipe, rf_region_pickle)
# pickle.dump(knn_land_model, knn_land)


# Get and reshape confusion matrix data
matrix = confusion_matrix(r_y_test,rf_y_pred)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
# class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down', 'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
