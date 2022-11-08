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


def create_knn_model(X_train,X_test,y_train,y_test):
    # Creating KNN Model
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k,weights = 'distance', metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']

    print(classification_report(y_test, y_pred, target_names=class_names))

    return knn


def create_svm_model(X_train,X_test,y_train,y_test):
    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    accuracy_lin = linear.score(X_test, y_test)
    accu = accuracy_score(y_test, linear_pred)
    print("SVM Accuracy:", accuracy_lin)
    print(confusion_matrix(y_test, linear_pred))
    class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']

    print(classification_report(y_test, linear_pred, target_names=class_names))

    return linear


def create_random_forest_model(X_train,X_test,y_train,y_test):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=1)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("RF Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']

    print(classification_report(y_test, y_pred, target_names=class_names))

    return clf


pose_region = pd.read_csv('pose_region_flip.csv')
pose_landmark = pd.read_csv('poseLandmarks_flip.csv')

r_X_train,r_X_test, r_y_train, r_y_test = data_split(pose_region)
l_X_train,l_X_test, l_y_train, l_y_test = data_split(pose_landmark)

class_names = ['Hand on chin', 'Hand on head', 'Hand raised', 'Hand Crossed', 'Leaning forward', 'Looking down',
                   'Looking to the side', 'Sitting Straight', 'Sleeping', 'Taking notes']
# Pipeline implementation

# REGION
print("REGION")
knn_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('knn', KNeighborsClassifier(n_neighbors=3,weights = 'distance', metric='euclidean'))])
knn_pipe.fit(r_X_train,r_y_train)
k_pred = knn_pipe.predict(r_X_test)
print("knn pipe accu", knn_pipe.score(r_X_test,r_y_test))
print("knn pipe f1", f1_score(r_y_test,k_pred, average=None))
# print("knn pipe f1", classification_report(r_y_test,k_pred))

svm_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('svm', svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))])
svm_pipe.fit(r_X_train,r_y_train)
s_pred = svm_pipe.predict(r_X_test)
print("SVM pipe accu", svm_pipe.score(r_X_test,r_y_test))
print("SVM pipe f1", f1_score(r_y_test,s_pred, average=None))
# print("SVM pipe f1", classification_report(r_y_test,s_pred))

rf_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('rf', RandomForestClassifier(n_estimators=100, random_state=1))])
rf_pipe.fit(r_X_train,r_y_train)
r_pred = rf_pipe.predict(r_X_test)
print("RF pipe accu", rf_pipe.score(r_X_test,r_y_test))
print("RF pipe f1", f1_score(r_y_test,r_pred, average=None))
# print("RF pipe f1", classification_report(r_y_test,r_pred))

# F1 Score table
lis = [class_names, f1_score(r_y_test,k_pred, average=None), f1_score(r_y_test,s_pred, average=None),  f1_score(r_y_test,r_pred, average=None)]
f1 = pd.DataFrame(class_names,columns=['Class Name'])
f1['KNN F1 Scores'] = f1_score(r_y_test,k_pred, average=None)
f1['SVM F1 Scores'] = f1_score(r_y_test,s_pred, average=None)
f1['RF F1 Scores'] = f1_score(r_y_test,r_pred, average=None)

# ,,'SVM F1 Score','RF F1 Score'
print(f1)

# Get and reshape confusion matrix data
matrix = confusion_matrix(r_y_test,k_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(10,10))
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
plt.title('Confusion Matrix for KNN Model')
plt.show()




# LANDMARKS
# print("LANDMARKS")
# knn_land_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('knn', KNeighborsClassifier(n_neighbors=3,weights = 'distance', metric='euclidean'))])
# knn_land_pipe.fit(l_X_train,l_y_train)
# kl_pred = knn_land_pipe.predict(l_X_test)
# print("knn pipe accu", knn_land_pipe.score(l_X_test,l_y_test))
# print("knn pipe f1", f1_score(l_y_test,kl_pred, average=None))
#
# svm_land_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('svm', svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))])
# svm_land_pipe.fit(l_X_train,l_y_train)
# sl_pred = svm_land_pipe.predict(l_X_test)
# print("SVM pipe accu", svm_land_pipe.score(l_X_test,l_y_test))
# print("SVM pipe f1", f1_score(l_y_test,sl_pred, average = 'samples'))
#
# rf_land_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('rf', RandomForestClassifier(n_estimators=100, random_state=1))])
# rf_land_pipe.fit(l_X_train,l_y_train)
# rl_pred = rf_land_pipe.predict(l_X_test)
# print("RF pipe accu", rf_land_pipe.score(l_X_test,l_y_test))
# print("RF pipe f1", f1_score(l_y_test,rl_pred, average = 'samples'))

# Normal Implementation
# regionScaler = StandardScaler()
# landScaler = StandardScaler()
#
# r_X_train_scaled = pd.DataFrame(regionScaler.fit_transform(r_X_train.iloc[:,:]), columns= r_X_train.columns[:])
# l_X_train_scaled = pd.DataFrame(landScaler.fit_transform(l_X_train.iloc[:,:]), columns= l_X_train.columns[:])
#
# r_X_test_scaled = pd.DataFrame(regionScaler.transform(r_X_test.iloc[:,:]), columns= r_X_test.columns[:])
# l_X_test_scaled = pd.DataFrame(landScaler.transform(l_X_test.iloc[:,:]), columns= l_X_test.columns[:])
#
# pca_region = PCA(n_components=0.95)
# pca_land = PCA(n_components=0.95)
#
# pc_regions_train = pd.DataFrame(pca_region.fit_transform(r_X_train_scaled))
# pc_regions_test = pd.DataFrame(pca_region.transform(r_X_test_scaled))
#
# pc_land_train = pd.DataFrame(pca_land.fit_transform(l_X_train_scaled))
# pc_land_test = pd.DataFrame(pca_land.transform(l_X_test_scaled))


# knn_region_model = create_knn_model(pc_regions_train,pc_regions_test, r_y_train, r_y_test)
# svm_region_model = create_svm_model(pc_regions_train,pc_regions_test, r_y_train, r_y_test)
# rf_region_model = create_random_forest_model(pc_regions_train,pc_regions_test, r_y_train, r_y_test)


# knn_land_model = create_knn_model(pc_land_train,pc_land_test, l_y_train, l_y_test)
# svm_land_model = create_svm_model(pc_land_train,pc_land_test, l_y_train, l_y_test)
# rf_land_model = create_random_forest_model(pc_land_train,pc_land_test, l_y_train, l_y_test)


# knn_region_pipe_pca = open('knn_r_pca_pipe.sav', 'wb')
# svm_region_pipe_pca = open('svm_r_pca_pipe.sav', 'wb')
# rf_region_pipe_pca = open('rf_r_pca_pipe.sav', 'wb')
#
# knn_land_pipe_pca = open('knn_l_pca_pipe.sav', 'wb')
# svm_land_pipe_pca = open('svm_l_pca_pipe.sav', 'wb')
# rf_land_pipe_pca = open('rf_l_pca_pipe.sav', 'wb')
#
# # knn_land = open('knnpickle_landmarks.sav', 'wb')
#
# # source, destination
# pickle.dump(knn_pipe, knn_region_pipe_pca)
# pickle.dump(svm_pipe, svm_region_pipe_pca)
# pickle.dump(rf_pipe, rf_region_pipe_pca)
#
# pickle.dump(knn_land_pipe, knn_land_pipe_pca)
# pickle.dump(svm_land_pipe, svm_land_pipe_pca)
# pickle.dump(rf_land_pipe, rf_land_pipe_pca)



