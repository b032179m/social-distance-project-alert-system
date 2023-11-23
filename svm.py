# from skimage.feature import hog
# #from skimage.io import imread
# import joblib,glob,os,cv2
#
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn import svm, metrics
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
# train_data = []
# train_labels = []
# pos_im_path = 'DATAIMAGE/positive'
# neg_im_path = 'DATAIMAGE/negative'
# model_path = 'models.dat'
# # Load the positive features
# for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
#     fd = cv2.imread(filename,0)
#     fd = cv2.resize(fd,(64,128))
#     fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
#     train_data.append(fd)
#     train_labels.append(1)
#
# # Load the negative features
# for filename in glob.glob(os.path.join(neg_im_path,"*.jpg")):
#     fd = cv2.imread(filename,0)
#     fd = cv2.resize(fd,(64,128))
#     fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
#     train_data.append(fd)
#     train_labels.append(0)
# train_data = np.float32(train_data)
# train_labels = np.array(train_labels)
# print('Data Prepared........')
# print('Train Data:',len(train_data))
# print('Train Labels (1,0)',len(train_labels))
# print("""
# Classification with SVM
#
# """)
#
# model = LinearSVC()
# print('Training...... Support Vector Machine')
# model.fit(train_data,train_labels)
# joblib.dump(model, 'models/models.dat')
# print('Model saved : {}'.format('models/models.dat'))


from skimage.feature import hog
import joblib
import glob
import os
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

train_data = []
train_labels = []
pos_im_path = 'DATAIMAGE/positive'
neg_im_path = 'DATAIMAGE/negative'
model_path = 'models.dat'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path, "*.png")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path, "*.jpg")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    train_data.append(fd)
    train_labels.append(0)

train_data = np.float32(train_data)
train_labels = np.array(train_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

print('Data Prepared........')
print('Train Data:', len(X_train))
print('Test Data:', len(X_test))

# Classification with SVM
model = LinearSVC()
print('Training...... Support Vector Machine')
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, model_path)
print('Model saved: {}'.format(model_path))

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
