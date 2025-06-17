import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('Project 2 Diabetes Data.csv')

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']

# Splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

# Model Training - Support Vector Machine Classifier
classifier = svm.SVC(kernel='linear', class_weight='balanced')
classifier.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the training data : ', train_accuracy)
print('Accuracy score of the test data : ', test_accuracy)
print('F1-score of the test data : ', f1_score(Y_test, X_test_prediction))

# Calculate confusion matrix
cm = confusion_matrix(Y_test, X_test_prediction)
TN, FP, FN, TP = cm.ravel()

print('\nTrue Positives (TP):', TP)
print('True Negatives (TN):', TN)
print('False Positives (FP):', FP)
print('False Negatives (FN):', FN)

# Saving the trained model
filename= 'diabetes_model.sav'
pickle.dump(classifier,open(filename,'wb'))

# Save Y_test, X_test_prediction, and feature importances (coefficients for SVM)
pickle.dump(Y_test, open('diabetes_Y_test.pkl', 'wb'))
pickle.dump(X_test_prediction, open('diabetes_X_test_prediction.pkl', 'wb'))
pickle.dump(classifier.coef_[0], open('diabetes_feature_importances.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('diabetes_feature_names.pkl', 'wb'))

# Creating a prediction function
def diabetes_prediction(input_data):
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # load the saved model
    loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic' 