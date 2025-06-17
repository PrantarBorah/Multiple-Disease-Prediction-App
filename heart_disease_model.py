import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import pickle
import matplotlib.pyplot as plt

# loading the heart disease dataset to a pandas DataFrame
heart_dataset = pd.read_csv('Project 9 Heart Disease Data.csv')

# separating the data and labels
X = heart_dataset.drop(columns='target', axis=1)
Y = heart_dataset['target']

# Splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training - Random Forest Classifier
classifier = RandomForestClassifier(random_state=2, max_depth=10, min_samples_leaf=5)
classifier.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Training data : ', training_data_accuracy)
print('Accuracy on Test data : ', test_data_accuracy)
print('F1-score on Test data : ', f1_score(Y_test, X_test_prediction))

# Calculate confusion matrix
cm = confusion_matrix(Y_test, X_test_prediction)
TN, FP, FN, TP = cm.ravel()

print('\nTrue Positives (TP):', TP)
print('True Negatives (TN):', TN)
print('False Positives (FP):', FP)
print('False Negatives (FN):', FN)

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

accuracies = []
f1_scores = []

print("\n--- K-Fold Cross-Validation Results ---")
for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    Y_train_fold, Y_val_fold = Y.iloc[train_index], Y.iloc[val_index]

    fold_classifier = RandomForestClassifier(random_state=2, max_depth=10, min_samples_leaf=5)
    fold_classifier.fit(X_train_fold, Y_train_fold)
    Y_pred_fold = fold_classifier.predict(X_val_fold)

    accuracies.append(accuracy_score(Y_val_fold, Y_pred_fold))
    f1_scores.append(f1_score(Y_val_fold, Y_pred_fold))

print(f"Mean Accuracy (K-Fold CV): {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Mean F1-Score (K-Fold CV): {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

# Calculate ROC Curve and AUC for initial test data
Y_pred_proba = classifier.predict_proba(X_test)[:, 1] # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
roc_auc = auc(fpr, tpr)

# Saving the trained model
filename = 'heart_disease_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Save Y_test, X_test_prediction, feature importances, and ROC data
pickle.dump(Y_test, open('heart_disease_Y_test.pkl', 'wb'))
pickle.dump(X_test_prediction, open('heart_disease_X_test_prediction.pkl', 'wb'))
pickle.dump(classifier.feature_importances_, open('heart_disease_feature_importances.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('heart_disease_feature_names.pkl', 'wb'))
pickle.dump({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, open('heart_disease_roc_curve.pkl', 'wb')) # Save ROC data

# Creating a prediction function
def heart_disease_prediction(input_data):
    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # load the saved model
    loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0]== 0):
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease' 