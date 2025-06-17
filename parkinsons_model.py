import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import pickle

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('Project 14 Parkinsons Disease Data.csv')

# separating the data and labels
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

# Splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training - Random Forest Classifier
classifier = RandomForestClassifier(random_state=2, max_depth=10, min_samples_leaf=5)
classifier.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of training data : ', training_data_accuracy)
print('Accuracy score of test data : ', test_data_accuracy)
print('F1-score of test data : ', f1_score(Y_test, X_test_prediction))

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
filename = 'parkinsons_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Save Y_test, X_test_prediction, feature importances, and ROC data
pickle.dump(Y_test, open('parkinsons_Y_test.pkl', 'wb'))
pickle.dump(X_test_prediction, open('parkinsons_X_test_prediction.pkl', 'wb'))
pickle.dump(classifier.feature_importances_, open('parkinsons_feature_importances.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('parkinsons_feature_names.pkl', 'wb'))
pickle.dump({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, open('parkinsons_roc_curve.pkl', 'wb')) # Save ROC data

# Creating a prediction function
def parkinsons_prediction(input_data):
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # load the saved model
    loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return "The Person does not have Parkinsons Disease"
    else:
        return "The Person has Parkinsons Disease" 