#Dataset - https://www.kaggle.com/datasets/shantanugarg274/heart-prediction-dataset-quantum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Load dataset
df = pd.read_csv('Heart Prediction Quantum Dataset.csv')
X = df.drop('HeartDisease', axis=1)
y = np.array(df[['HeartDisease']]).flatten()

#Print dataset shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

#Exploratory data analysis
print('\nMissing values:')
print(df.isnull().sum())
print('\nClass balance:')
print(df['HeartDisease'].value_counts())

#Class balance plot
plt.figure()
df['HeartDisease'].value_counts().sort_index().plot(kind='bar')
plt.title('Class Balance - Heart Disease')
plt.xlabel('HeartDisease')
plt.ylabel('Count')
plt.savefig('images/class_balance.png')

#Correlation heatmap
corr = df.corr()
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')

#Split the dataset into training and testing;  80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale our features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#5-fold cross validation, used to tune every model using training data only
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lbl = ['Not present', 'Present']
results = {}

#Begin SVC
print('\nBegin SVC:')

#Search kernels, C and gamma
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['poly', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1, 1]}
]
grid = GridSearchCV(SVC(), param_grid, cv=kf)
grid.fit(X_train, y_train)
print(f'Best parameters: {grid.best_params_}')
print(f'Best CV accuracy: {grid.best_score_:.2f}\n')

#The final SVC model is the best model from the grid search
clf = grid.best_estimator_

print(f'SVC Model Score: {clf.score(X_train, y_train)}')
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'SVC Accuracy score: {accuracy}')
print(f'SVC Recall score: {recall}\n')
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print(f'SVC missed {confusionMatrix[1][0]} cases of heart disease (false negatives)\n')
results['SVC'] = [grid.best_score_, accuracy, recall]

print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('SVC - Heart Disease')
plt.savefig('images/svc_confusion_matrix.png')

#Begin logistic regression
print('\nBegin Logistic Regression: ')
grid_lr = GridSearchCV(LogisticRegression(), {'C': [0.01, 0.1, 1, 10, 100]}, cv=kf)
grid_lr.fit(X_train, y_train)
print(f'Best parameters: {grid_lr.best_params_}')
print(f'Best CV accuracy: {grid_lr.best_score_:.2f}\n')
model = grid_lr.best_estimator_

#Y-intercept and coefs
print(f'Y - intercept: {model.intercept_}')
print('Coefs:')
print(pd.Series(model.coef_[0], index=X.columns))
print()

#Logistic Regression scores
print(f'Logistic Regression Model Score: {model.score(X_train, y_train)}')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Logistic Regression Accuracy score: {accuracy}')
print(f'Logistic Regression Recall score: {recall}\n')
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print(f'Logistic Regression missed {confusionMatrix[1][0]} cases of heart disease (false negatives)\n')
results['Logistic Regression'] = [grid_lr.best_score_, accuracy, recall]

print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('Logistic Regression - Heart Disease')
plt.savefig('images/logreg_confusion_matrix.png')

#Begin random forest
print('\nBegin Random Forest: ')
rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 3, 5, 10]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=kf)
grid_rf.fit(X_train, y_train)
print(f'Best parameters: {grid_rf.best_params_}')
print(f'Best CV accuracy: {grid_rf.best_score_:.2f}\n')
rf = grid_rf.best_estimator_

#Random Forest scores
print(f'Random Forest Model Score: {rf.score(X_train, y_train)}')
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Random Forest Accuracy score: {accuracy}')
print(f'Random Forest Recall score: {recall}\n')
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print(f'Random Forest missed {confusionMatrix[1][0]} cases of heart disease (false negatives)\n')
results['Random Forest'] = [grid_rf.best_score_, accuracy, recall]

print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('Random Forest - Heart Disease')
plt.savefig('images/rf_confusion_matrix.png')

#Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
print('Random Forest feature importances:')
print(importances.sort_values(ascending=False))
plt.figure()
importances.plot(kind='barh')
plt.title('Random Forest - Feature Importances')
plt.tight_layout()
plt.savefig('images/rf_feature_importances.png')

#ROC curves and AUC for all three models
roc_scores = {
    'SVC': clf.decision_function(X_test),
    'Logistic Regression': model.predict_proba(X_test)[:, 1],
    'Random Forest': rf.predict_proba(X_test)[:, 1]
}
plt.figure()
for name, score in roc_scores.items():
    fpr, tpr, thresholds = roc_curve(y_test, score)
    auc = roc_auc_score(y_test, score)
    results[name].append(auc)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Heart Disease')
plt.legend()
plt.savefig('images/roc_curves.png')

#Compare all models
summary = pd.DataFrame(results, index=['CV Accuracy', 'Test Accuracy', 'Recall', 'AUC']).T
print('\nModel comparison:')
print(summary.round(2))
plt.show()
