# Heart Disease Prediction

Binary classification of heart disease using **SVM**, **Logistic Regression** and **Random Forest** in scikit-learn. Each model is tuned with cross-validated grid search and compared on a held-out test set using accuracy, recall and ROC/AUC.

## Dataset

[Heart Prediction Quantum Dataset](https://www.kaggle.com/datasets/shantanugarg274/heart-prediction-dataset-quantum) (Kaggle)

- **Rows:** 500 (300 with heart disease, 200 without), no missing values
- **Features:** Age, Gender, BloodPressure, Cholesterol, HeartRate, QuantumPatternFeature
- **Target:** `HeartDisease` (0 = not present, 1 = present)

> The dataset is synthetic, so results show the modeling workflow rather than clinical performance.

X shape: (500, 6)
y shape: (500,)

Missing values:
Age                      0
Gender                   0
BloodPressure            0
Cholesterol              0
HeartRate                0
QuantumPatternFeature    0
HeartDisease             0
dtype: int64

Class balance:
HeartDisease
1    300
0    200
Name: count, dtype: int64

Begin SVC:
Best parameters: {'C': 10, 'kernel': 'linear'}
Best CV accuracy: 0.93

SVC Model Score: 0.925
SVC Accuracy score: 0.94
SVC Recall score: 0.9

[[40  0]
 [ 6 54]]
SVC missed 6 cases of heart disease (false negatives)

              precision    recall  f1-score   support

           0       0.87      1.00      0.93        40
           1       1.00      0.90      0.95        60

    accuracy                           0.94       100
   macro avg       0.93      0.95      0.94       100
weighted avg       0.95      0.94      0.94       100


Begin Logistic Regression:
Best parameters: {'C': 1}
Best CV accuracy: 0.92

Y - intercept: [1.35902001]
Coefs:
Age                     -0.014233
Gender                  -0.018662
BloodPressure           -0.233857
Cholesterol             -0.015758
HeartRate               -0.124645
QuantumPatternFeature   -4.827493
dtype: float64

Logistic Regression Model Score: 0.93
Logistic Regression Accuracy score: 0.95
Logistic Regression Recall score: 0.9166666666666666

[[40  0]
 [ 5 55]]
Logistic Regression missed 5 cases of heart disease (false negatives)

              precision    recall  f1-score   support

           0       0.89      1.00      0.94        40
           1       1.00      0.92      0.96        60

    accuracy                           0.95       100
   macro avg       0.94      0.96      0.95       100
weighted avg       0.96      0.95      0.95       100


Begin Random Forest:
Best parameters: {'max_depth': 5, 'n_estimators': 100}
Best CV accuracy: 0.93

Random Forest Model Score: 0.9625
Random Forest Accuracy score: 0.93
Random Forest Recall score: 0.8833333333333333

[[40  0]
 [ 7 53]]
Random Forest missed 7 cases of heart disease (false negatives)

              precision    recall  f1-score   support

           0       0.85      1.00      0.92        40
           1       1.00      0.88      0.94        60

    accuracy                           0.93       100
   macro avg       0.93      0.94      0.93       100
weighted avg       0.94      0.93      0.93       100

Random Forest feature importances:
QuantumPatternFeature    0.681094
Cholesterol              0.143714
Age                      0.092260
BloodPressure            0.044425
HeartRate                0.033291
Gender                   0.005216
dtype: float64

Model comparison:
                     CV Accuracy  Test Accuracy  Recall   AUC
SVC                         0.93           0.94    0.90  0.99
Logistic Regression         0.92           0.95    0.92  0.99
Random Forest               0.93           0.93    0.88  0.99

### Takeaways

- **The models perform about the same.** Test accuracy ranges from 0.93 to 0.95 and all three have an AUC of 0.99. With 100 test samples, one prediction is one percentage point, so the differences are within noise. A linear SVC doing as well as a Random Forest suggests the classes are close to linearly separable.
- **Recall matters more than accuracy here.** Every model had perfect precision on the "Present" class and no false positives, but missed 5 to 7 of the 60 real cases (recall of 0.88 to 0.92). In a medical screening setting, a missed case is usually costlier than a false alarm, so a lower decision threshold that trades some precision for recall would be worth trying.
- **One feature does most of the work.** `QuantumPatternFeature` has by far the largest Logistic Regression coefficient (-4.83, versus 0.23 or less for the others) and a Random Forest importance of 0.68. Together with the near-perfect AUC, this suggests the feature was built to separate the classes, which is another reason to treat the results as a demonstration on synthetic data.
- **The baseline is 0.60.** About 60% of samples are positive, so always predicting "Present" would score about 0.60.

## How to Run

Download the CSV from the Kaggle link above, place it in the project folder, then run:

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib

## Possible Next Steps

- Lower the decision threshold and plot the precision/recall trade-off
- Re-run without `QuantumPatternFeature` to see how the models do on the remaining features
- Add a gradient boosting model to the comparison
