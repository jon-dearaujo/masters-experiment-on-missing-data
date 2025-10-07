RANDOMNESS=10%
CTGAN TRAINING EPOCHS=1000

SDMetrics Quality Report for Complete Data:0.8502037400815139
SDMetrics Quality Report for Incomplete Data:0.7821975565751271

--- Results for Benchmark Model (Real Data) ---
Accuracy: 0.9645

                     precision    recall  f1-score   support

Insufficient_Weight       0.98      0.94      0.96        54
      Normal_Weight       0.88      0.97      0.92        58
     Obesity_Type_I       1.00      0.97      0.99        70
    Obesity_Type_II       0.98      1.00      0.99        60
   Obesity_Type_III       1.00      0.98      0.99        65
 Overweight_Level_I       0.95      0.90      0.92        58
Overweight_Level_II       0.97      0.98      0.97        58

           accuracy                           0.96       423
          macro avg       0.96      0.96      0.96       423
       weighted avg       0.97      0.96      0.96       423
------------------------------------------------------------



--- Results for Model A (Synthetic - Complete) ---
Accuracy: 0.4421

                     precision    recall  f1-score   support

Insufficient_Weight       0.52      0.48      0.50        54
      Normal_Weight       0.33      0.16      0.21        58
     Obesity_Type_I       0.12      0.04      0.06        70
    Obesity_Type_II       0.71      0.53      0.61        60
   Obesity_Type_III       0.91      0.98      0.95        65
 Overweight_Level_I       0.20      0.31      0.24        58
Overweight_Level_II       0.30      0.60      0.40        58

           accuracy                           0.44       423
          macro avg       0.44      0.44      0.43       423
       weighted avg       0.44      0.44      0.42       423
------------------------------------------------------------



--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.2837

                     precision    recall  f1-score   support

Insufficient_Weight       0.36      0.07      0.12        54
      Normal_Weight       0.21      0.71      0.32        58
     Obesity_Type_I       0.32      0.34      0.33        70
    Obesity_Type_II       0.41      0.53      0.46        60
   Obesity_Type_III       0.33      0.05      0.08        65
 Overweight_Level_I       0.26      0.14      0.18        58
Overweight_Level_II       0.38      0.14      0.20        58

           accuracy                           0.28       423
          macro avg       0.32      0.28      0.24       423
       weighted avg       0.32      0.28      0.24       423
------------------------------------------------------------



