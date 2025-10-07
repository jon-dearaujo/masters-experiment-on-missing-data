RANDOMNESS=20%
CTGAN TRAINING EPOCHS=2000

SDMetrics Quality Report for Complete Data:0.8732198988411297
SDMetrics Quality Report for Incomplete Data:0.7663622221451116

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
Accuracy: 0.4870

                     precision    recall  f1-score   support

Insufficient_Weight       0.64      0.59      0.62        54
      Normal_Weight       0.50      0.79      0.61        58
     Obesity_Type_I       0.27      0.37      0.31        70
    Obesity_Type_II       0.67      0.30      0.41        60
   Obesity_Type_III       0.94      0.98      0.96        65
 Overweight_Level_I       0.25      0.24      0.25        58
Overweight_Level_II       0.18      0.10      0.13        58

           accuracy                           0.49       423
          macro avg       0.49      0.48      0.47       423
       weighted avg       0.49      0.49      0.47       423
------------------------------------------------------------



--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.4113

                     precision    recall  f1-score   support

Insufficient_Weight       0.67      0.33      0.44        54
      Normal_Weight       0.27      0.71      0.39        58
     Obesity_Type_I       0.36      0.31      0.34        70
    Obesity_Type_II       0.51      0.70      0.59        60
   Obesity_Type_III       0.85      0.52      0.65        65
 Overweight_Level_I       0.42      0.09      0.14        58
Overweight_Level_II       0.26      0.21      0.23        58

           accuracy                           0.41       423
          macro avg       0.48      0.41      0.40       423
       weighted avg       0.48      0.41      0.40       423
------------------------------------------------------------



