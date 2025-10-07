RANDOMNESS=10%
CTGAN TRAINING EPOCHS=2000

SDMetrics Quality Report for Complete Data:0.861332326944334
SDMetrics Quality Report for Incomplete Data:0.7911990836496318

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
Accuracy: 0.5508

                     precision    recall  f1-score   support

Insufficient_Weight       0.73      0.59      0.65        54
      Normal_Weight       0.59      0.74      0.66        58
     Obesity_Type_I       0.37      0.56      0.45        70
    Obesity_Type_II       0.73      0.32      0.44        60
   Obesity_Type_III       0.95      0.92      0.94        65
 Overweight_Level_I       0.29      0.28      0.28        58
Overweight_Level_II       0.42      0.41      0.42        58

           accuracy                           0.55       423
          macro avg       0.58      0.55      0.55       423
       weighted avg       0.58      0.55      0.55       423
------------------------------------------------------------



--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.5083

                     precision    recall  f1-score   support

Insufficient_Weight       0.79      0.50      0.61        54
      Normal_Weight       0.50      0.81      0.62        58
     Obesity_Type_I       0.25      0.04      0.07        70
    Obesity_Type_II       0.79      0.43      0.56        60
   Obesity_Type_III       0.88      0.92      0.90        65
 Overweight_Level_I       0.36      0.47      0.40        58
Overweight_Level_II       0.24      0.43      0.30        58

           accuracy                           0.51       423
          macro avg       0.54      0.52      0.50       423
       weighted avg       0.54      0.51      0.49       423
------------------------------------------------------------



