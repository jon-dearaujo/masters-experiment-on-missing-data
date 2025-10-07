RANDOMNESS=20%
CTGAN TRAINING EPOCHS=1000

SDMetrics Quality Report for Complete Data:0.8465536094705488
SDMetrics Quality Report for Incomplete Data:0.7568385835032827

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
Accuracy: 0.4634

                     precision    recall  f1-score   support

Insufficient_Weight       0.64      0.17      0.26        54
      Normal_Weight       0.45      0.57      0.50        58
     Obesity_Type_I       0.27      0.44      0.34        70
    Obesity_Type_II       0.68      0.60      0.64        60
   Obesity_Type_III       0.98      0.78      0.87        65
 Overweight_Level_I       0.35      0.48      0.41        58
Overweight_Level_II       0.22      0.14      0.17        58

           accuracy                           0.46       423
          macro avg       0.51      0.45      0.46       423
       weighted avg       0.51      0.46      0.46       423
------------------------------------------------------------



--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.3546

                     precision    recall  f1-score   support

Insufficient_Weight       0.34      0.69      0.45        54
      Normal_Weight       0.28      0.16      0.20        58
     Obesity_Type_I       0.16      0.16      0.16        70
    Obesity_Type_II       0.40      0.57      0.47        60
   Obesity_Type_III       0.81      0.66      0.73        65
 Overweight_Level_I       0.17      0.16      0.16        58
Overweight_Level_II       0.28      0.12      0.17        58

           accuracy                           0.35       423
          macro avg       0.35      0.36      0.34       423
       weighted avg       0.35      0.35      0.34       423
------------------------------------------------------------



