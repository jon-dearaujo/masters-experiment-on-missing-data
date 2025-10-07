RANDOMNESS=15%
CTGAN TRAINING EPOCHS=1000

SDMetrics Quality Report for Complete Data:0.8496452520520574
SDMetrics Quality Report for Incomplete Data:0.7495280843914571

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
Accuracy: 0.4752

                     precision    recall  f1-score   support

Insufficient_Weight       0.83      0.35      0.49        54
      Normal_Weight       0.42      0.22      0.29        58
     Obesity_Type_I       0.53      0.11      0.19        70
    Obesity_Type_II       0.63      0.65      0.64        60
   Obesity_Type_III       0.88      0.94      0.91        65
 Overweight_Level_I       0.25      0.45      0.33        58
Overweight_Level_II       0.29      0.60      0.39        58

           accuracy                           0.48       423
          macro avg       0.55      0.48      0.46       423
       weighted avg       0.55      0.48      0.46       423
------------------------------------------------------------



--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.1868

                     precision    recall  f1-score   support

Insufficient_Weight       0.22      0.09      0.13        54
      Normal_Weight       0.19      0.88      0.31        58
     Obesity_Type_I       0.23      0.04      0.07        70
    Obesity_Type_II       0.65      0.22      0.33        60
   Obesity_Type_III       0.00      0.00      0.00        65
 Overweight_Level_I       0.00      0.00      0.00        58
Overweight_Level_II       0.09      0.12      0.10        58

           accuracy                           0.19       423
          macro avg       0.20      0.19      0.13       423
       weighted avg       0.20      0.19      0.13       423
------------------------------------------------------------



