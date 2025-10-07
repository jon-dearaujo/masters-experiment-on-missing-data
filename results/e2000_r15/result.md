RANDOMNESS=15%
CTGAN TRAINING EPOCHS=2000

SDMetrics Quality Report for Complete Data:0.8625821444335646
SDMetrics Quality Report for Incomplete Data:0.7816891273623026

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
Accuracy: 0.5083

                     precision    recall  f1-score   support

Insufficient_Weight       0.87      0.48      0.62        54
      Normal_Weight       0.59      0.69      0.63        58
     Obesity_Type_I       0.32      0.51      0.40        70
    Obesity_Type_II       0.59      0.40      0.48        60
   Obesity_Type_III       1.00      0.83      0.91        65
 Overweight_Level_I       0.41      0.33      0.37        58
Overweight_Level_II       0.22      0.28      0.25        58

           accuracy                           0.51       423
          macro avg       0.57      0.50      0.52       423
       weighted avg       0.57      0.51      0.52       423
------------------------------------------------------------



--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.3830

                     precision    recall  f1-score   support

Insufficient_Weight       0.43      0.52      0.47        54
      Normal_Weight       0.35      0.26      0.30        58
     Obesity_Type_I       0.25      0.34      0.29        70
    Obesity_Type_II       0.47      0.27      0.34        60
   Obesity_Type_III       0.92      0.52      0.67        65
 Overweight_Level_I       0.38      0.45      0.41        58
Overweight_Level_II       0.24      0.33      0.28        58

           accuracy                           0.38       423
          macro avg       0.43      0.38      0.39       423
       weighted avg       0.44      0.38      0.39       423
------------------------------------------------------------



