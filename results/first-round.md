RANDOMNESS=20%
CTGAN TRAINING EPOCHS=300

------LGBM RESULTS------
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

--- Training Model A (Synthetic - Complete) ---
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000444 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2070
[LightGBM] [Info] Number of data points in the train set: 1688, number of used features: 16
[LightGBM] [Info] Start training from score -2.278008
[LightGBM] [Info] Start training from score -1.427413
[LightGBM] [Info] Start training from score -1.653647
[LightGBM] [Info] Start training from score -2.158300
[LightGBM] [Info] Start training from score -2.143033
[LightGBM] [Info] Start training from score -1.870618
[LightGBM] [Info] Start training from score -2.533460
--- Results for Model A (Synthetic - Complete) ---
Accuracy: 0.1631
                     precision    recall  f1-score   support

Insufficient_Weight       0.20      0.15      0.17        54
      Normal_Weight       0.17      0.45      0.25        58
     Obesity_Type_I       0.17      0.30      0.22        70
    Obesity_Type_II       0.19      0.13      0.16        60
   Obesity_Type_III       0.03      0.02      0.02        65
 Overweight_Level_I       0.14      0.05      0.08        58
Overweight_Level_II       0.22      0.03      0.06        58

           accuracy                           0.16       423
          macro avg       0.16      0.16      0.14       423
       weighted avg       0.16      0.16      0.14       423

------------------------------------------------------------

--- Training Model B (Synthetic - Incomplete) ---
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000394 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2071
[LightGBM] [Info] Number of data points in the train set: 1688, number of used features: 16
[LightGBM] [Info] Start training from score -1.579097
[LightGBM] [Info] Start training from score -1.859146
[LightGBM] [Info] Start training from score -1.789393
[LightGBM] [Info] Start training from score -2.343703
[LightGBM] [Info] Start training from score -2.319312
[LightGBM] [Info] Start training from score -1.988882
[LightGBM] [Info] Start training from score -1.967468
--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.1135
                     precision    recall  f1-score   support

Insufficient_Weight       0.09      0.20      0.12        54
      Normal_Weight       0.07      0.16      0.09        58
     Obesity_Type_I       0.15      0.07      0.10        70
    Obesity_Type_II       0.50      0.02      0.03        60
   Obesity_Type_III       0.00      0.00      0.00        65
 Overweight_Level_I       0.22      0.21      0.21        58
Overweight_Level_II       0.20      0.17      0.19        58

           accuracy                           0.11       423
          macro avg       0.17      0.12      0.11       423
       weighted avg       0.17      0.11      0.10       423

------------------------------------------------------------