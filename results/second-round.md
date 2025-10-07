RANDOMNESS=20%
CTGAN TRAINING EPOCHS=1000

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
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000386 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2070
[LightGBM] [Info] Number of data points in the train set: 1688, number of used features: 16
[LightGBM] [Info] Start training from score -1.829181
[LightGBM] [Info] Start training from score -1.452414
[LightGBM] [Info] Start training from score -1.917871
[LightGBM] [Info] Start training from score -2.179026
[LightGBM] [Info] Start training from score -2.278008
[LightGBM] [Info] Start training from score -2.103424
[LightGBM] [Info] Start training from score -2.113180
--- Results for Model A (Synthetic - Complete) ---
Accuracy: 0.4539
                     precision    recall  f1-score   support

Insufficient_Weight       0.48      0.57      0.53        54
      Normal_Weight       0.40      0.50      0.45        58
     Obesity_Type_I       0.39      0.44      0.42        70
    Obesity_Type_II       0.49      0.65      0.56        60
   Obesity_Type_III       0.94      0.48      0.63        65
 Overweight_Level_I       0.38      0.31      0.34        58
Overweight_Level_II       0.27      0.22      0.25        58

           accuracy                           0.45       423
          macro avg       0.48      0.45      0.45       423
       weighted avg       0.48      0.45      0.45       423

------------------------------------------------------------

--- Training Model B (Synthetic - Incomplete) ---
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000382 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2071
[LightGBM] [Info] Number of data points in the train set: 1688, number of used features: 16
[LightGBM] [Info] Start training from score -1.821828
[LightGBM] [Info] Start training from score -2.024128
[LightGBM] [Info] Start training from score -1.789393
[LightGBM] [Info] Start training from score -1.913847
[LightGBM] [Info] Start training from score -2.420664
[LightGBM] [Info] Start training from score -2.006350
[LightGBM] [Info] Start training from score -1.782325
--- Results for Model B (Synthetic - Incomplete) ---
Accuracy: 0.2317
                     precision    recall  f1-score   support

Insufficient_Weight       0.21      0.46      0.29        54
      Normal_Weight       0.18      0.09      0.12        58
     Obesity_Type_I       0.19      0.21      0.20        70
    Obesity_Type_II       0.16      0.17      0.16        60
   Obesity_Type_III       0.57      0.49      0.53        65
 Overweight_Level_I       0.13      0.07      0.09        58
Overweight_Level_II       0.14      0.12      0.13        58

           accuracy                           0.23       423
          macro avg       0.23      0.23      0.22       423
       weighted avg       0.23      0.23      0.22       423

------------------------------------------------------------