## What Is The Idea Behind This Experiment?

Measure what is the effect of missinginess on an input dataset by using the downstream classification accuracy decay as proxy. We assume that "complete" data would yield better result than incomplete data, and then start measuring the impact of taking data points upon the downstream accuracy.

## Step By Step Process

- Start from a full dataset: src/ObesityDataSet_raw_and_data_synthetic.csv.
- Progressively take pieces of data out of it: from 10% to 40%.
- Train a CTGAN on the incomplete dataset
- Sample synthetic data using the trained CTGAN
- Train a downstream classifier on the synthetic data :LightGBM, XGBoost & Catboost
- Iteratively measure the decay in the accuracy of these models as we take more and more data out of the original dataset.

## How Does This Experiment Relate To Generative AI?

GEN AI is used as the engine to go from bad missing data in the original dataset to a full synthetic dataset, so what we are checking in reality is: "Is there any capability in the generative ai (ctgan specifically) to recover from missing datapoints?", even though indirectly.

One way to figure this, is by comparing the downstream accuracy using original untouched dataset vs the synthetic ctgan sampled datasets after taking out some pieces of data, and this is kinda of what is done here:
1. We trained & tested the downstream lightgbm/xgboost/catboost against original dataset
1. We iteratively removed pieces of data, trained CTGAN and sampled, then trained & tested downstream lightgbm/xgboost/catboost against this data.

## What Am I Trying To Answer?

At the end, what is really in question here is:
***How does "bad data" affect the ability of CTGAN (geneartive ai) to understand the inner structure of a dataset?*** This is answered by giving CTGAN bad data, let it train on it, sample new data, and see how this data affects the downstream classifier model, compared to CTGAN's data from a compelte model.