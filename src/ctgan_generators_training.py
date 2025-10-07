import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from utils import args as args_parser, results_collector

args = args_parser.parse_args()
print(f"Running Generators Training for: epochs={args.epochs}, randomness={args.randomness}")
collector = results_collector.ResultsCollector(args.epochs, args.randomness)

collector.collect([
    f"RANDOMNESS={args.randomness}%\n", ""
    f"CTGAN TRAINING EPOCHS={args.epochs}\n",
    "\n"
])


try:
    print(f"Reading from '../ObesityDataSet_raw_and_data_sinthetic.csv'...")
    data_full = pd.read_csv('../ObesityDataSet_raw_and_data_sinthetic.csv')
    print("CSV file loaded successfully. Shape:", data_full.shape)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise e

target_column = 'NObeyesdad'
X = data_full.drop(columns=[target_column])
y = data_full[target_column]

D_train, D_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

D_train = pd.concat([D_train, y_train], axis=1)
D_test = pd.concat([D_test, y_test], axis=1)

print(f"\nShape of D_train_complete: {D_train.shape}")
print(f"Shape of D_test: {D_test.shape}")

D_train.to_csv(collector.append_to_dir_name('D_train_complete.csv'), index=False)
D_test.to_csv(collector.append_to_dir_name('D_test.csv'), index=False)
print("D_train_complete and D_test saved as 'D_train_complete.csv' and 'D_test.csv' respectively.")

# Simulate missing data in D_train_incomplete
D_train_incomplete = D_train.copy()
columns_to_make_incomplete = D_train.columns[:-1]  # Exclude target column
missing_rate = args.randomness / 100.0

for col in columns_to_make_incomplete:
    D_train_incomplete.loc[
        D_train_incomplete.sample(
            frac=missing_rate, random_state=42
        ).index, col
    ] = np.nan
    print(f"Introduced {int(len(D_train_incomplete) * missing_rate)} missing values in column '{col}'")


print("\n--- Verification ---")
print("Number of missing values in each dataset:\n")
print("Original Training Set (D_train_complete):")
print(D_train.isnull().sum().sum()) # Should be 0

print("\nIncomplete Training Set (D_train_incomplete):")
print(D_train_incomplete[columns_to_make_incomplete].isnull().sum())

D_train_incomplete.to_csv(collector.append_to_dir_name('D_train_incomplete.csv'), index=False)

print("Training CTGAN model on D_train_complete")
d_train_metadata = SingleTableMetadata()
d_train_metadata.detect_from_dataframe(data=D_train)
model_complete = CTGANSynthesizer(
    metadata=d_train_metadata,
    epochs=args.epochs, verbose=True
)

model_complete.fit(D_train)
model_complete.save(collector.append_to_dir_name('ctgan_model_complete.pkl'))
print("Model trained and saved as 'ctgan_model_complete.pkl'")
print('*' * 50)

print("Training CTGAN model on D_train_incomplete")
model_incomplete = CTGANSynthesizer(
    metadata=d_train_metadata,
    epochs=args.epochs, verbose=True
)
model_incomplete.fit(D_train_incomplete)
model_incomplete.save(collector.append_to_dir_name('ctgan_model_incomplete.pkl'))
print("Model trained and saved as 'ctgan_model_incomplete.pkl'")
print('*' * 50)