import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from utils import args as args_parser, results_collector

args = args_parser.parse_args()
print(f"Running CTGAN Synthetic Generation for: epochs={args.epochs}, randomness={args.randomness}")
collector = results_collector.ResultsCollector(args.epochs, args.randomness)
# --- 1. Load All Datasets ---
print("Loading datasets...")

# Load the real dataframes (assuming they are in memory or saved as CSV)
D_train_complete = pd.read_csv(collector.append_to_dir_name("D_train_complete.csv")) # Example if loading from file
D_test = pd.read_csv(collector.append_to_dir_name("D_test.csv")) # Example if loading from file

# Load the synthetic datasets
S_complete = pd.read_csv(collector.append_to_dir_name("synthetic_data_complete.csv"))
S_incomplete = pd.read_csv(collector.append_to_dir_name("synthetic_data_incomplete.csv"))

target_column = 'NObeyesdad'

# --- 2. Data Preparation ---
# LightGBM requires all categorical features to be converted to the 'category' dtype
# and the target variable to be label encoded (converted to integers).

def prepare_data(dataset):
    """Prepares a dataframe for LightGBM training and evaluation."""
    # Separate features and target
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Convert all object columns to the 'category' dtype for LGBM
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    # Use a LabelEncoder for the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

# Prepare all datasets
X_train_real, y_train_real, label_encoder = prepare_data(D_train_complete)
X_test_real, y_test_real, _ = prepare_data(D_test) # Use the same encoder

X_train_synth_comp, y_train_synth_comp, _ = prepare_data(S_complete)
X_train_synth_incomp, y_train_synth_incomp, _ = prepare_data(S_incomplete)

# --- 3. Model Training and Evaluation Function ---
def train_and_evaluate(X_train, y_train, X_test, y_test, model_name, label_encoder):
    """Trains an LGBM model and prints its evaluation report."""
    print(f"--- Training {model_name} ---")

    # Initialize and train the LightGBM Classifier
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the REAL test set
    y_pred = model.predict(X_test)

    # Decode the numeric predictions back to original labels for the report
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Calculate accuracy and print the report
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels)

    result = [f"--- Results for {model_name} ---\n"
        f"Accuracy: {accuracy:.4f}\n\n",
        report,
        "-" * 60 + "\n\n",
        "\n\n"
    ]
    collector.collect(result)

    return accuracy

# --- 4. Run the Experiment ---

# A) Benchmark Model: Trained on REAL data
train_and_evaluate(
    X_train_real, y_train_real, X_test_real, y_test_real,
    "Benchmark Model (Real Data)", label_encoder
)

# B) Model A: Trained on synthetic data from COMPLETE source
train_and_evaluate(
    X_train_synth_comp, y_train_synth_comp, X_test_real, y_test_real,
    "Model A (Synthetic - Complete)", label_encoder
)

# C) Model B: Trained on synthetic data from INCOMPLETE source
train_and_evaluate(
    X_train_synth_incomp, y_train_synth_incomp, X_test_real, y_test_real,
    "Model B (Synthetic - Incomplete)", label_encoder
)