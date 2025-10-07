import pandas as pd
from sdv.utils  import load_synthesizer
from sdv.evaluation.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

from utils import args as args_parser, results_collector

args = args_parser.parse_args()
print(f"Running CTGAN Synthetic Generation for: epochs={args.epochs}, randomness={args.randomness}")
collector = results_collector.ResultsCollector(args.epochs, args.randomness)

print("Loading trained models...")
model_complete = load_synthesizer(collector.append_to_dir_name('ctgan_model_complete.pkl'))
model_incomplete = load_synthesizer(collector.append_to_dir_name('ctgan_model_incomplete.pkl'))
print("Models loaded successfully.")

try:
    data_train_complete = pd.read_csv(collector.append_to_dir_name('D_train_complete.csv'))
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise e

num_rows_to_generate = len(data_train_complete)
print(f"Generating {int(num_rows_to_generate)} synthetic samples using both models...")
samples_complete = model_complete.sample(num_rows_to_generate)
samples_incomplete = model_incomplete.sample(num_rows_to_generate)
samples_complete.to_csv(collector.append_to_dir_name('synthetic_data_complete.csv'), index=False)
samples_incomplete.to_csv(collector.append_to_dir_name('synthetic_data_incomplete.csv'), index=False)
print("Synthetic data generated and saved as 'synthetic_data_complete.csv' and 'synthetic_data_incomplete.csv'.")

print("Getting SDM Metrics score out of complete and incomplete synthetic data...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data_train_complete)
report_complete = QualityReport()
report_complete.generate(data_train_complete, samples_complete, metadata=metadata.to_dict())
collector.collect(["SDMetrics Quality Report for Complete Data:", str(report_complete.get_score()), "\n"])

report_incomplete = QualityReport()
report_incomplete.generate(data_train_complete, samples_incomplete, metadata=metadata.to_dict())
collector.collect(["SDMetrics Quality Report for Incomplete Data:", str(report_incomplete.get_score()), "\n\n"])

print("*" * 50)


