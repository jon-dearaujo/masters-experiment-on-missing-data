# import subprocess
# import sys

script_params_to_run_in_parallel = [
    (1000, 10),
    (1000, 15),
    (1000, 20),
    (2000, 10),
    (2000, 15),
    (2000, 20),
]
# processes = []

# print("🚀 Launching all processes in parallel...")
# # Start all processes without waiting
# for script_params in script_params_to_run_in_parallel:
#     params = f"--epochs={script_params[0]} --randomness={script_params[1]}"
#     command = [
#         sys.executable,
#         f"ctgan_generators_training.py {params};",
#         f"ctgan_synthetic_generation.py {params};",
#         f"lightgbm_evaluation.py {params}",
#     ]
#     # Popen starts the process and immediately moves on
#     process = subprocess.Popen(command)
#     processes.append(process)

# print("⏳ All processes are running. Now waiting for them to complete...")
# # Wait for all launched processes to finish
# for p in processes:
#     p.wait()

# print("✅ All parallel processes have finished.")

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from utils.results_collector import ResultsCollector

EXPERIMENTS = [{
        "name": f"Experiment 1: {params[0]} Epochs, {params[1]} Randomness",
        "params": {"--epochs": params[0], "--randomness": params[1]},
        "scripts": [
            "ctgan_generators_training.py",
            "ctgan_synthetic_generation.py",
            "lightgbm_evaluation.py"
        ]
    } for params in script_params_to_run_in_parallel
]

def run_experiment_set(experiment):
    """Worker function to run one full, sequential set of scripts."""
    name = experiment['name']
    params = experiment['params']
    scripts = experiment['scripts']

    results_collector = ResultsCollector(params['--epochs'], params['--randomness'])
    os.makedirs(results_collector.dir_name, exist_ok=True)

    print(f"▶️ Starting Set: {name}")

    # This inner loop is the sequential part
    for script_name in scripts:
        param_list = [item for k, v in params.items() for item in (k, str(v))]
        command = [sys.executable, script_name] + param_list
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # In case of an error, return the name and the error for reporting
            return f"❌ FAILED: {name} on script {script_name}\nError: {e.stderr}"

    return f"✅ SUCCESS: {name}"


def main():
    print(f"🚀 Launching {len(EXPERIMENTS)} experiment sets in parallel...")

    # Use a process pool to run each set in a separate process
    with ProcessPoolExecutor() as executor:
        # executor.map applies the worker function to each item in EXPERIMENTS
        results = executor.map(run_experiment_set, EXPERIMENTS)

        print("\n--- Results ---")
        for result in results:
            print(result)

    print("\n🎉 All parallel sets have finished.")


if __name__ == "__main__":
    main()