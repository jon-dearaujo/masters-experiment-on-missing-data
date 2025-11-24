import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from utils.results_collector import ResultsCollector

script_params_to_run_in_parallel = [
    (1000, 10),
    (1000, 15),
    (1000, 20),
    (2000, 10),
    (2000, 15),
    (2000, 20),
]

EXPERIMENTS = [{
        "name": f"Experiment 1: {epochs} Epochs, {randomness} Randomness",
        "params": {"--epochs": epochs, "--randomness": randomness},
        "scripts": [
            "ctgan_generators_training.py",
            "ctgan_synthetic_generation.py",
            "lightgbm_evaluation.py"
        ]
    } for [epochs, randomness] in script_params_to_run_in_parallel
]

'''
explicar qualidade dos dados
e qualidade dos modelos

[caso o CESAR permita um trabalho em formato de paper em vez da dissertação]
-- template ieee, coluna dupla.

-- Executar Kolmogorov-Smirnov [KS] https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
-- simulação deve rodar 30x com 1000 épocas e 30x com 2000, extrair médias.

Há uma melhora? não eh possível comprovar com 95% de confiança?
 De fato, dados os resultados de Kolmogorov-Smirnov, a melhora preliminar se mantém?
o que é a melhora?
"objetivo -- substituir teoria por inferência causal -- aumento de épocas cria melhora substancial dados os resultados de KS test"

REALIZAR revisão sistematica

DOIS INDICADORES
1. qualidade do sdmetrics
2. melhora dos dados sintéticos

2 tipos de gráficos para cada seção:
gráfico de linha - media da qualidade época/época
precisa normalizar 1000 com 2000

boxplot - distribuição da métrica


seção que explica synthetic data generation: VARs and GANs
  - what are GANs?
    - add images
    - go deep on CTGAN

- tell that researches used  Lightgbm for evaluation to justify why?

dor oportunidade e qual minha proposta.

add artigos no notebooklm e pergunta: insights e gaps.

--- próximos passos ---

1. gerar gráficos e melhora analise de resultados. Após isso, experimento concluído.
2. concluir metodologia.
3. concluir conclusão.
4. finalizar resumo e introdução.
5. finalizar revisão sistemática.
'''

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