from collections.abc import Iterable
import os

class ResultsCollector:
    def __init__(self, epochs, randomness):
        self.results = []
        self.epochs = epochs
        self.randomness = randomness
        self.dir_name = f"../results/e{self.epochs}_r{self.randomness}"
        self.file_name = f"{self.dir_name}/result.md"


    def append_to_dir_name(self, path: str):
        return f"{self.dir_name}/{path}"

    def collect(self, content: Iterable[str]):
        with open(self.file_name, 'a') as f:
            f.writelines(content)
