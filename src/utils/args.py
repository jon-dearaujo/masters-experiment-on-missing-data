import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--randomness", type=int, required=True)
    return parser.parse_args()
