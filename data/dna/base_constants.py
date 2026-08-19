import os
import wandb


PROJECT_PATH = os.environ.get("PROJECT_PATH")

DATA_PATH = os.environ.get("DATA_PATH")

X3DNA_PATH = os.environ.get("X3DNA")

ETERNAFOLD_PATH = os.environ.get("ETERNAFOLD")


FILL_VALUE = 1e-5


DISTANCE_EPS = 0.001


DNA_ATOMS = [
    'P', "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    'N1', 
    'C2', 
    'O2', 'N2',
    'N3', 
    'C4', 'O4', 'N4',
    'C5', 
    'C6', 
    'O6', 'N6', 
    'N7', 
    'C8', 
    'N9',
    'OP1', 'OP2',
]


DNA_NUCLEOTIDES = [
    'A', 
    'G', 
    'C', 
    'T',
]


PURINES = ["A", "G"]


PYRIMIDINES = ["C", "T"]


LETTER_TO_NUM = dict(zip(
    DNA_NUCLEOTIDES, 
    list(range(len(DNA_NUCLEOTIDES)))
))


NUM_TO_LETTER = {v:k for k, v in LETTER_TO_NUM.items()}


DOTBRACKET_TO_NUM = {
    '.': 0,
    '(': 1,
    ')': 2
}