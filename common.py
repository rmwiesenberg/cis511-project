from abc import ABC
from pathlib import Path
from typing import Tuple, NamedTuple

import torch
from torch.utils.data import Dataset

TRAIN_ON_GPU = torch.cuda.is_available()

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'
NUM_TO_PRINT = 20


class Sentence(Tuple[str]):
    @classmethod
    def from_token_str(cls, token_str: str) -> 'Sentence':
        token_str = token_str.rstrip('\n')
        token_str = token_str.strip()
        return Sentence(token_str.split(' '))

    def __str__(self):
        return ' '.join(self)


class SimplicityPair(NamedTuple):
    norm: Sentence
    simp: Sentence


class SimplicityDataset(Dataset):
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> SimplicityPair:
        raise NotImplementedError()


class SimplicityMetric(ABC):
    def __call__(self, simplicity_pair: SimplicityPair) -> float:
        """Return a simplicity value based on the norm and simp of the pair"""
        raise NotImplementedError()


def compute_metric_main(metric: SimplicityMetric, dataset: SimplicityDataset):
    results = [metric(p) for p in dataset]
    for idx in range(NUM_TO_PRINT):
        pair = dataset[idx]
        result = results[idx]
        print(f'{idx}: {pair.norm}\n->{pair.simp}\n--> Score: {result}')




