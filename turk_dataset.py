from pathlib import Path
from typing import NamedTuple, List, Tuple

from common import Sentence, SimplicityPair, SimplicityDataset, DATA_DIR

TURK_DIR = DATA_DIR / 'turkcorpus'
TURK_TUNE = TURK_DIR / 'tune'
TURK_TEST = TURK_DIR / 'test'

TURK_FILE_FORMAT = '.8turkers.tok.{}'
TURK_NORM_FILE = TURK_FILE_FORMAT.format('norm')
TURK_WIKI_SIMP_FILE = TURK_FILE_FORMAT.format('simp')
TURK_TURK_SIMP_FILES = [TURK_FILE_FORMAT.format(f'turk.{i}') for i in range(8)]


class TurkData(NamedTuple):
    norm: Sentence
    wiki_simp: Sentence
    turk_simps: Tuple[Sentence]

    def simps(self) -> List[Sentence]:
        """Get all simplifications (wiki + turk)."""
        return [self.wiki_simp, *self.turk_simps]


class TurkDataset(SimplicityDataset):
    def __init__(self, turk_data: List[TurkData]):
        self.pairs = []
        for entry in turk_data:
            self.pairs.extend([SimplicityPair(entry.norm, s)
                               for s in entry.simps()])

    @staticmethod
    def sentences_from_file(filepath: Path) -> List[Sentence]:
        sentences = []
        with filepath.open(encoding='utf-8') as f:
            for token_str in f.readlines():
                sentences.append(Sentence.from_token_str(token_str))
        return sentences

    @classmethod
    def _from_x(cls, root: Path) -> 'TurkDataset':
        norm_data = cls.sentences_from_file(
            root.with_suffix(TURK_NORM_FILE)
        )
        wiki_simp_data = cls.sentences_from_file(
            root.with_suffix(TURK_WIKI_SIMP_FILE)
        )
        turk_simp_data = zip(*[
            cls.sentences_from_file(root.with_suffix(s))
            for s in TURK_TURK_SIMP_FILES
        ])
        return TurkDataset([
            TurkData(n, w, t) for n, w, t
            in zip(norm_data, wiki_simp_data, turk_simp_data)
        ])

    @classmethod
    def from_tune(cls) -> 'TurkDataset':
        return cls._from_x(TURK_TUNE)

    @classmethod
    def from_test(cls) -> 'TurkDataset':
        return cls._from_x(TURK_TEST)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> SimplicityPair:
        return self.pairs[idx]
