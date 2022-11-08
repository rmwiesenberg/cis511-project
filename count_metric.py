from common import *
from turk_dataset import TurkDataset


class CountMetric(SimplicityMetric):
    """Computes an example simplicity metric as length_diff / norm_length"""
    def __init__(self, tune_data: SimplicityDataset):
        # This metric is not learned so no need to use tune_data.
        # But I include it for copying this example file.
        pass

    def __call__(self, simplicity_pair: SimplicityPair) -> float:
        """Return a simplicity value based on the norm and simp of the pair"""
        diff = len(simplicity_pair.norm) - len(simplicity_pair.simp)
        return diff / len(simplicity_pair.norm)


def main():
    tune_data = TurkDataset.from_tune()

    test_data = TurkDataset.from_test()
    compute_metric_main(CountMetric(tune_data), test_data)


if __name__ == '__main__':
    main()
