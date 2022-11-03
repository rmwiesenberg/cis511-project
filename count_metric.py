from common import compute_metric_main, SimplicityMetric, SimplicityPair
from turk_dataset import TurkDataset


class CountMetric(SimplicityMetric):
    """Computes an example simplicity metric as length_diff / norm_length"""
    def __call__(self, simplicity_pair: SimplicityPair) -> float:
        """Return a simplicity value based on the norm and simp of the pair"""
        diff = len(simplicity_pair.norm) - len(simplicity_pair.simp)
        return diff / len(simplicity_pair.norm)


def main():
    # This metric is not learned so no need to use tune.
    # But I include it for copying this example file.
    tune_data = TurkDataset.from_tune()

    test_data = TurkDataset.from_test()
    compute_metric_main(CountMetric(), test_data)


if __name__ == '__main__':
    main()
