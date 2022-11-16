from collections import Counter, defaultdict
from typing import Dict, Iterable

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from common import *
from turk_dataset import TurkDataset


class HomebrewModel(nn.Module):
    def __init__(self, *, vocab_size: int):
        super(HomebrewModel, self).__init__()

        self.num_layers = 4
        self.embedding_size = 256
        self.hidden_size = 64

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.gru = nn.GRU(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=0.5)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        hidden_state = self.init_hidden(x.shape[1])
        x = self.embedding(x)
        output, hidden_state = self.gru(x, hidden_state)
        output = self.classifier(output[-1]).squeeze()
        return output

    def init_hidden(self, batch_size: int):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if TRAIN_ON_GPU:
            hidden = hidden.cuda()
        return hidden


def featurize_sentence(sentence: Sentence, lookup: Dict[str, int]) -> List[np.ndarray]:
    return np.array([lookup[w] for w in sentence], dtype=int)


class HomebrewDataset(Dataset):
    def __init__(self, dataset: SimplicityDataset, lookup: Dict[str, int]):
        self.sentences: List[np.ndarray] = []
        self.labels: List[int] = []

        for pairing in dataset.pairings:
            norm = pairing.norm
            self.sentences.append(featurize_sentence(norm, lookup))
            self.labels.append(1)

            for simp in pairing.simps:
                if norm == simp:
                    continue
                self.sentences.append(featurize_sentence(simp, lookup))
                self.labels.append(0)
        assert len(self.sentences) == len(self.labels)
        print(f'Created HomebrewDataset with {len(self.sentences)} sentences.')
        print(f'-> {100 * sum(self.labels) / len(self.labels):.2f}% complex')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.sentences[index], self.labels[index]


def collate_fn_pad(batch: Iterable[Tuple[np.ndarray, int]]):
    lengths = torch.tensor([t[0].shape[0] for t in batch])
    sentence_tensors = [torch.tensor(t[0], dtype=torch.long) for t in batch]
    label_tensor = torch.tensor([t[1] for t in batch], dtype=torch.float)

    sentences_tensor = torch.nn.utils.rnn.pad_sequence(sentence_tensors).long()
    return sentences_tensor, label_tensor, lengths


class HomebrewMetric(SimplicityMetric):
    LEARNING_RATE = 0.0005

    def __init__(self, tune_data: SimplicityDataset):
        counter = Counter()
        for sentence in tune_data.all:
            counter.update(sentence)
        self.lookup = defaultdict(int)
        for idx, (word, num) in enumerate(counter.most_common()):
            self.lookup[word] = idx + 1

        # Add 1 to vocab size to accommodate for zeros.
        vocab_size = len(self.lookup)+1
        self.model = HomebrewModel(vocab_size=vocab_size)

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.LEARNING_RATE)

    def predict(self, sentence: Sentence) -> float:
        self.model.eval()
        with torch.no_grad():
            feat_sentence = [featurize_sentence(sentence, self.lookup)]
            tensor = torch.tensor(feat_sentence).transpose(0, 1)
            return self.model(tensor)

    def tune(self, tune_data: SimplicityDataset, num_epochs: int = 10,
             batch_size: int = 50, print_interval: int = 100):
        tune_loader = DataLoader(HomebrewDataset(tune_data, self.lookup),
                                 shuffle=True, batch_size=batch_size,
                                 collate_fn=collate_fn_pad)
        if TRAIN_ON_GPU:
            self.model.cuda()

        self.model.train()
        for epoch in range(num_epochs):
            for i, (sentences, labels, lengths) in enumerate(tune_loader):
                outputs = self.model(sentences)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                if (i + 1) % print_interval == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Step [{i + 1}/{len(tune_loader)}], "
                        f"Loss: {loss.item():.4f}"
                    )

    def eval(self, test_data: SimplicityDataset):
        num_correct = 0

        self.model.eval()
        with torch.no_grad():
            for (sentence, label) in HomebrewDataset(test_data, self.lookup):
                tensor = torch.tensor([sentence]).transpose(0, 1)
                output = self.model(tensor)
                pred = torch.round(output)
                num_correct += bool(pred == label)

        print(f'Accuracy: {100 * num_correct / len(test_data):.4f}%')

    def __call__(self, simplicity_pair: SimplicityPair) -> float:
        norm_score = self.predict(simplicity_pair.norm)
        simp_score = self.predict(simplicity_pair.simp)
        return norm_score - simp_score


def main():
    tune_data = TurkDataset.from_tune()
    test_data = TurkDataset.from_test()

    metric = HomebrewMetric(tune_data)
    metric.tune(tune_data)
    metric.eval(test_data)

    compute_metric_main(metric, test_data)


if __name__ == '__main__':
    main()
