from collections import Counter, defaultdict
from typing import Dict

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

from common import *
from turk_dataset import TurkDataset


class HomebrewModel(nn.Module):
    def __init__(
            self, *, vocab_size, embedding_dim, hidden_dim, n_layers
    ):
        super(HomebrewModel, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=0.5, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        )

        if TRAIN_ON_GPU:
            hidden = (h.cuda() for h in hidden)

        return hidden


class HomebrewDataset(Dataset):
    def __init__(self, dataset: SimplicityDataset, lookup: Dict[str, int]):
        self.x: List[np.ndarray] = []
        self.y: List[int] = []

        def featurize_sentence(sentence: Sentence) -> List[np.ndarray]:
            return np.array([lookup[w] for w in sentence], dtype=int)

        for pairing in dataset.pairings:
            self.x.append(featurize_sentence(pairing.norm))
            self.y.append(1)

            for simp in pairing.simps:
                self.x.append(featurize_sentence(simp))
                self.y.append(0)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.x[index], self.y[index]


class HomebrewMetric(SimplicityMetric):
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 32
    N_LAYERS = 4

    LEARNING_RATE = 0.001

    def __init__(self, tune_data: SimplicityDataset):
        counter = Counter()
        for sentence in tune_data.all:
            counter.update(sentence)
        self.lookup = defaultdict(int)
        for idx, word in enumerate(counter.most_common()):
            self.lookup[word] = idx + 1

        self.model = HomebrewModel(
            vocab_size=self.lookup, embedding_dim=64,
            hidden_dim=32, n_layers=4
        )

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.LEARNING_RATE)

    def tune(self, tune_data: SimplicityDataset,
             epochs: int = 4, batch_size: int = 50, print_every: int = 100):
        tune_loader = DataLoader(HomebrewDataset(tune_data, self.lookup),
                                 shuffle=True, batch_size=batch_size)

        # training params
        counter = 0
        clip = 5  # gradient clipping

        # move model to GPU, if available
        if TRAIN_ON_GPU:
            self.model.cuda()

        self.model.train()
        # train for some number of epochs
        for e in range(epochs):
            # initialize hidden state
            h = self.model.init_hidden(batch_size)

            # batch loop
            for inputs, labels in tune_loader:
                counter += 1

                if TRAIN_ON_GPU:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                self.model.zero_grad()

                # get the output from the model
                output, h = self.model(inputs, h)

                # calculate the loss and perform backprop
                loss = self.criterion(output.squeeze(), labels.float())
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = self.model.init_hidden(batch_size)
                    val_losses = []
                    self.model.eval()
                    for inputs, labels in valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if (train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    net.train()
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

    def __call__(self, simplicity_pair: SimplicityPair) -> float:
        return 0


def main():
    tune_data = TurkDataset.from_tune()

    test_data = TurkDataset.from_test()
    compute_metric_main(HomebrewMetric(tune_data), test_data)


if __name__ == '__main__':
    main()
